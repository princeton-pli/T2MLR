"""
Medusa-style multi-head training script.

Trains multiple LM heads to predict future tokens (t+2, t+3, ..., t+K) using
hidden representations from a specific layer of a frozen backbone model.

Key features:
- Backbone model is frozen
- Each head predicts a different future token offset (>=2 steps ahead)
- Uses cross-entropy loss for each head
- For NLP tasks like WikiText or FineWeb
"""

import os
import json
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments as HFTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset, load_from_disk

# Import RCOT components for checkpoint loading
try:
    from rcot_wrapper import RCOTWrapper
    from rcot_wrapper.rcot_config import RCOTConfig
    RCOT_AVAILABLE = True
except ImportError:
    RCOT_AVAILABLE = False
    RCOTWrapper = None
    RCOTConfig = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Arguments
# =============================================================================

@dataclass
class MedusaModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name or path (defaults to model_name_or_path)"}
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation (flash_attention_2, sdpa, eager)"}
    )
    hidden_layer_index: int = field(
        default=-1,
        metadata={"help": "Layer index to extract hidden states from. Negative indices supported (e.g., -1 = last layer before LM head)."}
    )
    num_medusa_heads: int = field(
        default=4,
        metadata={"help": "Number of Medusa heads to train (predicting tokens at offsets 2, 3, ..., num_heads+1)"}
    )
    medusa_head_hidden_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden dimension for Medusa head MLPs. If None, uses model hidden size."}
    )
    medusa_head_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers in each Medusa head MLP (1 = linear projection)"}
    )
    use_residual_connection: bool = field(
        default=True,
        metadata={"help": "Whether to use residual connection in Medusa heads (when num_layers > 1)"}
    )
    rcot_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable RCOT for the backbone model. When True, the backbone is loaded as an "
                  "RCOTWrapper and control_flows are generated for the input data."}
    )
    control_flow_all_recurrent: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When constructing control_flow automatically, set all positions "
                "to 2 so the entire sequence is treated as recurrent. "
                "If False, first token is non-recurrent (1), rest are recurrent (2)."
            )
        },
    )


@dataclass
class MedusaDataArguments:
    """Arguments for data configuration."""
    train_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the training dataset from HuggingFace hub"}
    )
    train_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local training data"}
    )
    train_dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Configuration name for training dataset"}
    )
    train_dataset_split: str = field(
        default="train",
        metadata={"help": "Split to use for training"}
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the evaluation dataset"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local evaluation data"}
    )
    eval_dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Configuration name for evaluation dataset"}
    )
    eval_dataset_split: str = field(
        default="validation",
        metadata={"help": "Split to use for evaluation"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    text_column: str = field(
        default="text",
        metadata={"help": "Name of the text column in the dataset"}
    )
    # Holdout split configuration
    eval_holdout_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of examples to holdout from train for eval (mutually exclusive with eval_holdout_ratio)"}
    )
    eval_holdout_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Fraction of train data to holdout for eval (mutually exclusive with eval_holdout_size)"}
    )
    eval_holdout_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed for holdout split (defaults to training seed)"}
    )
    # Tokenized cache paths
    train_tokenized_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load/save tokenized training dataset cache"}
    )
    eval_tokenized_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load/save tokenized evaluation dataset cache"}
    )


@dataclass
class MedusaTrainingArguments(HFTrainingArguments):
    """Extended training arguments for Medusa."""
    project_name: str = field(
        default="medusa-training",
        metadata={"help": "W&B project name"}
    )
    head_loss_weights: Optional[str] = field(
        default=None,
        metadata={"help": "JSON list of loss weights for each head. If None, uniform weighting."}
    )


# =============================================================================
# Medusa Head Module
# =============================================================================

class MedusaHead(nn.Module):
    """
    A single Medusa head that predicts a token at a specific offset.
    
    Architecture:
    - Input: hidden states from backbone layer (batch, seq_len, hidden_size)
    - MLP: Optional multi-layer transformation
    - Output: logits over vocabulary (batch, seq_len, vocab_size)
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        head_hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.use_residual = use_residual and num_layers > 1
        
        head_hidden_dim = head_hidden_dim or hidden_size
        
        if num_layers == 1:
            # Simple linear projection
            self.layers = nn.Linear(hidden_size, vocab_size)
        else:
            # Multi-layer MLP
            layers = []
            in_dim = hidden_size
            for i in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, head_hidden_dim))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = head_hidden_dim
            layers.append(nn.Linear(head_hidden_dim, vocab_size))
            self.layers = nn.Sequential(*layers)
            
            # Residual projection if dimensions differ
            if self.use_residual and hidden_size != head_hidden_dim:
                self.residual_proj = nn.Linear(hidden_size, head_hidden_dim)
            else:
                self.residual_proj = None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        if self.num_layers == 1:
            return self.layers(hidden_states)
        
        # Multi-layer forward with optional residual
        x = hidden_states
        if self.use_residual:
            # Apply residual after intermediate layers, before final projection
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x)
            # Last layer is always the output projection (no residual for that)
            logits = self.layers[-1](x)
        else:
            logits = self.layers(hidden_states)
        
        return logits


class MedusaHeads(nn.Module):
    """
    Collection of Medusa heads for multi-token prediction.
    
    Each head predicts a token at a different future offset:
    - Head 0: predicts token at position t+2
    - Head 1: predicts token at position t+3
    - ...
    - Head K-1: predicts token at position t+K+1
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
        head_hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Create individual heads
        self.heads = nn.ModuleList([
            MedusaHead(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                head_hidden_dim=head_hidden_dim,
                num_layers=num_layers,
                use_residual=use_residual,
            )
            for _ in range(num_heads)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            List of logits tensors, one per head: [(batch, seq_len, vocab_size), ...]
        """
        return [head(hidden_states) for head in self.heads]


# =============================================================================
# Model Wrapper with Frozen Backbone
# =============================================================================

class MedusaModelWrapper(nn.Module):
    """
    Wraps a frozen backbone model with trainable Medusa heads.
    
    During forward pass:
    1. Run frozen backbone to get hidden states at specified layer
    2. Pass hidden states through each Medusa head
    3. Return logits for each future token prediction
    """
    
    def __init__(
        self,
        backbone: PreTrainedModel,
        medusa_heads: MedusaHeads,
        hidden_layer_index: int = -1,
        rcot_enabled: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.medusa_heads = medusa_heads
        self.hidden_layer_index = hidden_layer_index
        self.rcot_enabled = rcot_enabled
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Get the transformer layers for hook registration
        # For RCOT models, the layers are accessed through the wrapper
        self.layers = self._get_transformer_layers()
        self.num_layers = len(self.layers)
        
        # Resolve negative indexing
        if hidden_layer_index < 0:
            self.resolved_layer_index = self.num_layers + hidden_layer_index
        else:
            self.resolved_layer_index = hidden_layer_index
        
        logger.info(f"MedusaModelWrapper initialized:")
        logger.info(f"  - Backbone: {type(backbone).__name__}")
        logger.info(f"  - Hidden size: {backbone.config.hidden_size}")
        logger.info(f"  - Num layers: {self.num_layers}")
        logger.info(f"  - Hidden layer index: {self.resolved_layer_index}")
        logger.info(f"  - Num Medusa heads: {medusa_heads.num_heads}")
        logger.info(f"  - RCOT enabled: {rcot_enabled}")
    
    def _get_transformer_layers(self) -> nn.ModuleList:
        """Extract transformer layer list from various model architectures."""
        # For RCOT models, access layers through the wrapper's layers property
        if self.rcot_enabled and hasattr(self.backbone, 'layers'):
            layers = self.backbone.layers
            if isinstance(layers, (nn.ModuleList, list)):
                logger.info(f"Found transformer layers via RCOTWrapper.layers")
                if not isinstance(layers, nn.ModuleList):
                    layers = nn.ModuleList(layers)
                return layers
        
        # Common patterns for transformer layer access
        candidates = [
            ("model.layers", lambda m: m.model.layers),
            ("transformer.h", lambda m: m.transformer.h),
            ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
            ("transformer.layers", lambda m: m.transformer.layers),
        ]
        
        # For RCOT models, also try accessing through the inner model
        if self.rcot_enabled and hasattr(self.backbone, 'rcot_model'):
            candidates = [
                ("rcot_model.model.layers", lambda m: m.rcot_model.model.layers),
            ] + candidates
        
        for name, accessor in candidates:
            try:
                layers = accessor(self.backbone)
                if isinstance(layers, nn.ModuleList):
                    logger.info(f"Found transformer layers via: {name}")
                    return layers
            except AttributeError:
                continue
        
        raise ValueError(
            f"Could not find transformer layers in model of type {type(self.backbone)}. "
            "Please add the appropriate accessor pattern."
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        control_flows: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that returns Medusa head logits.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) - used for computing per-head losses
            control_flows: (batch, seq_len) - RCOT control flows (required when rcot_enabled)
            
        Returns:
            Dict with:
                - loss: Total loss (if labels provided)
                - head_losses: Individual losses per head
                - head_logits: List of logits from each head
                - hidden_states: Captured hidden states from backbone
        """
        # Capture hidden states via forward hook
        captured_hidden = {}
        
        def capture_hook(module, inputs, outputs):
            # Handle tuple outputs (hidden_states, attention, ...)
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            captured_hidden["hidden_states"] = hidden
        
        # Register hook on target layer
        target_layer = self.layers[self.resolved_layer_index]
        hook_handle = target_layer.register_forward_hook(capture_hook)
        
        # When RCOT is enabled, temporarily enable batch_forward so the backbone
        # uses batch_approximate_forward (parallel) instead of exact_sequence_recurrent_forward
        # (one-token-at-a-time). The latter is both extremely slow and incompatible with our
        # hook-based hidden state capture (the hook fires per-token and overwrites each time).
        original_batch_forward = None
        if self.rcot_enabled and hasattr(self.backbone, 'config'):
            original_batch_forward = getattr(self.backbone.config, 'batch_forward', None)
            self.backbone.config.batch_forward = True
        
        try:
            # Build backbone kwargs
            backbone_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Pass control_flows to the backbone when RCOT is enabled
            if self.rcot_enabled and control_flows is not None:
                backbone_kwargs["control_flows"] = control_flows
            else:
                # Only pass these for non-RCOT backbones; RCOTWrapper's
                # batch_approximate_forward sets them internally
                backbone_kwargs["output_hidden_states"] = False
                backbone_kwargs["use_cache"] = False
            # Merge any remaining caller kwargs, but don't let them override
            # keys we already set (e.g. a stray output_hidden_states in kwargs).
            for k, v in kwargs.items():
                if k not in backbone_kwargs:
                    backbone_kwargs[k] = v
            
            # Forward through frozen backbone
            with torch.no_grad():
                _ = self.backbone(**backbone_kwargs)
        finally:
            hook_handle.remove()
            # Restore original batch_forward setting
            if original_batch_forward is not None:
                self.backbone.config.batch_forward = original_batch_forward
        
        # Get captured hidden states
        hidden_states = captured_hidden["hidden_states"]
        
        # Forward through Medusa heads (these are trainable)
        head_logits = self.medusa_heads(hidden_states)
        
        # Compute losses if labels provided
        loss = None
        head_losses = None
        
        if labels is not None:
            head_losses = []
            total_loss = 0.0
            num_valid_heads = 0
            seq_len = hidden_states.size(1)
            
            for head_idx, logits in enumerate(head_logits):
                # Offset for this head: head 0 predicts t+2, head 1 predicts t+3, etc.
                offset = head_idx + 2
                
                # Check if sequence is long enough for this offset
                # We need at least offset+1 positions to have valid predictions
                if seq_len <= offset:
                    # Not enough positions for this head - use zero loss
                    head_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                    head_losses.append(head_loss)
                    logger.warning(
                        f"Head {head_idx} (offset={offset}) skipped: seq_len={seq_len} <= offset. "
                        f"Consider using longer sequences or fewer heads."
                    )
                    continue
                
                # Shift logits and labels appropriately
                # logits at position t should predict token at position t+offset
                # So we compare logits[:-offset] with labels[offset:]
                shift_logits = logits[:, :-offset, :].contiguous()
                shift_labels = labels[:, offset:].contiguous()
                
                # Compute cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                head_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                head_losses.append(head_loss)
                total_loss = total_loss + head_loss
                num_valid_heads += 1
            
            # Average loss across valid heads
            if num_valid_heads > 0:
                loss = total_loss / num_valid_heads
            else:
                # No valid heads - this shouldn't happen with reasonable configs
                loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
                logger.error(
                    f"No valid heads for loss computation! seq_len={seq_len}, "
                    f"num_heads={len(head_logits)}. All offsets exceed sequence length."
                )
        
        return {
            "loss": loss,
            "head_losses": head_losses,
            "head_logits": head_logits,
            "hidden_states": hidden_states.detach(),
        }
    
    def train(self, mode: bool = True):
        """Set training mode. Backbone stays frozen/eval."""
        super().train(mode)
        # Keep backbone in eval mode always
        self.backbone.eval()
        return self


# =============================================================================
# Custom Trainer
# =============================================================================

class MedusaTrainer(Trainer):
    """Custom trainer for Medusa heads training."""
    
    def __init__(self, *args, head_loss_weights: Optional[List[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_loss_weights = head_loss_weights
        # For accumulating per-head train losses (logged via callback)
        self._train_head_loss_sums: Optional[List[float]] = None
        self._train_head_loss_counts: Optional[List[int]] = None
        # For accumulating per-head eval losses
        self._eval_head_loss_sums: Optional[List[float]] = None
        self._eval_head_loss_counts: Optional[List[int]] = None
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute weighted loss across Medusa heads."""
        labels = inputs.pop("labels", None)
        control_flows = inputs.pop("control_flows", None)
        
        forward_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=labels,
        )
        if control_flows is not None:
            forward_kwargs["control_flows"] = control_flows
        
        outputs = model(**forward_kwargs)
        
        loss = outputs["loss"]
        
        # Apply custom head weights if specified
        if self.head_loss_weights is not None and outputs["head_losses"] is not None:
            weighted_loss = sum(
                w * hl for w, hl in zip(self.head_loss_weights, outputs["head_losses"])
            )
            loss = weighted_loss / sum(self.head_loss_weights)
        
        # Accumulate per-head losses during training
        if self.model.training and outputs["head_losses"] is not None:
            if self._train_head_loss_sums is None:
                num_heads = len(outputs["head_losses"])
                self._train_head_loss_sums = [0.0] * num_heads
                self._train_head_loss_counts = [0] * num_heads
            for i, head_loss in enumerate(outputs["head_losses"]):
                self._train_head_loss_sums[i] += head_loss.item()
                self._train_head_loss_counts[i] += 1
        
        # Accumulate per-head losses during evaluation
        if not self.model.training and outputs["head_losses"] is not None:
            if self._eval_head_loss_sums is not None:
                for i, head_loss in enumerate(outputs["head_losses"]):
                    self._eval_head_loss_sums[i] += head_loss.item()
                    self._eval_head_loss_counts[i] += 1
        
        if return_outputs:
            return loss, outputs
        return loss

    def _save_checkpoint(self, model, trial):
        # Overwrite for now due to error in saving shared weights
        pass
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None, **kwargs) -> None:
        """Override log to include accumulated per-head training losses."""
        if self._train_head_loss_sums is not None and self._train_head_loss_counts is not None:
            for i in range(len(self._train_head_loss_sums)):
                if self._train_head_loss_counts[i] > 0:
                    logs[f"head_{i}_loss"] = (
                        self._train_head_loss_sums[i] / self._train_head_loss_counts[i]
                    )
            # Reset accumulators after flushing
            self._train_head_loss_sums = [0.0] * len(self._train_head_loss_sums)
            self._train_head_loss_counts = [0] * len(self._train_head_loss_counts)
        super().log(logs, start_time=start_time, **kwargs)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to compute per-head eval losses."""
        # Initialize accumulators
        num_heads = self.model.medusa_heads.num_heads
        self._eval_head_loss_sums = [0.0] * num_heads
        self._eval_head_loss_counts = [0] * num_heads
        
        # Run standard evaluation
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add per-head losses to metrics
        for i in range(num_heads):
            if self._eval_head_loss_counts[i] > 0:
                avg_loss = self._eval_head_loss_sums[i] / self._eval_head_loss_counts[i]
                metrics[f"{metric_key_prefix}_head_{i}_loss"] = avg_loss
        
        # Clean up
        self._eval_head_loss_sums = None
        self._eval_head_loss_counts = None
        
        # Log metrics (prints to stdout and sends to W&B/TensorBoard)
        self.log(metrics)
        
        return metrics


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_dataset_source(
    path: Optional[str],
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    dataset_split: str,
    role: str,
):
    """Load dataset from path or HuggingFace hub."""
    if dataset_name:
        logger.info(f"Loading {role} dataset from HuggingFace: {dataset_name} (config={dataset_config}, split={dataset_split})")
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, split=dataset_split)
        return load_dataset(dataset_name, split=dataset_split)
    
    if path:
        logger.info(f"Loading {role} dataset from path: {path}")
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json")):
            return load_from_disk(path)
        if path.endswith(".json") or path.endswith(".jsonl"):
            return load_dataset("json", data_files=path, split="train")
    
    raise ValueError(f"No valid data source for {role} dataset")


def preprocess_dataset(
    dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_column: str,
    num_proc: int = 4,
    rcot_enabled: bool = False,
    control_flow_all_recurrent: bool = False,
):
    """Tokenize and prepare dataset for language modeling.
    
    When rcot_enabled is True, generates a control_flow field for each example
    following the same convention as the main RCOT trainer:
        - control_flow_all_recurrent=True:  all positions are 2 (recurrent)
        - control_flow_all_recurrent=False: first token is 1 (non-recurrent),
          rest are 2 (recurrent)
    """
    
    def tokenize_function(examples):
        texts = examples[text_column]
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )
        
        # For language modeling, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        # Generate control flows when RCOT is enabled
        if rcot_enabled:
            control_flows = []
            for ids in tokenized["input_ids"]:
                seq_len = len(ids)
                if seq_len == 0:
                    control_flows.append([])
                elif control_flow_all_recurrent:
                    control_flows.append([2] * seq_len)
                else:
                    # First token non-recurrent, rest recurrent
                    control_flows.append([1] + [2] * (seq_len - 1))
            tokenized["control_flow"] = control_flows
        
        return tokenized
    
    # Remove all columns except what we need
    columns_to_remove = [col for col in dataset.column_names if col != text_column]
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_remove,
        desc="Tokenizing dataset",
    )
    
    # Filter out empty sequences
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        num_proc=num_proc,
    )
    
    return tokenized_dataset


# =============================================================================
# Data Collator
# =============================================================================

class MedusaDataCollator:
    """Data collator that pads sequences for Medusa training."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, rcot_enabled: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.rcot_enabled = rcot_enabled
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_control_flows = [] if self.rcot_enabled else None
        
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length
        )
        
        for feature in features:
            input_ids = feature["input_ids"][:max_len]
            labels = feature["labels"][:max_len]
            
            # Padding
            padding_length = max_len - len(input_ids)
            
            if padding_length > 0:
                input_ids = input_ids + [self.pad_token_id] * padding_length
                labels = labels + [-100] * padding_length
                attention_mask = [1] * (max_len - padding_length) + [0] * padding_length
            else:
                attention_mask = [1] * max_len
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
            
            # Handle control flows for RCOT
            if self.rcot_enabled and batch_control_flows is not None:
                cf = feature.get("control_flow", [1] * len(feature["input_ids"]))
                cf = cf[:max_len]
                # Pad control flows with 0 (non-recurrent) for padding positions
                if len(cf) < max_len:
                    cf = cf + [0] * (max_len - len(cf))
                batch_control_flows.append(cf)
        
        result = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }
        
        if batch_control_flows is not None:
            result["control_flows"] = torch.tensor(batch_control_flows, dtype=torch.long)
        
        return result


# =============================================================================
# Model Loading Utilities
# =============================================================================

def _is_rcot_checkpoint(model_path: str) -> bool:
    """Check if the given path is an RCOT checkpoint."""
    if not RCOT_AVAILABLE:
        return False
    try:
        rcot_cfg = RCOTConfig.from_pretrained(model_path)
        return getattr(rcot_cfg, "model_type", "") == "rcot"
    except Exception:
        return False


def _load_backbone_model(
    model_name_or_path: str,
    attn_impl: str = "flash_attention_2",
    torch_dtype: torch.dtype = torch.bfloat16,
    rcot_enabled: bool = False,
) -> PreTrainedModel:
    """
    Load a backbone model from a checkpoint path or HuggingFace model ID.
    
    Supports:
    - Standard HuggingFace models (e.g., meta-llama/Llama-3.2-1B)
    - RCOT checkpoints (extracts base model, or keeps wrapper if rcot_enabled)
    
    Args:
        model_name_or_path: Path to checkpoint or HuggingFace model ID
        attn_impl: Attention implementation (flash_attention_2, sdpa, eager)
        torch_dtype: Data type for model weights
        rcot_enabled: If True, keep the RCOTWrapper instead of extracting the base model
        
    Returns:
        The loaded backbone model (RCOTWrapper if rcot_enabled, else base model)
    """
    # Check if this is an RCOT checkpoint
    if _is_rcot_checkpoint(model_name_or_path):
        logger.info(f"Detected RCOT checkpoint, loading with RCOTWrapper...")
        rcot_model = RCOTWrapper.from_pretrained_with_rcot(
            model_name_or_path,
            torch_dtype=torch_dtype,
            attn_impl=attn_impl,
        )
        if rcot_enabled:
            logger.info(f"RCOT enabled: keeping RCOTWrapper as backbone")
            return rcot_model
        # Extract the underlying base model from the RCOT wrapper
        # The base model is stored as rcot_model.rcot_model
        if hasattr(rcot_model, "rcot_model"):
            backbone = rcot_model.rcot_model
            logger.info(f"Extracted base model from RCOT wrapper: {type(backbone).__name__}")
        else:
            # Fallback: use the wrapper itself (shouldn't happen normally)
            logger.warning("Could not extract base model from RCOT wrapper, using wrapper directly")
            backbone = rcot_model
        return backbone
    
    # Try loading as a standard HuggingFace model
    try:
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        logger.info(f"Loaded HuggingFace model: {type(backbone).__name__}")
        
        if rcot_enabled:
            if not RCOT_AVAILABLE:
                raise RuntimeError("RCOT is enabled but rcot_wrapper is not available. Install rcot_wrapper first.")
            logger.info("RCOT enabled: wrapping base model with RCOTWrapper")
            from components.all_arguments import RCOTArguments
            rcot_args = RCOTArguments()
            rcot_model = RCOTWrapper.from_base_model(backbone, rcot_args)
            return rcot_model
        
        return backbone
    except Exception as e:
        logger.warning(f"Standard loading failed: {e}")
        
        # Last resort: try RCOT loading if available
        if RCOT_AVAILABLE:
            logger.info("Attempting RCOT loading as fallback...")
            try:
                rcot_model = RCOTWrapper.from_pretrained_with_rcot(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    attn_impl=attn_impl,
                )
                if rcot_enabled:
                    return rcot_model
                if hasattr(rcot_model, "rcot_model"):
                    return rcot_model.rcot_model
                return rcot_model
            except Exception as rcot_e:
                logger.error(f"RCOT loading also failed: {rcot_e}")
        
        raise RuntimeError(
            f"Failed to load model from {model_name_or_path}. "
            f"Original error: {e}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    # Parse arguments
    parser = HfArgumentParser((MedusaModelArguments, MedusaDataArguments, MedusaTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup
    os.makedirs(training_args.output_dir, exist_ok=True)
    set_seed(training_args.seed)
    
    # Setup W&B
    os.environ["WANDB_PROJECT"] = training_args.project_name
    os.environ["WANDB_DIR"] = training_args.output_dir
    if training_args.run_name:
        os.environ["WANDB_NAME"] = training_args.run_name
    
    logger.info("=" * 60)
    logger.info("Medusa Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Num Medusa heads: {model_args.num_medusa_heads}")
    logger.info(f"Hidden layer index: {model_args.hidden_layer_index}")
    logger.info(f"Max length: {data_args.max_length}")
    logger.info(f"RCOT enabled: {model_args.rcot_enabled}")
    logger.info(f"Control flow all recurrent: {model_args.control_flow_all_recurrent}")
    logger.info("=" * 60)
    
    # Load tokenizer
    tokenizer_name = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load backbone model
    logger.info(f"Loading backbone model: {model_args.model_name_or_path}")
    
    attn_impl = model_args.attn_impl
    if not torch.cuda.is_available():
        attn_impl = "sdpa"
        logger.warning("CUDA not available, using SDPA attention")
    
    backbone = _load_backbone_model(
        model_name_or_path=model_args.model_name_or_path,
        attn_impl=attn_impl,
        torch_dtype=torch.bfloat16,
        rcot_enabled=model_args.rcot_enabled,
    )
    
    hidden_size = backbone.config.hidden_size
    vocab_size = backbone.config.vocab_size
    
    logger.info(f"Backbone loaded: hidden_size={hidden_size}, vocab_size={vocab_size}")
    
    # Create Medusa heads
    medusa_heads = MedusaHeads(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=model_args.num_medusa_heads,
        head_hidden_dim=model_args.medusa_head_hidden_dim,
        num_layers=model_args.medusa_head_num_layers,
        use_residual=model_args.use_residual_connection,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in medusa_heads.parameters())
    trainable_params = sum(p.numel() for p in medusa_heads.parameters() if p.requires_grad)
    logger.info(f"Medusa heads: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Create wrapper model
    model = MedusaModelWrapper(
        backbone=backbone,
        medusa_heads=medusa_heads,
        hidden_layer_index=model_args.hidden_layer_index,
        rcot_enabled=model_args.rcot_enabled,
    )
    
    # Load datasets
    train_dataset = None
    eval_dataset = None
    
    # Check for holdout configuration
    use_holdout = (data_args.eval_holdout_size is not None or data_args.eval_holdout_ratio is not None)
    if data_args.eval_holdout_size is not None and data_args.eval_holdout_ratio is not None:
        raise ValueError("Set only one of eval_holdout_size or eval_holdout_ratio, not both.")
    
    # Load training dataset
    if training_args.do_train:
        # Check cache first
        loaded_from_cache = False
        if data_args.train_tokenized_cache is not None and os.path.exists(data_args.train_tokenized_cache):
            logger.info(f"Loading tokenized training dataset from cache: {data_args.train_tokenized_cache}")
            train_dataset = load_from_disk(data_args.train_tokenized_cache)
            loaded_from_cache = True
            logger.info(f"Loaded {len(train_dataset)} training examples from cache")
        
        if not loaded_from_cache:
            logger.info("Tokenizing training dataset...")
            raw_train = load_dataset_source(
                path=data_args.train_data_path,
                dataset_name=data_args.train_dataset_name,
                dataset_config=data_args.train_dataset_config,
                dataset_split=data_args.train_dataset_split,
                role="train",
            )
            
            train_dataset = preprocess_dataset(
                raw_train,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
                text_column=data_args.text_column,
                rcot_enabled=model_args.rcot_enabled,
                control_flow_all_recurrent=model_args.control_flow_all_recurrent,
            )
            logger.info(f"Tokenized {len(train_dataset)} training examples")
            
            # Save to cache if path specified
            if data_args.train_tokenized_cache is not None:
                logger.info(f"Saving tokenized training dataset to cache: {data_args.train_tokenized_cache}")
                train_dataset.save_to_disk(data_args.train_tokenized_cache)
        
        if use_holdout and training_args.do_eval and train_dataset is not None:
            holdout_seed = data_args.eval_holdout_seed or training_args.seed
            split_kwargs = {"seed": holdout_seed, "shuffle": True}
            
            if data_args.eval_holdout_size is not None:
                split_kwargs["test_size"] = int(data_args.eval_holdout_size)
                logger.info(f"Creating holdout split with size={data_args.eval_holdout_size}")
            else:
                split_kwargs["test_size"] = float(data_args.eval_holdout_ratio)
                logger.info(f"Creating holdout split with ratio={data_args.eval_holdout_ratio}")
            
            split = train_dataset.train_test_split(**split_kwargs)
            train_dataset = split["train"]
            eval_dataset = split["test"]
            logger.info(f"Holdout split: train={len(train_dataset)}, eval={len(eval_dataset)}")
    
    # Load evaluation dataset (if not from holdout)
    if training_args.do_eval and eval_dataset is None:
        # Check cache first
        loaded_from_cache = False
        if data_args.eval_tokenized_cache is not None and os.path.exists(data_args.eval_tokenized_cache):
            logger.info(f"Loading tokenized evaluation dataset from cache: {data_args.eval_tokenized_cache}")
            eval_dataset = load_from_disk(data_args.eval_tokenized_cache)
            loaded_from_cache = True
            logger.info(f"Loaded {len(eval_dataset)} evaluation examples from cache")
        
        if not loaded_from_cache:
            logger.info("Tokenizing evaluation dataset...")
            eval_dataset_name = data_args.eval_dataset_name or data_args.train_dataset_name
            eval_dataset_config = data_args.eval_dataset_config or data_args.train_dataset_config
            raw_eval = load_dataset_source(
                path=data_args.eval_data_path,
                dataset_name=eval_dataset_name,
                dataset_config=eval_dataset_config,
                dataset_split=data_args.eval_dataset_split,
                role="eval",
            )
            eval_dataset = preprocess_dataset(
                raw_eval,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
                text_column=data_args.text_column,
                rcot_enabled=model_args.rcot_enabled,
                control_flow_all_recurrent=model_args.control_flow_all_recurrent,
            )
            logger.info(f"Tokenized {len(eval_dataset)} evaluation examples")
            
            # Save to cache if path specified
            if data_args.eval_tokenized_cache is not None:
                logger.info(f"Saving tokenized evaluation dataset to cache: {data_args.eval_tokenized_cache}")
                eval_dataset.save_to_disk(data_args.eval_tokenized_cache)
    
    # Data collator
    data_collator = MedusaDataCollator(tokenizer, data_args.max_length, rcot_enabled=model_args.rcot_enabled)
    
    # Parse head loss weights
    head_loss_weights = None
    if training_args.head_loss_weights:
        head_loss_weights = json.loads(training_args.head_loss_weights)
        logger.info(f"Using head loss weights: {head_loss_weights}")
    
    # Create trainer
    training_args.remove_unused_columns = False
    
    trainer = MedusaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        head_loss_weights=head_loss_weights,
    )
    
    # Training
    if training_args.do_train:
        logger.info("Starting Medusa heads training...")
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
    
    # Evaluation
    if training_args.do_eval and eval_dataset is not None:
        logger.info("Running evaluation...")
        metrics = trainer.evaluate()
        
        # Compute perplexity for each head
        for i in range(model_args.num_medusa_heads):
            if f"eval_head_{i}_loss" in metrics:
                metrics[f"eval_head_{i}_perplexity"] = math.exp(metrics[f"eval_head_{i}_loss"])
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Save Medusa heads separately for easy loading
    if training_args.do_train:
        medusa_path = os.path.join(training_args.output_dir, "medusa_heads.pt")
        torch.save({
            "state_dict": medusa_heads.state_dict(),
            "config": {
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "num_heads": model_args.num_medusa_heads,
                "head_hidden_dim": model_args.medusa_head_hidden_dim,
                "num_layers": model_args.medusa_head_num_layers,
                "use_residual": model_args.use_residual_connection,
            },
        }, medusa_path)
        logger.info(f"Saved Medusa heads to: {medusa_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

