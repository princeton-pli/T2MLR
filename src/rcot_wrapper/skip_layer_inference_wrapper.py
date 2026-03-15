"""
Skip Layer Inference Wrapper

A wrapper for inference that can skip a specified number of transformer layers
at the end of the model (before the LM head). Works with both RCOT enabled and disabled.

Example usage:
    from rcot_wrapper.skip_layer_inference_wrapper import SkipLayerInferenceWrapper
    
    # Wrap an existing model (RCOTWrapper or base model)
    wrapper = SkipLayerInferenceWrapper(
        model,
        num_layers_to_skip=4,  # Skip the last 4 layers before LM head
        rcot_enabled=True,     # Enable/disable RCOT
    )
    
    # Generate
    outputs = wrapper.generate(input_ids, max_new_tokens=50)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class SkipLayerConfig:
    """Configuration for the skip layer inference wrapper."""
    num_layers_to_skip: int = 0
    rcot_enabled: bool = True
    # RCOT parameters (used only if wrapping a base model, not RCOTWrapper)
    l_start: Optional[int] = None
    l_end: Optional[int] = None
    recurrent_weight: float = 0.2
    orig_weight: float = 0.8


class SkipLayerInferenceWrapper(nn.Module, GenerationMixin):
    """
    Inference wrapper that skips a specified number of layers at the end of the model.
    
    This wrapper modifies the forward pass to skip the last N transformer layers
    (excluding the LM head), which can be useful for:
    - Early exit strategies
    - Layer ablation studies
    - Reducing inference latency
    
    Works with both:
    - RCOTWrapper models (with RCOT enabled/disabled)
    - Base HuggingFace models (AutoModelForCausalLM)
    
    Args:
        model: The model to wrap (RCOTWrapper or base CausalLM model)
        num_layers_to_skip: Number of layers to skip at the end (before LM head)
        rcot_enabled: Whether to use RCOT during inference (only applies if model is RCOTWrapper)
    """
    
    main_input_name = "input_ids"
    
    def __init__(
        self,
        model: PreTrainedModel,
        num_layers_to_skip: int = 0,
        rcot_enabled: Optional[bool] = None,
    ):
        super().__init__()
        self.model = model
        self.num_layers_to_skip = num_layers_to_skip
        
        # Detect if the model is an RCOTWrapper
        self.is_rcot_wrapper = hasattr(model, 'rcot_enabled') # and hasattr(model, '_layer_container')
        
        # Set RCOT enabled state
        if rcot_enabled is not None:
            self._rcot_enabled = rcot_enabled
            if self.is_rcot_wrapper:
                # Store original and set new state
                self._original_rcot_enabled = model.rcot_enabled
                model.rcot_enabled = rcot_enabled
        else:
            self._rcot_enabled = model.rcot_enabled if self.is_rcot_wrapper else False
        
        # Resolve layer container and attributes
        self._resolve_layer_structure()
        
        # Validate skip count
        if self.num_layers_to_skip >= self.num_layers:
            raise ValueError(
                f"num_layers_to_skip ({num_layers_to_skip}) must be less than "
                f"total number of layers ({self.num_layers})"
            )
        
        # Calculate effective layer count
        self.effective_num_layers = self.num_layers - self.num_layers_to_skip
        
        # Cache for RCOT recurrent state
        self._recurrent_cache = None
        
        # Store original layers for restoration
        self._original_layers = None
        self._layers_modified = False
        
    def _resolve_layer_structure(self):
        """Resolve the transformer layer container based on model architecture."""
        # Get the actual model (unwrap RCOTWrapper if needed)
        if self.is_rcot_wrapper:
            # RCOTWrapper stores the underlying model in rcot_model
            base_model = self.model.rcot_model
        else:
            base_model = self.model
        
        # Detect layer structure (same logic for both RCOTWrapper and base models)
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            # LLaMA-style
            self._layer_container = base_model.model
            self._layer_attr_name = "layers"
        elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            # GPT-2-style
            self._layer_container = base_model.transformer
            self._layer_attr_name = "h"
        else:
            raise AttributeError(
                "Unsupported model architecture: cannot locate transformer blocks "
                "(expected model.model.layers or model.transformer.h)."
            )
        
        self.num_layers = len(getattr(self._layer_container, self._layer_attr_name))
    
    @property
    def layers(self) -> nn.ModuleList:
        """Get the transformer layers."""
        return getattr(self._layer_container, self._layer_attr_name)
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self.model.parameters()).device
    
    @property
    def config(self):
        """Get the model config."""
        return self.model.config
    
    @property
    def generation_config(self):
        """Get generation config from wrapped model."""
        return self.model.generation_config
    
    @generation_config.setter
    def generation_config(self, value):
        """Set generation config on wrapped model."""
        self.model.generation_config = value
    
    def get_input_embeddings(self):
        """Get input embeddings from wrapped model."""
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self):
        """Get output embeddings (LM head) from wrapped model."""
        return self.model.get_output_embeddings()
    
    def can_generate(self):
        """Check if the model can generate."""
        return True
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Delegate to wrapped model's prepare_inputs_for_generation."""
        if hasattr(self.model, 'prepare_inputs_for_generation'):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)
        # Fallback for models without this method
        input_ids = args[0] if args else kwargs.get('input_ids')
        return {"input_ids": input_ids}
    
    def _update_model_kwargs_for_generation(self, *args, **kwargs):
        """Delegate to wrapped model."""
        if hasattr(self.model, '_update_model_kwargs_for_generation'):
            return self.model._update_model_kwargs_for_generation(*args, **kwargs)
        return super()._update_model_kwargs_for_generation(*args, **kwargs)
    
    def _enable_layer_skipping(self):
        """Temporarily modify the model to skip layers."""
        if self._layers_modified or self.num_layers_to_skip == 0:
            return
        
        # Store original layers
        original_layers = self.layers
        self._original_layers = list(original_layers)
        
        # Create a new ModuleList with only the active layers
        active_layers = nn.ModuleList(self._original_layers[:self.effective_num_layers])
        setattr(self._layer_container, self._layer_attr_name, active_layers)
        
        self._layers_modified = True
    
    def _restore_layers(self):
        """Restore the original layers after inference."""
        if not self._layers_modified or self._original_layers is None:
            return
        
        # Restore original layers
        restored_layers = nn.ModuleList(self._original_layers)
        setattr(self._layer_container, self._layer_attr_name, restored_layers)
        
        self._original_layers = None
        self._layers_modified = False
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        control_flows: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[CausalLMOutputWithPast, Tuple]:
        """
        Forward pass with layer skipping.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Cached key-value pairs for efficient generation
            control_flows: RCOT control flow tensor (if using RCOT)
            use_cache: Whether to return key-value cache
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a ModelOutput object
            **kwargs: Additional arguments passed to the model
        
        Returns:
            Model outputs with logits and optionally cached states
        """
        original_l_end = None
        original_batch_forward = None
        try:
            # Enable layer skipping by modifying the model structure
            self._enable_layer_skipping()
            
            # Update RCOT l_end if using RCOTWrapper to account for skipped layers
            if self.is_rcot_wrapper and self._rcot_enabled:
                original_l_end = self.model.l_end
                # Adjust l_end if it exceeds the effective layer count
                if self.model.l_end >= self.effective_num_layers:
                    self.model.l_end = self.effective_num_layers - 1
                
                # IMPORTANT: Enable batch_forward for evaluation to properly compute
                # recurrent states. The default eval() mode sets batch_forward=False,
                # which uses simple_recurrent_forward that requires pre-populated
                # recurrent_cache. batch_approximate_forward builds up the recurrent
                # state progressively, which is what we need for evaluation.
                original_batch_forward = self.model.config.batch_forward
                self.model.config.batch_forward = True
            
            # Handle RCOT control flows
            if self.is_rcot_wrapper and self._rcot_enabled:
                # If no control flows provided, create default ones
                if control_flows is None:
                    batch_size, seq_len = input_ids.shape
                    # first token = 1 (no recurrence), rest = 2 (recurrence)
                    control_flows = torch.ones(batch_size, seq_len, dtype=torch.long, device=input_ids.device)
                    if seq_len > 1:
                        control_flows[:, 1:] = 2
                
                # Pass through RCOTWrapper's forward
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    control_flows=control_flows,
                    use_cache=use_cache,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs
                )
            else:
                # Regular forward pass (RCOT disabled or base model)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs
                )
            
            return outputs
            
        finally:
            # Restore l_end if modified
            if self.is_rcot_wrapper and original_l_end is not None:
                self.model.l_end = original_l_end
            # Restore batch_forward if modified
            if self.is_rcot_wrapper and original_batch_forward is not None:
                self.model.config.batch_forward = original_batch_forward
            # Restore original layers
            self._restore_layers()
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        auto_control_flow: bool = True,
        **kwargs
    ):
        """
        Generate sequences with layer skipping.
        
        Args:
            input_ids: Input token IDs
            auto_control_flow: If True and RCOT is enabled, automatically manage control flows
            **kwargs: Additional generation arguments
        
        Returns:
            Generated token IDs
        """
        try:
            # Enable layer skipping
            self._enable_layer_skipping()
            
            # Update RCOT l_end if using RCOTWrapper to account for skipped layers
            original_l_end = None
            if self.is_rcot_wrapper and self._rcot_enabled:
                original_l_end = self.model.l_end
                # Adjust l_end if it exceeds the effective layer count
                if self.model.l_end >= self.effective_num_layers:
                    self.model.l_end = self.effective_num_layers - 1
            
            # Delegate to model's generate method
            if self.is_rcot_wrapper:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    auto_control_flow=auto_control_flow and self._rcot_enabled,
                    **kwargs
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    **kwargs
                )
            
            return outputs
            
        finally:
            # Restore l_end if modified
            if self.is_rcot_wrapper and original_l_end is not None:
                self.model.l_end = original_l_end
            
            # Restore original layers
            self._restore_layers()
    
    def compute_future_token_probabilities(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        control_flows: Optional[torch.Tensor] = None,
        return_log_probs: bool = False,
        temperature: float = 1.0,
        num_future_tokens: int = 5,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the probability of future tokens using the skip-layer model.
        
        For each position i in [0, seq_len - num_future_tokens - 1], computes the 
        probability of the next `num_future_tokens` tokens using the SAME context.
        This measures how well the model can predict multiple tokens ahead from
        a single context position, done in parallel with a single forward pass.
        
        For position i, we compute (all using logits[i]):
            - P(token[i+1] | tokens[0:i+1])   from logits[i]
            - P(token[i+2] | tokens[0:i+1])   from logits[i]
            - ...
            - P(token[i+k] | tokens[0:i+1])   from logits[i]
        
        where k = num_future_tokens.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            control_flows: Optional RCOT control flows
            return_log_probs: If True, return log probabilities instead of probabilities
            temperature: Temperature for softmax (default 1.0)
            num_future_tokens: Number of future tokens to compute probabilities for (default 5)
            **kwargs: Additional arguments passed to forward
        
        Returns:
            Tuple of:
                - probs: Tensor of shape (batch_size, num_positions, num_future_tokens) with 
                         token probabilities, where num_positions = seq_len - num_future_tokens.
                         probs[:, i, k] is the probability of input_ids[:, i+k+1] given context 
                         up to position i (same context for all k).
                - target_indices: Tensor of shape (num_positions, num_future_tokens) containing 
                                  the token position indices that were evaluated.
                                  target_indices[i, k] = i + k + 1 (the position of the k-th 
                                  future token from position i).
        
        Example:
            >>> wrapper = SkipLayerInferenceWrapper(model, num_layers_to_skip=4)
            >>> input_ids = tokenizer("The capital of France is Paris", return_tensors="pt").input_ids
            >>> probs, indices = wrapper.compute_future_token_probabilities(input_ids, num_future_tokens=5)
            >>> # probs[0, i, k] = probability of token at position i+k+1 given tokens 0..i
            >>> # Expect probs for k=0 to be highest (next token), decreasing for larger k
            >>> print(f"Mean token probability: {probs.mean().item():.4f}")
            >>> print(f"Shape: {probs.shape}")  # (batch_size, seq_len - 5, 5)
        """
        batch_size, seq_len = input_ids.shape
        
        if num_future_tokens < 1:
            raise ValueError(
                f"num_future_tokens must be at least 1, got {num_future_tokens}"
            )
        
        if seq_len < num_future_tokens + 1:
            raise ValueError(
                f"input_ids must have at least {num_future_tokens + 1} tokens to compute "
                f"{num_future_tokens} future token probabilities, got seq_len={seq_len}"
            )
        
        # Single forward pass to get all logits
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                control_flows=control_flows,
                use_cache=False,
                **kwargs
            )
        
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = outputs.logits
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Compute (log) probabilities over the full vocabulary
        if return_log_probs:
            all_probs = torch.log_softmax(logits, dim=-1)
        else:
            all_probs = torch.softmax(logits, dim=-1)
        
        # Number of starting positions for which we can compute all num_future_tokens
        num_positions = seq_len - num_future_tokens
        
        # Collect probabilities for each future token offset
        # For each starting position i, we predict tokens i+1, i+2, ..., i+num_future_tokens
        # ALL using logits[i] (same context for all future token offsets)
        # probs_list[k] will have shape (batch_size, num_positions)
        probs_list = []
        indices_list = []
        
        # Use logits from positions [0, 1, ..., num_positions-1] for all predictions
        # These are the "prediction" logits at each starting position
        pred_probs = all_probs[:, :num_positions, :]  # (batch_size, num_positions, vocab_size)
        
        for k in range(num_future_tokens):
            # For the k-th future token offset:
            # - From position i, we want P(token[i+k+1] | context up to i)
            # - This uses logits[i] (already in pred_probs[:, i, :])
            # - Target token is at position i+k+1
            
            # Target tokens: positions k+1, k+2, ..., k+num_positions
            # For position i, target is input_ids[:, i+k+1]
            target_tokens = input_ids[:, k + 1:k + 1 + num_positions]  # (batch_size, num_positions)
            
            # Gather probabilities for target tokens from the SAME logits position
            target_probs_k = pred_probs.gather(
                dim=-1,
                index=target_tokens.unsqueeze(-1)
            ).squeeze(-1)  # (batch_size, num_positions)
            
            probs_list.append(target_probs_k)
            
            # Target indices: k+1, k+2, ..., k+num_positions
            target_idx = torch.arange(k + 1, k + 1 + num_positions, device=input_ids.device)
            indices_list.append(target_idx)
        
        # Stack to get final shape (batch_size, num_positions, num_future_tokens)
        target_probs = torch.stack(probs_list, dim=-1)
        
        # Stack indices to get shape (num_positions, num_future_tokens)
        target_indices = torch.stack(indices_list, dim=-1)
        
        return target_probs, target_indices
    
    def compute_sequence_perplexity(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        control_flows: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute perplexity of the sequence using the skip-layer model.
        
        Perplexity = exp(-mean(log_probs)) computed over all next-token predictions.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            control_flows: Optional RCOT control flows
            **kwargs: Additional arguments passed to forward
        
        Returns:
            Tensor of shape (batch_size,) containing perplexity for each sequence
        
        Example:
            >>> wrapper = SkipLayerInferenceWrapper(model, num_layers_to_skip=4)
            >>> ppl = wrapper.compute_sequence_perplexity(input_ids)
            >>> print(f"Perplexity with 4 layers skipped: {ppl.item():.2f}")
        """
        log_probs, _ = self.compute_future_token_probabilities(
            input_ids=input_ids,
            attention_mask=attention_mask,
            control_flows=control_flows,
            return_log_probs=True,
            **kwargs
        )
        
        # Mean log probability across tokens
        mean_log_prob = log_probs.mean(dim=-1)  # (batch_size,)
        
        # Perplexity = exp(-mean_log_prob)
        perplexity = torch.exp(-mean_log_prob)
        
        return perplexity
    
    def __repr__(self):
        return (
            f"SkipLayerInferenceWrapper(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_layers_to_skip={self.num_layers_to_skip},\n"
            f"  effective_num_layers={self.effective_num_layers},\n"
            f"  rcot_enabled={self._rcot_enabled}\n"
            f")"
        )


def wrap_model_for_skip_layer_inference(
    model: PreTrainedModel,
    num_layers_to_skip: int = 0,
    rcot_enabled: Optional[bool] = None,
) -> SkipLayerInferenceWrapper:
    """
    Convenience function to wrap a model for skip-layer inference.
    
    Args:
        model: The model to wrap (RCOTWrapper or base CausalLM)
        num_layers_to_skip: Number of layers to skip at the end
        rcot_enabled: Whether to enable RCOT (None = use model's default)
    
    Returns:
        SkipLayerInferenceWrapper instance
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from rcot_wrapper import RCOTWrapper
        >>> 
        >>> # With RCOTWrapper
        >>> base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> rcot_model = RCOTWrapper.from_base_model(base_model, rcot_args)
        >>> wrapper = wrap_model_for_skip_layer_inference(rcot_model, num_layers_to_skip=4)
        >>> 
        >>> # With base model (no RCOT)
        >>> base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> wrapper = wrap_model_for_skip_layer_inference(base_model, num_layers_to_skip=4)
    """
    return SkipLayerInferenceWrapper(
        model=model,
        num_layers_to_skip=num_layers_to_skip,
        rcot_enabled=rcot_enabled,
    )


def compute_future_token_probabilities(
    model: PreTrainedModel,
    input_ids: torch.LongTensor,
    num_layers_to_skip: int = 0,
    rcot_enabled: Optional[bool] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_log_probs: bool = False,
    temperature: float = 1.0,
    num_future_tokens: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute future token probabilities using a model with skipped layers.
    
    Convenience function that creates a temporary SkipLayerInferenceWrapper
    and computes probabilities for multiple future tokens in parallel.
    
    Args:
        model: The model to use (RCOTWrapper or base CausalLM)
        input_ids: Input token IDs of shape (batch_size, seq_len)
        num_layers_to_skip: Number of layers to skip at the end (before LM head)
        rcot_enabled: Whether to enable RCOT (None = use model's default)
        attention_mask: Optional attention mask
        return_log_probs: If True, return log probabilities instead of probabilities
        temperature: Temperature for softmax (default 1.0)
        num_future_tokens: Number of future tokens to compute probabilities for (default 5)
    
    Returns:
        Tuple of:
            - probs: Tensor of shape (batch_size, num_positions, num_future_tokens) with 
                     token probabilities, where num_positions = seq_len - num_future_tokens.
                     probs[:, i, k] is the probability of input_ids[:, i+k+1] given context 
                     up to position i (same context for all k).
            - target_indices: Tensor of shape (num_positions, num_future_tokens) with 
                              evaluated position indices.
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> 
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> 
        >>> text = "The capital of France is Paris"
        >>> input_ids = tokenizer(text, return_tensors="pt").input_ids
        >>> 
        >>> # Get probabilities for next 5 tokens with 4 layers skipped
        >>> probs, indices = compute_future_token_probabilities(
        ...     model, input_ids, num_layers_to_skip=4, num_future_tokens=5
        ... )
        >>> print(f"Mean probability: {probs.mean().item():.4f}")
        >>> print(f"Shape: {probs.shape}")  # (batch_size, seq_len - 5, 5)
        >>> # Expect probs for k=0 to be highest (next token), decreasing for larger k
        >>> 
        >>> # Compare with full model (0 layers skipped)
        >>> probs_full, _ = compute_future_token_probabilities(
        ...     model, input_ids, num_layers_to_skip=0, num_future_tokens=5
        ... )
        >>> print(f"Full model mean probability: {probs_full.mean().item():.4f}")
    """
    wrapper = SkipLayerInferenceWrapper(
        model=model,
        num_layers_to_skip=num_layers_to_skip,
        rcot_enabled=rcot_enabled,
    )
    
    return wrapper.compute_future_token_probabilities(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_log_probs=return_log_probs,
        temperature=temperature,
        num_future_tokens=num_future_tokens,
    )

