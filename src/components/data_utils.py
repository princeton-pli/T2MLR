import hashlib
import torch
from typing import List, Dict, Any, Optional
import numpy as np

from components.all_arguments import RCOTArguments


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalize_length_value(value: Any) -> List[int]:
    """Normalize an example's `length` field into a list of ints (for tuple binning)."""
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, (float, np.floating)):
        return [int(value)]
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [int(value.item())]
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def build_length_bin_tensor(features: List[Dict[str, Any]]) -> torch.Tensor:
    """
    Build a padded integer tensor representing possibly-tuple `length` values.

    Returns:
      - shape (B,) if all values are scalar
      - shape (B, K) padded with -1 if any value is a tuple/list (K=max tuple size in batch)
    """
    normalized = [_normalize_length_value(example.get("length")) for example in features]
    max_k = max((len(v) for v in normalized), default=0)
    if max_k <= 1:
        out = torch.empty((len(features),), dtype=torch.long)
        for i, v in enumerate(normalized):
            out[i] = v[0] if v else 0
        return out

    out = torch.full((len(features), max_k), -1, dtype=torch.long)
    for i, v in enumerate(normalized):
        if not v:
            continue
        k = min(len(v), max_k)
        out[i, :k] = torch.tensor(v[:k], dtype=torch.long)
    return out


def align_rcot_steps_rear_padding(control_flow_ls):

    # Find the longest control flow
    longest_control_flow_index = int(np.argmax([len(x) for x in control_flow_ls]))
    total_length = len(control_flow_ls[longest_control_flow_index])
    offset_dict = {x: 0 for x in range(len(control_flow_ls))}
    return offset_dict, total_length

def get_labels_from_control_flow(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    for example in features:
        if 'labels' in example:
            continue
        assert len(example['input_ids']) == len(example['control_flow']), f"Input ids and control flow have different lengths: {len(example['input_ids'])} != {len(example['control_flow'])}"
        # create label such that it is equal to input_ids at control_flow > 1 and -100 otherwise
        neg_labels = torch.ones_like(example['input_ids']) * -100

        labels = torch.where(example['control_flow'] > 1, example['input_ids'], neg_labels)
        example['labels'] = labels

    return features

def cast_list_to_tensor(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for example in features:
        for key in ("input_ids", "control_flow", "labels"):
            if key in example and not isinstance(example[key], torch.Tensor):
                example[key] = torch.tensor(example[key], dtype=torch.long)
    return features


def _stable_int_seed(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        digest = hashlib.md5(value.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)
    try:
        return int(value)
    except Exception:
        return 0


def _maybe_insert_pause_tokens(
    features: List[Dict[str, Any]],
    *,
    pause_token_id: Optional[int],
    pause_token_mean: Optional[float],
    pause_token_seed: int,
    pause_token_only_recurrent: bool,
    pause_token_control_flow_value: int,
    pause_token_replace_prob: Optional[float] = None,
    pause_token_replace_only_recurrent: bool = True,
    pause_token_replace_control_flow_value: int = 2,
) -> List[Dict[str, Any]]:
    replace_prob = None if pause_token_replace_prob is None else float(pause_token_replace_prob)
    if pause_token_id is None or ((pause_token_mean is None or pause_token_mean <= 0.0) and (replace_prob is None or replace_prob <= 0.0)):
        return features

    for batch_idx, example in enumerate(features):
        if "input_ids" not in example or "control_flow" not in example:
            continue

        ids = example["input_ids"]
        cf = example["control_flow"]
        labels = example.get("labels")

        ids_list = ids.tolist() if torch.is_tensor(ids) else list(ids)
        cf_list = cf.tolist() if torch.is_tensor(cf) else list(cf)
        labels_list = None
        if labels is not None:
            labels_list = labels.tolist() if torch.is_tensor(labels) else list(labels)
            if len(labels_list) != len(ids_list):
                labels_list = None

        if pause_token_only_recurrent:
            valid_positions = [pos for pos, cf_val in enumerate(cf_list) if cf_val == pause_token_control_flow_value]
        else:
            valid_positions = list(range(len(cf_list)))
        if not valid_positions:
            continue

        seed = pause_token_seed
        for key in ("idx", "index", "id"):
            if key in example:
                seed += _stable_int_seed(example.get(key))
                break
        else:
            seed += batch_idx

        rng = np.random.default_rng(seed)
        if replace_prob is not None and replace_prob > 0.0:
            if pause_token_replace_only_recurrent:
                replace_positions = [
                    pos for pos, cf_val in enumerate(cf_list) if cf_val == pause_token_replace_control_flow_value
                ]
            else:
                replace_positions = list(range(len(cf_list)))
            for pos in replace_positions:
                if rng.random() < replace_prob:
                    ids_list[pos] = pause_token_id
                    if labels_list is not None and labels_list[pos] != -100:
                        labels_list[pos] = pause_token_id

        if pause_token_mean is None or pause_token_mean <= 0.0:
            example["input_ids"] = ids_list
            example["control_flow"] = cf_list
            if labels_list is not None:
                example["labels"] = labels_list
            continue
        offset = 0
        for pos in valid_positions:
            k = rng.poisson(pause_token_mean)
            if k <= 0:
                continue

            cf_val = cf_list[pos]
            label_ref = None
            if labels_list is not None and pos < len(labels_list):
                label_ref = labels_list[pos]

            insert_at = pos + offset
            for _ in range(int(k)):
                ids_list.insert(insert_at, pause_token_id)
                cf_list.insert(insert_at, cf_val)
                if labels_list is not None:
                    if label_ref == -100:
                        labels_list.insert(insert_at, -100)
                    else:
                        labels_list.insert(insert_at, pause_token_id)
                insert_at += 1
                offset += 1

        example["input_ids"] = ids_list
        example["control_flow"] = cf_list
        if labels_list is not None:
            example["labels"] = labels_list

    return features


class rcot_collator:

    def __init__(
            self,
            tokenizer,
            rcot_args: RCOTArguments,
            pad_prompt_left: bool = False,
            skip_labels_from_control_flow: bool = False,
            pause_token_id: Optional[int] = None,
            pause_token_mean: Optional[float] = None,
            pause_token_seed: int = 42,
            pause_token_only_recurrent: bool = True,
            pause_token_control_flow_value: int = 2,
            pause_token_replace_prob: Optional[float] = None,
            pause_token_replace_only_recurrent: bool = True,
            pause_token_replace_control_flow_value: int = 2,
        ):
        self.tokenizer = tokenizer
        self.rcot_args = rcot_args
        self.pad_prompt_left = pad_prompt_left
        self.skip_labels_from_control_flow = skip_labels_from_control_flow
        self.pause_token_id = pause_token_id
        self.pause_token_mean = pause_token_mean
        self.pause_token_seed = int(pause_token_seed)
        self.pause_token_only_recurrent = bool(pause_token_only_recurrent)
        self.pause_token_control_flow_value = int(pause_token_control_flow_value)
        self.pause_token_replace_prob = pause_token_replace_prob
        self.pause_token_replace_only_recurrent = bool(pause_token_replace_only_recurrent)
        self.pause_token_replace_control_flow_value = int(pause_token_replace_control_flow_value)
        self.preprocess_features_config: Dict[str, Any] = {}
        self.postprocess_features_config: Dict[str, Any] = {}

    def set_preprocess_features_config(self, config: Dict[str, Any]):
        self.preprocess_features_config.update(config)

    def set_postprocess_features_config(self, config: Dict[str, Any]):
        self.postprocess_features_config.update(config)

    def preprocess_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return features

    def postprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = self.preprocess_features(features)
        features = _maybe_insert_pause_tokens(
            features,
            pause_token_id=self.pause_token_id,
            pause_token_mean=self.pause_token_mean,
            pause_token_seed=self.pause_token_seed,
            pause_token_only_recurrent=self.pause_token_only_recurrent,
            pause_token_control_flow_value=self.pause_token_control_flow_value,
            pause_token_replace_prob=self.pause_token_replace_prob,
            pause_token_replace_only_recurrent=self.pause_token_replace_only_recurrent,
            pause_token_replace_control_flow_value=self.pause_token_replace_control_flow_value,
        )
        features = cast_list_to_tensor(features)
        if not self.skip_labels_from_control_flow:
            features = get_labels_from_control_flow(features)

        lengths = None
        if features and "length" in features[0]:
            lengths = build_length_bin_tensor(features)
        
        binary_control_flows = [example['control_flow'] > 1 for example in features]
        batch_size = len(binary_control_flows)

        # align the control flows
        offset_dict, total_length = align_rcot_steps_rear_padding(binary_control_flows)
        input_ids = torch.zeros((batch_size, total_length), dtype=torch.long)
        labels = torch.full((batch_size, total_length), -100, dtype=torch.long)
        control_flows = torch.zeros((batch_size, total_length), dtype=torch.int)
        attention_mask = torch.zeros((batch_size, total_length), dtype=torch.int)

        for i, example in enumerate(features):
            
            seq_len = len(example['input_ids'])
            forward_offset = offset_dict[i]
            if self.pad_prompt_left:
                forward_offset = max(total_length - seq_len, 0)

            input_ids[i, forward_offset: forward_offset + seq_len] = example['input_ids']
            if not self.skip_labels_from_control_flow:
                labels[i, forward_offset: forward_offset + seq_len] = example['labels']
            control_flows[i, forward_offset: forward_offset + seq_len] = example['control_flow']
            attention_mask[i, forward_offset: forward_offset + seq_len] = 1
        
        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        if self.rcot_args.rcot_enabled:
            batch['control_flows'] = control_flows
        if lengths is not None:
            batch["length"] = lengths

        return self.postprocess_batch(batch)


class RCOTPaddingFreeCollator:
    """
    Padding-free / packed collator for decoder-only training with FlashAttention varlen support.

    This is inspired by HF's `DataCollatorWithFlattening`, but adapted to also carry RCOT `control_flow`.

    It:
    - concatenates the entire mini-batch into a single sequence: shape [1, total_tokens]
    - emits `position_ids` that reset to 0 at each sample boundary (so FA2 can infer packed segments)
    - optionally emits FlashAttention varlen kwargs: `cu_seq_lens_q/k`, `max_length_q/k`
    - sets labels such that the first token of each packed sample is ignored (-100)
    """

    def __init__(
        self,
        *,
        return_flash_attn_kwargs: bool = True,
        separator_id: int = -100,
        pause_token_id: Optional[int] = None,
        pause_token_mean: Optional[float] = None,
        pause_token_seed: int = 42,
        pause_token_only_recurrent: bool = True,
        pause_token_control_flow_value: int = 2,
        pause_token_replace_prob: Optional[float] = None,
        pause_token_replace_only_recurrent: bool = True,
        pause_token_replace_control_flow_value: int = 2,
    ):
        self.return_flash_attn_kwargs = bool(return_flash_attn_kwargs)
        self.separator_id = int(separator_id)
        self.pause_token_id = pause_token_id
        self.pause_token_mean = pause_token_mean
        self.pause_token_seed = int(pause_token_seed)
        self.pause_token_only_recurrent = bool(pause_token_only_recurrent)
        self.pause_token_control_flow_value = int(pause_token_control_flow_value)
        self.pause_token_replace_prob = pause_token_replace_prob
        self.pause_token_replace_only_recurrent = bool(pause_token_replace_only_recurrent)
        self.pause_token_replace_control_flow_value = int(pause_token_replace_control_flow_value)
        self.preprocess_features_config: Dict[str, Any] = {}
        self.postprocess_features_config: Dict[str, Any] = {}

    def set_preprocess_features_config(self, config: Dict[str, Any]):
        self.preprocess_features_config.update(config)

    def set_postprocess_features_config(self, config: Dict[str, Any]):
        self.postprocess_features_config.update(config)

    def preprocess_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return features

    def postprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = self.preprocess_features(features)
        features = _maybe_insert_pause_tokens(
            features,
            pause_token_id=self.pause_token_id,
            pause_token_mean=self.pause_token_mean,
            pause_token_seed=self.pause_token_seed,
            pause_token_only_recurrent=self.pause_token_only_recurrent,
            pause_token_control_flow_value=self.pause_token_control_flow_value,
            pause_token_replace_prob=self.pause_token_replace_prob,
            pause_token_replace_only_recurrent=self.pause_token_replace_only_recurrent,
            pause_token_replace_control_flow_value=self.pause_token_replace_control_flow_value,
        )
        features = cast_list_to_tensor(features)
        features = get_labels_from_control_flow(features)

        # Flatten/pack across batch into one long sequence.
        flat_input_ids: List[int] = []
        flat_labels: List[int] = []
        flat_control: List[int] = []
        flat_pos: List[int] = []

        cu_seq_lens = [0]
        max_len = 0

        for seq_idx, ex in enumerate(features):
            ids = ex["input_ids"].tolist() if torch.is_tensor(ex["input_ids"]) else list(ex["input_ids"])
            labs = ex["labels"].tolist() if torch.is_tensor(ex["labels"]) else list(ex["labels"])
            cf = ex["control_flow"].tolist() if torch.is_tensor(ex["control_flow"]) else list(ex["control_flow"])

            if not ids:
                continue

            # Ensure the first token of each sample is non-recurrent to avoid RCOT cache leakage at boundaries.
            # (We also reset caches in-model when `position_ids==0`.)
            cf[0] = 1

            flat_input_ids.extend(ids)
            # Ignore label for the first token of each sample (no previous token inside that segment).
            flat_labels.append(self.separator_id)
            flat_labels.extend(labs[1:])
            flat_control.extend(cf)
            flat_pos.extend(list(range(len(ids))))

            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(ids))
                max_len = max(max_len, len(ids))

        input_ids_t = torch.tensor([flat_input_ids], dtype=torch.long)
        labels_t = torch.tensor([flat_labels], dtype=torch.long)
        control_t = torch.tensor([flat_control], dtype=torch.long)
        position_ids_t = torch.tensor([flat_pos], dtype=torch.long)

        batch = {
            "input_ids": input_ids_t,
            "labels": labels_t,
            "control_flows": control_t,
            "position_ids": position_ids_t,
        }

        if self.return_flash_attn_kwargs:
            # Transformers expects these as int32 in its own collator.
            # IMPORTANT: these must be 1D of shape (num_sequences + 1,), not [1, ...].
            # See HF `DataCollatorWithFlattening` in `transformers.data.data_collator`.
            cu = torch.tensor(cu_seq_lens, dtype=torch.int32)
            batch["cu_seq_lens_q"] = cu
            batch["cu_seq_lens_k"] = cu
            batch["max_length_q"] = int(max_len)
            batch["max_length_k"] = int(max_len)

        return self.postprocess_batch(batch)


class RCOTEvalCollator:
    """Evaluation-time collator mirroring the training collator structure."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.preprocess_features_config: Dict[str, Any] = {}
        self.postprocess_features_config: Dict[str, Any] = {}

    def set_preprocess_features_config(self, config: Dict[str, Any]):
        self.preprocess_features_config.update(config)

    def set_postprocess_features_config(self, config: Dict[str, Any]):
        self.postprocess_features_config.update(config)

    def preprocess_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return features

    def postprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = self.preprocess_features(features)

        lengths = None
        if features and "length" in features[0]:
            lengths = build_length_bin_tensor(features)

        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            prompt_pad = self.tokenizer.pad(
                {"input_ids": [example["input_ids"] for example in features]},
                padding=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        label_pad = self.tokenizer.pad(
            {"input_ids": [example["labels"] for example in features]},
            padding=True,
            return_tensors="pt",
        )

        labels = label_pad["input_ids"].masked_fill(label_pad["attention_mask"] == 0, -100)

        attention_mask = prompt_pad.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_pad["input_ids"])

        batch = {
            "input_ids": prompt_pad["input_ids"],
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if lengths is not None:
            batch["length"] = lengths

        return self.postprocess_batch(batch)

class SkipLayerEvalCollator:
    """Collate function for skip-layer evaluation."""
    
    def __init__(self, tokenizer, include_control_flows: bool = True):
        """
        Args:
            tokenizer: Tokenizer with pad_token_id
            include_control_flows: Whether to include control_flows in the batch
        """
        self.tokenizer = tokenizer
        self.include_control_flows = include_control_flows
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples for skip-layer evaluation.
        
        Args:
            batch: List of examples, each containing 'input_ids' and optionally 'control_flow'
        
        Returns:
            Dictionary with padded 'input_ids', 'attention_mask', and optionally 'control_flows'
        """
        # Extract input_ids from batch
        if isinstance(batch[0], dict):
            input_ids_list = [
                torch.tensor(ex["input_ids"]) if not isinstance(ex["input_ids"], torch.Tensor) 
                else ex["input_ids"] 
                for ex in batch
            ]
            if self.include_control_flows:
                control_flows_list = [
                    torch.tensor(ex.get("control_flow", [1] * len(ex["input_ids"])))
                    if not isinstance(ex.get("control_flow"), torch.Tensor)
                    else ex.get("control_flow", torch.ones(len(ex["input_ids"]), dtype=torch.long))
                    for ex in batch
                ]
            else:
                control_flows_list = None
        else:
            input_ids_list = [
                torch.tensor(ex) if not isinstance(ex, torch.Tensor) else ex 
                for ex in batch
            ]
            control_flows_list = None
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids_list)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        padded_input_ids = torch.full((len(input_ids_list), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
        
        for i, ids in enumerate(input_ids_list):
            padded_input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1
        
        result = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
        }
        
        if control_flows_list is not None:
            padded_control_flows = torch.ones((len(control_flows_list), max_len), dtype=torch.long)
            for i, cf in enumerate(control_flows_list):
                # Ensure control flow length matches the corresponding input_ids length
                actual_len = len(input_ids_list[i])
                cf_truncated = cf[:actual_len]  # Truncate if cf is longer than input_ids
                padded_control_flows[i, :len(cf_truncated)] = cf_truncated
            result["control_flows"] = padded_control_flows
        
        return result
