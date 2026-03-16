from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union

from transformers import TrainingArguments as TA

@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."},
    )

    from_pretrained: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to load the model from a pretrained checkpoint."},
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional tokenizer identifier; defaults to model_name_or_path when omitted."},
    )

    depth_scaling: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use depth scaling in the model."},
    )

    attn_impl: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "help": (
                "Attention backend to use (e.g., flash_attention_2, sdpa, eager). "
                "Defaults to flash_attention_2."
            )
        },
    )

    set_hidden_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set (and from_pretrained=False / model is initialized from config), override the model "
                "config hidden size before instantiating the model."
            )
        },
    )

    # -------------------------
    # RNN baseline configuration
    # -------------------------
    rnn_hidden_size: int = field(
        default=256,
        metadata={"help": "Hidden size for `model_name_or_path=rnnlm`."},
    )
    rnn_num_layers: int = field(
        default=4,
        metadata={"help": "Number of recurrent layers for `model_name_or_path=rnnlm`."},
    )
    rnn_type: str = field(
        default="lstm",
        metadata={"help": "Recurrent cell type for `rnnlm`: one of {gru,lstm,rnn}."},
    )
    rnn_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout for `rnnlm` (applied to RNN outputs)."},
    )

    # -------------------------
    # TinyLlama (small LLaMA-style) configuration
    # -------------------------
    disable_positional_encoding: bool = field(
        default=False,
        metadata={
            "help": (
                "Only used when model_name_or_path is `tinyllama`. "
                "If True, disables positional encoding by neutralizing RoPE."
            )
        },
    )
    
    # TinyLlama architecture overrides (when model_name_or_path=tinyllama)
    tinyllama_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden size for tinyllama. None uses default (256)."},
    )
    tinyllama_num_hidden_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of hidden layers for tinyllama. None uses default (4)."},
    )
    tinyllama_num_attention_heads: Optional[int] = field(
        default=None,
        metadata={"help": "Number of attention heads for tinyllama. None uses default (4)."},
    )
    tinyllama_num_key_value_heads: Optional[int] = field(
        default=None,
        metadata={"help": "Number of key-value heads for tinyllama. None uses default (4)."},
    )
    tinyllama_intermediate_size: Optional[int] = field(
        default=None,
        metadata={"help": "Intermediate (MLP) size for tinyllama. None uses default (4 * hidden_size)."},
    )

@dataclass
class TrainingArguments(TA):
   
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed to use for training (including data shuffling)"}
    )

    do_eval: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to run evaluation"}
    )

    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to run training"}
    )

    project_name: Optional[str] = field(
        default='t2mlr',
        metadata={"help": "Name of the W&B project"}
    )

    eval_strategy: Optional[str] = field(
        default='steps',
        metadata={"help": "Mode of evaluation strategy (no, steps, epoch)"}
    )

    eval_steps: Optional[int] = field(
        default=4,
        metadata={"help": "Frequence of evaluation"}
    )

    eval_on_start: Optional[bool] = field(
        default=False,
        metadata={"help": "Eval on start of training"}
    )

    eval_generation: Optional[bool] = field(
        default=False,
        metadata={"help": "Eval on generation"}
    )

    disable_tqdm: Optional[bool] = field(
        default=True,
        metadata={"help": "Disable TQDM for cleaner logging"}
    )

    logging_first_step: Optional[bool] = field(
        default=True,
        metadata={"help": "Log metrics at the first step"}
    )

    logging_steps: Optional[int] = field(
        default=5,
        metadata={"help": "Log metrics every x steps"}
    )

    dataloader_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "Number of workers for dataloader"}
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing (Hugging Face .map)."}
    )

    shuffle_train: Optional[bool] = field(
        default=True,
        metadata={"help": "Shuffle training data"}
    )

    batch_gather_by_length: Optional[bool] = field(
        default=False,
        metadata={"help": "Gather batch by length, default is False"}
    )

    batch_gather_by_nonrecur: Optional[bool] = field(
        default=False,
        metadata={"help": "Gather batch by non-recurrence, default is False"}
    )

    use_liger_kernel: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Enable Liger kernels (https://github.com/linkedin/Liger-Kernel) when available. "
                "This is a best-effort toggle; training will fall back to standard kernels if Liger is not installed."
            )
        },
    )
 
    save_only_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Save only model. Needs to be False for sharded state dict saving."}
    )

    save_safetensors: Optional[bool] = field(
        default=False, # There is a bug in safetensors that prevents saving some models, so default to False for now
        metadata={"help": "Save model in safetensors format"}
    )

    eval_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "Number of samples to evaluate on, default is -1 (evaluate on all samples)"}
    )

    eval_print_examples: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, print a few decoded prediction examples each time evaluation runs."},
    )

    eval_print_examples_count: Optional[int] = field(
        default=2,
        metadata={"help": "Number of examples to print per evaluation when eval_print_examples is enabled."},
    )

    eval_print_max_positions: Optional[int] = field(
        default=64,
        metadata={"help": "Maximum number of token positions to show per printed example."},
    )

    eval_log_position_max_len: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional cap on the number of token positions used when aggregating eval loss by position. "
                "Defaults to data_args.max_length when available."
            )
        },
    )

    log_gate_activity: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log gate activity statistics (mean, std, min, max) during training. Adds minimal overhead by sampling one batch per logging step."}
    )

    gate_extreme_eps: float = field(
        default=0.01,
        metadata={
            "help": (
                "Threshold for 'extreme' gate values when logging gate stats. "
                "A value is considered extreme if it is <= gate_extreme_eps or >= (1 - gate_extreme_eps)."
            )
        },
    )

    gate_extreme_range_tol: float = field(
        default=0.05,
        metadata={
            "help": (
                "Tolerance for deciding whether a logged tensor is 'gate-like' (in [0,1]) "
                "before computing extreme-value shares. Values are treated as gate-like if most "
                "samples lie in [-tol, 1+tol]."
            )
        },
    )

    training_stage: str = field(
        default="sft",
        metadata={
            "help": (
                "Training routine to execute. Use 'sft' for supervised fine-tuning or 'rl' for reinforcement "
                "learning (GRPO)."
            )
        },
    )

    # Skip layer evaluation arguments
    do_skip_layer_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run skip-layer evaluation to compute future token probabilities."}
    )

    skip_layer_eval_layers_to_skip: Optional[int] = field(
        default=0,
        metadata={"help": "Number of layers to skip at the end of the model for skip-layer evaluation."}
    )

    skip_layer_eval_t2mlr_enabled: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to enable T2MLR during skip-layer evaluation. None uses model's default."}
    )

    skip_layer_eval_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Batch size for skip-layer evaluation."}
    )
    
    skip_layer_eval_num_future_tokens: Optional[int] = field(
        default=5,
        metadata={"help": "Number of future tokens to compute probabilities for in skip-layer evaluation."}
    )


@dataclass
class DataArguments:

    train_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data (JSON/JSONL/HF saved dataset)."}
    )

    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation data (JSON/JSONL/HF saved dataset)."}
    )

    train_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hugging Face dataset identifier for the training split (e.g., 'gsm8k')."}
    )

    train_dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Optional dataset configuration name (e.g., 'main' for gsm8k)."}
    )

    train_dataset_split: Optional[str] = field(
        default="train",
        metadata={"help": "Split name to load from Hugging Face when using train_dataset_name."}
    )

    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hugging Face dataset identifier for the evaluation split."}
    )

    eval_dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Optional dataset configuration name for the evaluation dataset."}
    )

    eval_dataset_split: Optional[str] = field(
        default="validation",
        metadata={"help": "Split name to load from Hugging Face when using eval_dataset_name."}
    )

    eval_holdout_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional number of examples to hold out from the training split for evaluation. "
                "When set, a randomized train/test split is created from the train dataset."
            )
        },
    )

    eval_holdout_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional ratio (0-1) of examples to hold out from the training split for evaluation. "
                "When set, a randomized train/test split is created from the train dataset."
            )
        },
    )

    eval_holdout_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Random seed to use when creating a holdout split from the training data. "
                "Defaults to TrainingArguments.seed when not provided."
            )
        },
    )

    control_flow_alias: Optional[str] = field(
        default="control_flow",
        metadata={"help": "Alias for the control flow field in the data"}
    )

    control_flow_all_recurrent: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When constructing control_flow automatically (dataset lacks it), set all positions "
                "to 2 so the entire sequence is treated as recurrent."
            )
        },
    )

    control_flow_split_answer: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When True, split response on the first '###'/'####' marker and set control_flow=2 for steps "
                "and control_flow=3 for the final answer (prompt stays 1)."
            )
        },
    )

    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum tokenized sequence length for preprocessing. When provided, samples are "
                "truncated to this length without padding."
            )
        },
    )

    padding_free: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, use a padding-free (flattened) collator that concatenates the whole mini-batch into a "
                "single sequence and provides `position_ids` (and optional FlashAttention varlen kwargs) so "
                "FlashAttention-2 can enforce correct block-diagonal attention across packed examples. "
                "NOTE: This changes the effective batch dimension to 1; throughput improves via fewer wasted pads."
            )
        },
    )

    padding_free_return_flash_attn_kwargs: bool = field(
        default=True,
        metadata={
            "help": (
                "When padding_free=True, also return FlashAttention varlen kwargs "
                "(`cu_seq_lens_q/k`, `max_length_q/k`) to the model."
            )
        },
    )

    # Pause token insertion parameters
    insert_pause_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to insert pause tokens during batching (collator-time)."}
    )

    pause_token_mean: Optional[float] = field(
        default=None,
        metadata={"help": "Mean of the per-position Poisson distribution for pause token insertion. When None, defaults to 0 (no insertion)."}
    )

    pause_token_seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed for pause token insertion (ensures deterministic caching)"}
    )

    pause_token_only_recurrent: Optional[bool] = field(
        default=True,
        metadata={"help": "Only insert pause tokens where control_flow == 2."}
    )

    pause_token_string: Optional[str] = field(
        default=None,
        metadata={"help": "Pause token string to use (overrides default Llama-3 reserved token)."}
    )

    pause_token_replace_prob: Optional[float] = field(
        default=None,
        metadata={"help": "Probability of replacing a token with a pause token at eligible positions."}
    )

    pause_token_replace_only_recurrent: Optional[bool] = field(
        default=True,
        metadata={"help": "Only replace tokens in recurrent regions (control_flow == 2) when enabled."}
    )

    pause_token_replace_prob_end: Optional[float] = field(
        default=None,
        metadata={"help": "End probability for scheduled pause token replacement."}
    )

    pause_token_replace_prob_schedule: str = field(
        default="none",
        metadata={"help": "Schedule for pause token replacement prob: none|linear."}
    )

    pause_token_replace_prob_warmup_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Warmup steps for pause token replacement prob schedule."}
    )

    pause_token_replace_prob_warmup_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Warmup ratio for pause token replacement prob schedule (if warmup steps not set)."}
    )

    concat_response_to_input: bool = field(
        default=True,
        metadata={
            "help": (
                "When True (default), training preprocessing concatenates prompt and response for SFT. "
                "When False, inputs remain prompt-only and labels are provided separately (LM-style targets)."
            )
        },
    )

    label_mask_prompt: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, set labels to ignore prompt tokens (mask with -100) while keeping prompt+response "
                "input_ids. Useful for response-only loss even when control_flow_all_recurrent is enabled."
            )
        },
    )

    label_shift: int = field(
        default=1,
        metadata={
            "help": (
                "How to align `labels` with model logits during loss/metrics. "
                "Use 1 for causal next-token alignment (default, logits[t] predicts labels[t+1] after shifting). "
                "Use 0 for per-position supervision (logits[t] compared to labels[t])."
            )
        },
    )

    custom_dataset_preprocessing: List[str] = field(
        default_factory=lambda: ["none"],
        metadata={
            "help": (
                "Custom dataset preprocessing/formatting pipeline(s) to apply after loading datasets, in order. "
                "Example: --custom_dataset_preprocessing gsm8k_aug other_proc. Use 'none' to disable."
            )
        },
    )

    custom_dataset_postprocessing: List[str] = field(
        default_factory=lambda: ["none"],
        metadata={
            "help": (
                "Custom dataset postprocessing pipeline(s) to apply after tokenization, in order. "
                "Use 'none' to disable."
            )
        },
    )

    custom_ctrl_flow_tokenization: str = field(
        default="t2mlr",
        metadata={
            "help": (
                "Custom control flow tokenization method to use. "
                "Example: --custom_ctrl_flow_tokenization t2mlr. Use 'none' to disable."
            )
        },
    )

    train_tokenized_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the tokenized cache for training"}
    )

    eval_tokenized_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the tokenized cache for evaluation"}
    )
    
@dataclass
class T2MLRArguments:

    t2mlr_enabled: Optional[bool] = field(
        default=True,
        metadata={"help": "Enable T2MLR, default is True"}
    )
    l_start: Optional[int] = field(
        default=0,
        metadata={"help": "Start layer for T2MLR (the layer that recieves recurrent information)"}
    )

    l_end: Optional[int] = field(
        default=-1,
        metadata={"help": "End layer for T2MLR (the layer that yeilds recurrent information for the next token position)"}
    )

    recurrent_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Recurrent weight for t2mlr merging, default is 1.0 (or initial value if using curriculum)"}
    )

    orig_weight: Optional[float] = field(
        default=0.0,
        metadata={"help": "Original weight for t2mlr merging, default is 0.0"}
    )

    # Curriculum learning parameters for recurrent_weight
    use_recurrent_weight_curriculum: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use curriculum learning for recurrent_weight"}
    )
    
    recurrent_weight_curriculum_start: Optional[float] = field(
        default=0.0,
        metadata={"help": "Starting value for recurrent_weight curriculum (if enabled)"}
    )
    
    recurrent_weight_curriculum_end: Optional[float] = field(
        default=1.0,
        metadata={"help": "Ending value for recurrent_weight curriculum (if enabled)"}
    )
    
    recurrent_weight_curriculum_schedule: Optional[str] = field(
        default="linear",
        metadata={"help": "Schedule type for recurrent_weight curriculum: 'linear', 'cosine', 'exponential', or 'step'"}
    )
    
    recurrent_weight_curriculum_warmup_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of warmup steps for recurrent_weight curriculum (None means use total training steps)"}
    )
    
    recurrent_weight_curriculum_warmup_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Ratio of total steps for warmup (alternative to warmup_steps, e.g., 0.1 for 10% of training)"}
    )

    batch_forward: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enable approximated batch forward for t2mlr (for efficient training / prefilling), default is False"}
    )

    batch_forward_approximate_depth: Optional[int] = field(
        default=1,
        metadata={"help": "The depth of the approximate batch forward for t2mlr (for efficient training / prefilling), default is 1"}
    )

    batch_forward_approximate_depth_values: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If set, sample BFAD (batch_forward_approximate_depth) EACH training batch. "
                "Accepts either a single max depth N (sample uniformly from MIN..N), "
                "or an explicit set of depths. Examples: "
                "'32' (=> MIN..32, where MIN defaults to 1), '8,16,32', or '[8, 16, 32]'. "
                "When provided, this overrides --batch_forward_approximate_depth at runtime."
            )
        },
    )

    batch_forward_approximate_depth_min: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Minimum value when sampling BFAD from a range. "
                "Only used when --batch_forward_approximate_depth_values is a single max value N, "
                "in which case depths are sampled from MIN..N. Defaults to 1."
            )
        },
    )

    batch_forward_approximate_depth_sampling: Optional[str] = field(
        default="uniform",
        metadata={
            "help": (
                "Sampling strategy for --batch_forward_approximate_depth_values. "
                "Currently supports: 'uniform'."
            )
        },
    )

    pre_norm_streams: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If True, apply RMS normalization to both input and recurrent streams "
                "before mixing (helps stabilize mixing by separating magnitude from importance)."
            )
        },
    )
    pre_norm_type: Optional[str] = field(
        default="rmsnorm",
        metadata={
            "help": (
                "Type of normalization for pre_norm_streams. "
                "Options: 'rmsnorm' (default) or 'layernorm'."
            )
        },
    )

    post_norm: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If True, normalize the gate output (representation being added to residual stream) "
                "using RMS norm before adding to residual (helps stabilize the contribution)."
            )
        },
    )
    post_norm_eps: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Epsilon for post_norm to avoid divide-by-zero."},
    )
    post_norm_clamp: Optional[float] = field(
        default=5.0,
        metadata={
            "help": (
                "Clamp for the normalization scale factor when post_norm=True. "
                "Effective clamp range is [1/clamp, clamp]."
            )
        },
    )

    batch_backward_approximate_depth: Optional[int] = field(
        default=10000000,
        metadata={"help": "The depth of the approximate batch backward for t2mlr (for efficient training / prefilling), default is 10000000 (infinite backward depth)"}
    )

    eval_batch_forward: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enable approximated batch forward for t2mlr during evaluation, default is False"}
    )

    # --- BFA memory optimization flags ---
    bfa_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Apply gradient checkpointing to intermediate BFA recurrent-layer "
                "passes. Trades recompute for lower peak activation memory."
            )
        },
    )
    bfa_memory_efficient_cache: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Drop intermediate cache references in l_end_output_caches during BFA. "
                "Only the latest cache is kept in Python; earlier detached caches are freed. "
                "Non-detached tensors remain alive via the autograd graph."
            )
        },
    )

    connection_detach: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient stopping for recurrent information flow, default is False"}
    )

    recurrent_residual_to_recurrent_cache: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If True, add a residual from the previous recurrent embedding (injected at l_start) "
                "into the next recurrent embedding (newly-captured l_end state used as cache for the next step), "
                "applied only for recurrent positions (control_flow > 1)."
            )
        },
    )

    recurrent_residual_to_recurrent_cache_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Scale factor for recurrent_residual_to_recurrent_cache (default 1.0)."},
    )

    recurrent_residual_to_recurrent_cache_detach: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, detach the previous recurrent embedding before adding into the next cache."},
    )

    recurrent_residual_to_recurrent_cache_post_norm: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If True, post-normalize the updated recurrent cache using RMS norm "
                "(stabilizes temporal residual accumulation)."
            )
        },
    )

    recurrent_residual_to_recurrent_cache_post_norm_eps: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Epsilon for recurrent_residual_to_recurrent_cache_post_norm."},
    )

    recurrent_residual_to_recurrent_cache_post_norm_clamp: Optional[float] = field(
        default=5.0,
        metadata={"help": "Clamp factor for recurrent_residual_to_recurrent_cache_post_norm."},
    )

    recurrent_skip_to_l_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If True, apply the skip connection to the l_end residual stream (add the injected recurrent embedding "
                "to the l_end hidden state for recurrent positions) instead of adding it during recurrent-cache update."
            )
        },
    )

    recurrent_skip_to_l_end_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Scale factor for recurrent_skip_to_l_end (default 1.0)."},
    )

    recurrent_skip_to_l_end_detach: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, detach the skip recurrent embedding before adding to l_end."},
    )

    recurrent_skip_to_l_end_post_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, RMS post-normalize the skip vector before adding to l_end."},
    )

    recurrent_skip_to_l_end_post_norm_eps: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Epsilon for recurrent_skip_to_l_end_post_norm."},
    )

    recurrent_skip_to_l_end_post_norm_clamp: Optional[float] = field(
        default=5.0,
        metadata={"help": "Clamp factor for recurrent_skip_to_l_end_post_norm."},
    )

    # Recurrent-state transform (linear vs MLP) and learnable gates for mixing
    use_recurrent_projection: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Enable a trainable transform on the recurrent state before it is mixed into the forward pass. "
                "Use --recurrent_state_proj_type to pick 'linear' vs 'mlp' (or 'auto')."
            )
        },
    )
    recurrent_projection_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Bottleneck size for the recurrent-state MLP adapter (when recurrent_state_proj_type='mlp' or when auto selects it). "
                "None => defaults to hidden_size. "
                "Back-compat: historically, setting this != hidden_size implied using the residual bottleneck MLP adapter."
            )
        },
    )
    recurrent_state_mlp_hidden_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Hidden/bottleneck dimension for the recurrent-state MLP adapter when recurrent_state_proj_type='mlp'. "
                "If set, overrides recurrent_projection_dim for the MLP path."
            )
        },
    )
    recurrent_state_mlp_num_layers: int = field(
        default=2,
        metadata={
            "help": (
                "Number of Linear layers in the recurrent-state MLP adapter when recurrent_state_proj_type='mlp'. "
                "2 means down-proj + up-proj (default). >2 adds extra bottleneck->bottleneck layers."
            )
        },
    )
    recurrent_state_mlp_activation: str = field(
        default="gelu",
        metadata={
            "help": (
                "Activation for the recurrent-state MLP adapter when recurrent_state_proj_type='mlp'. "
                "Options: gelu, relu, silu, tanh."
            )
        },
    )
    recurrent_state_mlp_dropout: float = field(
        default=0.0,
        metadata={
            "help": (
                "Dropout probability between layers of the recurrent-state MLP adapter when recurrent_state_proj_type='mlp'. "
                "Default 0.0."
            )
        },
    )
    recurrent_state_proj_type: str = field(
        default="auto",
        metadata={
            "help": (
                "Recurrent-state transform type when use_recurrent_projection=True. "
                "Options: 'auto' (back-compat: linear if recurrent_projection_dim==hidden_size else mlp), "
                "'linear' (identity-initialized Linear), 'mlp' (residual bottleneck MLP)."
            )
        },
    )
    use_learnable_gate: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable a learnable gate that scales the recurrent contribution."}
    )
    # New: independently toggle recurrent and input gates; if None fall back to use_learnable_gate for backward compatibility
    use_learnable_recurrent_gate: Optional[bool] = field(
        default=None,
        metadata={"help": "If True, recurrent gate is parameterized; if False, scalar. None => inherit use_learnable_gate."}
    )
    use_learnable_input_gate: Optional[bool] = field(
        default=None,
        metadata={"help": "If True, input gate is parameterized; if False, scalar. None => inherit use_learnable_gate."}
    )
    raise_on_nonfinite_gates: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If True, raise an error immediately when gate values contain NaN/Inf (fail-fast). "
                "If False, gate values are sanitized/clamped for numerical stability."
            )
        },
    )
    normalize_gates: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Only for recurrent_mixing_module_name='gated'. If True, renormalize learned gates so "
                "recurrent_gate + input_gate = 1 per element (helps prevent amplification when both are learned)."
            )
        },
    )
    mixing_module_kwargs: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "JSON string of extra kwargs passed to the selected mixing module constructor. "
                "Must be a *flat* dict, e.g. '{\"normalize_gates\": true}'. "
                "If any key is not supported by the selected mixing module, the run will error."
            )
        },
    )
    recurrent_gate_init: Optional[float] = field(
        default=None,  # If None, will use recurrent_weight value as default
        metadata={
            "help": (
                "Initial expected value for the recurrent gate before sigmoid (0<value<1). "
                "Lower values keep the recurrent path mostly closed at start."
            )
        },
    )
    input_gate_init: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Initial expected value for the input/original gate before sigmoid (0<value<1). "
                "Higher values keep the original path mostly open at start."
            )
        },
    )
    gate_weight_init_std: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Standard deviation for the Gaussian initialization of learnable gate projection weights. "
                "Use 0 to keep the previous zero-init behaviour."
            )
        },
    )
    gate_proj_type: str = field(
        default="linear",
        metadata={
            "help": (
                "Architecture for learnable T2MLR gates. "
                "Options: 'linear' (default) or 'mlp'. "
                "When 'mlp', the gate projection becomes an MLP and still ends with a sigmoid."
            )
        },
    )
    gate_mlp_hidden_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Hidden dimension for the gate MLP (when gate_proj_type='mlp'). "
                "If None, defaults to hidden_size."
            )
        },
    )
    gate_mlp_num_layers: int = field(
        default=2,
        metadata={
            "help": (
                "Number of Linear layers in the gate MLP (when gate_proj_type='mlp'). "
                "1 means a single linear layer (equivalent to 'linear'); 2 means one hidden layer + output."
            )
        },
    )
    gate_mlp_activation: str = field(
        default="gelu",
        metadata={
            "help": (
                "Activation to use in the gate MLP (when gate_proj_type='mlp'). "
                "Options: gelu, relu, silu, tanh."
            )
        },
    )
    gate_mlp_dropout: float = field(
        default=0.0,
        metadata={
            "help": (
                "Dropout probability between gate MLP layers (when gate_proj_type='mlp'). "
                "Default 0.0."
            )
        },
    )
    concat_recurrent: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Use concatenation instead of weighted summation to combine recurrent and input states. "
                "When enabled, concatenates recurrent_embedding and hidden_states directly (without gating), "
                "then projects back to hidden_size. Gates (recurrent_gate, input_gate) are not used in this mode."
            )
        },
    )
    recurrent_alpha: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Mixing weight for constant_weight gate: out = (1-alpha)*input + alpha*recurrent."
        },
    )
    recurrent_mixing_module_name: Optional[str] = field(
        default="none",
        metadata={
            "help": "Name of the mixing module: 'none', 'constant_weight', 'concat', or 'gated'."
        },
    )
    gate_lr_multiplier: Optional[Union[float, str]] = field(
        default=None,
        metadata={
            "help": (
                "Learning rate multiplier for gate projection parameters. "
                "If set, gate parameters will use (base_lr * gate_lr_multiplier). "
                "Example: 10.0 means gate params learn 10x faster than other params. "
                "None or 1.0 means same learning rate for all parameters."
                "When passed in as str, it will be parsed as a JSON string of a dict, e.g. '{\"gate\": 10.0, \"adapter\": 1.0}'."
            )
        },
    )
    weight_decay_exclusions: Optional[str] = field(
        default=r'["^.*\\.t2mlr_mixing_module\\.(rezero_gamma.*|gamma.*)$"]',
        metadata={
            "help": (
                "Regex patterns (JSON list or comma/space-separated string) for parameters that should "
                "receive no weight decay. Example: '[\"^.*\\\\.t2mlr_mixing_module\\\\.rezero_gamma$\"]'."
            )
        },
    )
    freeze_base_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze all base model parameters; only T2MLR adapters/gates remain trainable."}
    )


@dataclass
class GenerationEvalArguments:
    prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column name in eval dataset containing the generation prompt."}
    )

    response_column: str = field(
        default="response",
        metadata={"help": "Column name in eval dataset containing the reference response."}
    )

    eval_prompt_column: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional override for eval dataset prompt column. "
                "If unset, falls back to --prompt_column."
            )
        },
    )

    eval_response_column: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional override for eval dataset response column. "
                "If unset, falls back to --response_column."
            )
        },
    )

    reward_mode: str = field(
        default="exact",
        metadata={
            "help": (
                "Reward mode: 'math' for math_verify-based checking, 'gsm8k' for normalized answer match, "
                "'prosqa_path' for ProsQA path validation, 'pathfinding' for shortest path validation, "
                "'exact' for literal string match."
            )
        }
    )

    def get_eval_prompt_column(self) -> str:
        return (self.eval_prompt_column or self.prompt_column).strip()

    def get_eval_response_column(self) -> str:
        return (self.eval_response_column or self.response_column).strip()

    save_eval_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to persist the evaluation dataset with model outputs."}
    )

    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Override for max_new_tokens during evaluation generation."}
    )

    target_length_buffer: int = field(
        default=16,
        metadata={"help": "Additional tokens to request beyond target length when inferring generation budget."}
    )

    default_new_tokens: int = field(
        default=128,
        metadata={"help": "Fallback number of tokens to generate if target length is unknown."}
    )

    # Optional generation sampling/decoding parameters (None -> use model/generate defaults)
    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams for beam search (None leaves generate() default)."}
    )
    do_sample: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use sampling instead of greedy/beam (None leaves default)."}
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={"help": "Nucleus sampling p (None leaves default)."}
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Top-k sampling (None leaves default)."}
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Sampling temperature (None leaves default)."}
    )

    num_generations_per_sample: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of generations to produce per evaluation example. "
                "Defaults to max(pass_at_k) when omitted."
            )
        },
    )

    pass_at_k: List[int] = field(
        default_factory=lambda: [1],
        metadata={
            "help": (
                "List of k values for pass@k computation (e.g., --pass_at_k 1 5 10). "
                "Values must be positive integers."
            )
        },
    )

    task_pass_at_k: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional JSON or Python-literal mapping from task name to pass@k list, "
                "e.g. '{\"gsm8k\":[1,5], \"prosqa\":[1]}'. Overrides the global pass_at_k for listed tasks."
            )
        },
    )

    capture_gate_trace: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, record gate activations for a selected evaluation example and save them "
                "to 'gate_trace_example.json' under the generation_eval directory."
            )
        },
    )

    gate_trace_example_index: int = field(
        default=0,
        metadata={
            "help": (
                "Dataset index of the evaluation example used for gate tracing. "
                "Only the first generation for this example is captured."
            )
        },
    )

    save_all_generations: bool = field(
        default=False,
        metadata={
            "help": "If True, include all generations and rewards per example in eval output (larger files)."
        },
    )


@dataclass
class RLArguments:
    """Arguments for GRPO reinforcement-learning training."""

    model_name_or_path: str = field(
        default="",
        metadata={"help": "Path to the SFT checkpoint to initialise the policy from."},
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer to load (defaults to model_name_or_path). "
                  "Use the Instruct variant to get a chat template for base models."},
    )

    dataset_name: str = field(
        default="AI-MO/NuminaMath-CoT",
        metadata={"help": "HuggingFace dataset identifier for RL training."},
    )

    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset configuration name (e.g. 'default')."},
    )

    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for RL training."},
    )

    prompt_column: str = field(
        default="problem",
        metadata={"help": "Column containing the problem / prompt text."},
    )

    solution_column: str = field(
        default="solution",
        metadata={"help": "Column containing the ground-truth solution."},
    )

    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Maximum prompt sequence length (tokens)."},
    )

    max_completion_length: int = field(
        default=4096,
        metadata={"help": "Maximum completion sequence length (tokens)."},
    )

    num_generations: int = field(
        default=2,
        metadata={"help": "Number of completions to sample per prompt in each GRPO rollout."},
    )

    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size for the generation (rollout) phase. Auto-computed if None."},
    )

    # -- vLLM / generation backend --
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Use vLLM for rollout generation. Set False for HF-native generate()."},
    )

    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={"help": "GPU memory fraction for vLLM (only used when use_vllm=True)."},
    )

    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor-parallel size for vLLM (only used when use_vllm=True)."},
    )

    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={"help": "Enable vLLM sleep mode (only used when use_vllm=True)."},
    )

    # -- Dataset filtering --
    extract_boxed_from_solution: bool = field(
        default=True,
        metadata={"help": "Extract \\boxed{} answer from the solution column."},
    )

    filter_missing_solution: bool = field(
        default=True,
        metadata={"help": "Drop rows where the solution column is empty after extraction."},
    )

    filter_unparseable_solution: bool = field(
        default=True,
        metadata={"help": "Drop rows whose solution cannot be parsed for reward computation."},
    )

    filter_prompt_length: bool = field(
        default=True,
        metadata={"help": "Drop rows whose tokenised prompt exceeds max_prompt_length."},
    )

    custom_dataset_preprocessing: str = field(
        default="qwen_math_prompt",
        metadata={"help": "Name of the custom preprocessing pipeline to apply."},
    )

    num_proc: int = field(
        default=0,
        metadata={"help": "Number of processes for dataset map operations (0 = main process)."},
    )

    # -- Optimizer: gate LR boost (mirrors T2MLRArguments) --
    gate_lr_multiplier: Optional[Union[float, str]] = field(
        default=None,
        metadata={
            "help": (
                "Learning rate multiplier for gate parameters. "
                "Can be a single float or a JSON dict of regex→multiplier, "
                'e.g. \'{"^.*\\\\.t2mlr_mixing_module\\\\..*": 10.0}\'.'
            )
        },
    )

    weight_decay_exclusions: Optional[str] = field(
        default=r'["^.*\\.t2mlr_mixing_module\\.(rezero_gamma.*|gamma.*)$"]',
        metadata={
            "help": "Regex patterns (JSON list) for parameters excluded from weight decay."
        },
    )
