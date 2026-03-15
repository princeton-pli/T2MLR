"""
RCOT (Recurrent Chain of Thought) Package

This package provides tools for implementing continuous chain of thought in transformer models.
It includes wrappers for both individual transformer blocks and entire models to enable
recurrent connections between layers.

Main Components:
- RCOTWrapper: Wraps entire transformer models to enable continuous chain of thought
- RCOTConfig: Configuration class for RCOT models
- BlockWrapper: Wraps individual transformer blocks to enable recurrent connections
- TransformerBlockWrapperFactory: Factory for creating wrapped transformer blocks

Example Usage:
    from transformers import AutoModelForCausalLM
    from rcot_wrapper import RCOTWrapper
    from components.all_arguments import RCOTArguments

    # Load base model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-1b")
    
    # Create RCOT arguments
    rcot_args = RCOTArguments(l_start=5, l_end=10)
    
    # Wrap with RCOT
    rcot_model = RCOTWrapper.from_base_model(model, rcot_args)
    
    # Save as pretrained model
    rcot_model.save_pretrained("./my_rcot_model")
    
    # Load later
    rcot_model = RCOTWrapper.from_pretrained("./my_rcot_model")

    # Use with flow control
    flow_control = torch.tensor([[0, 1, 2, 2, 2, 3]])
    outputs = rcot_model(input_ids=input_ids, control_flows=flow_control)
"""

from .rcot_wrapper import RCOTWrapper
from .rcot_config import RCOTConfig
from .block_wrapper import BlockWrapper, apply_block_wrapper
from .recurrent_mixer import RecurrentInputMixer, RecurrentMixerConfig
from .skip_layer_inference_wrapper import (
    SkipLayerInferenceWrapper,
    SkipLayerConfig,
    wrap_model_for_skip_layer_inference,
    compute_future_token_probabilities,
)
# from .model_io_utils import load_base_model_from_config, fetch_hidden_size, load_weights_for_model
# from .model_io_utils import load_base_model_from_config, fetch_hidden_size, load_weights_for_model

__version__ = "0.1.0"
__all__ = [
    "RCOTWrapper",
    "RCOTConfig",
    "BlockWrapper",
    "apply_block_wrapper",
    "RecurrentInputMixer",
    "RecurrentMixerConfig",
    "SkipLayerInferenceWrapper",
    "SkipLayerConfig",
    "wrap_model_for_skip_layer_inference",
    "compute_future_token_probabilities",
    # "load_base_model_from_config",
    # "fetch_hidden_size",
    # "load_weights_for_model",
    # "load_base_model_from_config",
    # "fetch_hidden_size",
    # "load_weights_for_model",
]
