import torch
import torch.nn as nn

class PatchedDecoderLayer(nn.Module):
    def __init__(self, base_layer, layer_idx, l1, l2, alpha, state_cache, debug=False):
        super().__init__()
        self.base_layer = base_layer
        self.layer_idx = layer_idx
        self.l1 = l1
        self.l2 = l2
        
        self.alpha = alpha
        self.state_cache = state_cache  # shared across layers
        self.debug = debug

    def __getattr__(self, name):
        # First, use nn.Module attribute resolution (finds registered modules/params)
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        # Fallback: delegate to the wrapped base layer via the module registry
        modules = object.__getattribute__(self, "_modules")
        base = modules.get("base_layer", None)
        if base is not None and hasattr(base, name):
            return getattr(base, name)
        raise AttributeError(f"{self.__class__.__name__} object has no attribute '{name}'")

    def forward(self, hidden_states, *args, **kwargs):
        batch_size = hidden_states.size(0)
        
        # Capture h_{t-1} at l2 (vectorized batch operation)
        if self.layer_idx == self.l2:
            # Extract last token for all batch elements at once
            last_tokens = hidden_states[:, -1, :].detach().clone()  # [batch_size, hidden_dim]
            for b in range(batch_size):
                self.state_cache[b] = last_tokens[b]
            print(f"State cache: {self.state_cache}")

        # Apply substitution at l1 (vectorized batch operation)
        if self.layer_idx == self.l1:
            # Check which batch elements have cached states
            valid_indices = [b for b in range(batch_size) if b in self.state_cache]
            
            if valid_indices:
                # Vectorized operation for all valid batch elements
                last_tokens = hidden_states[:, -1, :].clone()  # [batch_size, hidden_dim]
                prev_states = torch.stack([self.state_cache[b] for b in valid_indices])  # [num_valid, hidden_dim]
                
                valid_indices_tensor = torch.tensor(valid_indices, device=last_tokens.device)
                current_states = last_tokens[valid_indices_tensor]  # [num_valid, hidden_dim]
                
                mixed_states = (1 - self.alpha) * current_states + self.alpha * prev_states
                hidden_states[valid_indices_tensor, -1, :] = mixed_states
            
        return self.base_layer(hidden_states, *args, **kwargs)

def patch_model(model, l1, l2, alpha, debug=False):

    state_cache = {}  # shared cache across layers
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = PatchedDecoderLayer(layer, i, l1, l2 + 1, alpha, state_cache, debug=debug)
    return model