"""Capture attention at the consumed object subtoken."""
import torch
from .model_adapter import model_type, repeat_kv_for_family

def install_causal_capture(model):
    """Exact single-query eager backend used by original S feature extraction."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    repeat_kv = repeat_kv_for_family(model_type(model))
    state = {'raw': {}, 'attention': {}}

    def capture(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        if query.shape[-2] != 1:
            raise ValueError('Canonical S capture requires single-query KV replay')
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        raw = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        state['raw'][int(module.layer_idx)] = raw.detach()
        weights = raw
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, :key_states.shape[-2]]
        weights = torch.softmax(weights, -1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        state['attention'][int(module.layer_idx)] = weights.detach()
        return output, weights

    ALL_ATTENTION_FUNCTIONS.register('rase_attention_capture', capture)
    return state
