"""Structural normalization and the three semantic features per decoder layer."""
import math
import torch


def conditional_attention(visual_attention, raw_visual):
    values = visual_attention.float().clamp_min(0)
    conditional = values / values.sum(-1, keepdim=True).clamp_min(1e-12)
    fallback = bool((conditional.sum(-1) <= 0).any())
    if fallback:
        conditional = raw_visual.float().softmax(-1)
    return conditional, fallback


@torch.inference_mode()
def semantic_features(query_hidden, patch_hidden, visual_attention, conditional,
                      target_id, output_norm, output_head):
    """Return [top cosine mean, attention alignment, target probability] per layer.

    query_hidden: [layers, hidden]; patch_hidden: [layers, visual tokens, hidden].
    The top ceil(0.05 * visual tokens) cosine-aligned positions are shared by
    the first and third statistics. Vocabulary softmax uses the native head.
    """
    if patch_hidden.ndim != 3 or query_hidden.shape != (patch_hidden.shape[0], patch_hidden.shape[2]):
        raise ValueError('Expected aligned layer-wise query and visual states')
    layers, count, _ = patch_hidden.shape
    if count == 0 or visual_attention.shape[0] != layers or visual_attention.shape[-1] != count:
        raise ValueError('Visual states and attention must share their layer/token dimensions')
    k = max(1, math.ceil(.05 * count))
    results = []
    for layer in range(layers):
        query = torch.nn.functional.normalize(query_hidden[layer].float(), dim=-1)
        patches = torch.nn.functional.normalize(patch_hidden[layer].float(), dim=-1)
        cosine = patches @ query
        top = cosine.topk(k).indices
        attention = visual_attention[layer].float().mean(0)
        if float(attention.sum()) <= 0:
            attention = conditional[layer].float().mean(0)
        attention = attention / attention.sum()
        probabilities = []
        for chunk in top.split(16):
            logits = output_head(output_norm(patch_hidden[layer].index_select(0, chunk))).float()
            probabilities.append((logits[:, target_id] - logits.logsumexp(-1)).exp())
        results.append(torch.stack((cosine[top].mean(),
            (attention * cosine).sum() - cosine.mean(), torch.cat(probabilities).mean())))
    return torch.stack(results).flatten()
