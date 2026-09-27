"""Native visual evidence collection and online RASE decoding."""
from dataclasses import dataclass
import math
import torch
from .candidate_vocab import ObjectCandidateVocabulary
from .features import conditional_attention, semantic_features
from .model_adapter import decoder_layers, model_type, processor_inputs
from .decoding import OnlineRollback
from .capture import install_causal_capture

def dimensions(config):
    text = config.text_config
    layers, heads = int(text.num_hidden_layers), int(text.num_attention_heads)
    total = layers*heads
    edges = total*(total-1)//2
    return dict(layers=layers, heads_per_layer=heads, total_heads=total,
                total_edges=edges, selected_edges=math.ceil(edges*.01), g_width=3*layers)

@dataclass
class SGSignal:
    conditional: torch.Tensor
    attention: torch.Tensor
    query: torch.Tensor
    token_id: int

@torch.inference_mode()
def selected_values(distributions, edge_heads, chunk=256):
    values = distributions.float().reshape(-1, distributions.shape[-1])
    roots = (values/values.sum(-1, keepdim=True)).sqrt()
    output = torch.empty(len(edge_heads), device=values.device, dtype=torch.float16)
    for i in range(0, len(edge_heads), chunk):
        pair = edge_heads[i:i+chunk]
        output[i:i+len(pair)] = (roots[pair[:, 0]]*roots[pair[:, 1]]).sum(-1).clamp(0, 1).half()
    if not torch.isfinite(output).all():
        raise ValueError('Nonfinite selected S')
    return output

class SGEvidence:
    def __init__(self, model):
        self.model = model
        self.dim = dimensions(model.config)
        self.capture = install_causal_capture(model)
        self.hidden, self.mode, self.visual, self.patches = {}, None, None, None
        self.handles = []
        for number, layer in enumerate(decoder_layers(model)):
            def hook(module, args, out, number=number):
                if self.mode is None:
                    return
                value = out[0] if isinstance(out, tuple) else out
                self.hidden[number] = (value[0].index_select(0, self.visual)
                    if self.mode == 'patch' else value[0, -1]).detach()
            self.handles.append(layer.register_forward_hook(hook))

    def before_prefill(self, visual):
        self.visual, self.patches, self.hidden, self.mode = visual, None, {}, 'patch'

    def after_prefill(self):
        self.patches = self.stack_hidden()
        self.hidden, self.mode = {}, None

    def before_token(self, token_id, want=True):
        self.token_id = int(token_id)
        self.hidden, self.mode = {}, 'token' if want else None
        self.capture.update(raw={}, attention={})

    def stack_hidden(self):
        if len(self.hidden) != self.dim['layers']:
            raise ValueError('Missing layer hidden states')
        return torch.stack([self.hidden[i] for i in range(self.dim['layers'])])

    def signal(self):
        layers = self.dim['layers']
        if len(self.capture['attention']) != layers:
            raise ValueError('Missing attention layers')
        attention = torch.stack([self.capture['attention'][i][0, :, -1].float().index_select(-1, self.visual)
                                 for i in range(layers)])
        raw = torch.stack([self.capture['raw'][i][0, :, -1].float().index_select(-1, self.visual)
                           for i in range(layers)])
        conditional, fallback = conditional_attention(attention, raw)
        result = SGSignal(conditional, attention, self.stack_hidden(), self.token_id)
        self.hidden, self.mode = {}, None
        self.capture.update(raw={}, attention={})
        return result, fallback

    @torch.inference_mode()
    def g(self, signal):
        value = semantic_features(signal.query, self.patches, signal.attention, signal.conditional,
                                  signal.token_id, self.model.model.language_model.norm,
                                  self.model.get_output_embeddings())
        if value.shape != (self.dim['g_width'],) or not torch.isfinite(value).all():
            raise ValueError('Invalid model-sized G')
        return value

    def clear(self):
        self.hidden, self.mode, self.patches = {}, None, None
        self.capture.update(raw={}, attention={})

    def close(self):
        self.clear()
        for handle in self.handles:
            handle.remove()

class OnlineRASE(OnlineRollback):
    """Collect first-subtoken evidence and apply RASE risk during decoding."""
    def __init__(self, model, processor, nlp, vocabulary_path, detector, *,
                 threshold, device, dtype, max_new_tokens, prompt):
        self.model, self.processor, self.tokenizer = model, processor, processor.tokenizer
        self.nlp, self.device, self.dtype = nlp, device, dtype
        self.candidate_vocabulary = ObjectCandidateVocabulary.from_file(vocabulary_path, nlp)
        self.layers = len(decoder_layers(model))
        self.heads_per_layer = int(model.config.text_config.num_attention_heads)
        self.detector = detector
        self.threshold, self.max_new_tokens = float(threshold), int(max_new_tokens)
        self.prompt = prompt
        self.evidence = SGEvidence(model)
        self.capture = self.evidence.capture
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def prepare_inputs(self, image, instruction):
        return self._move(dict(processor_inputs(self.processor, model_type(self.model), image, self.prompt))), self.prompt

    def _before_online_prefill(self, visual):
        self.evidence.before_prefill(visual)

    def _after_online_prefill(self):
        self.evidence.after_prefill()

    def _before_online_probe(self, token):
        self.evidence.before_token(token)

    def _online_signal(self, visual):
        return self.evidence.signal()

    def _pending_score(self, signal):
        return self.detector.score(signal, self.evidence)[0]

    def _finish_online(self):
        self.evidence.clear()
