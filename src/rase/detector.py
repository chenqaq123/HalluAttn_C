"""Structural scoring and fusion of structural and semantic risks."""
from pathlib import Path
import math
import torch
from torch import nn
from .learned_s import FrozenSDetector
from .s_relations import canonical_edge_heads


class SemanticMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, values):
        return self.net(values).squeeze(-1)


class RiskFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, risks):
        return self.linear(risks).squeeze(-1)


class StructuralDetector:
    def __init__(self, checkpoint, device='cpu'):
        payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True) if isinstance(checkpoint, (str, Path)) else checkpoint
        if payload['classifier_type'] != 's_mlp_v1' or payload['positive_class'] != 'hallucinated':
            raise ValueError('Expected a hallucination-risk S MLP checkpoint')
        if payload['object_query_offset'] != 1 or payload['visual_token_scope'] != 'all':
            raise ValueError('Expected consumed-first-subtoken, all-visual-token features')
        self.payload = payload
        self.device = torch.device(device)
        self.total_heads = (1 + math.isqrt(1 + 8 * payload['total_edges'])) // 2
        if self.total_heads * (self.total_heads - 1) // 2 != payload['total_edges']:
            raise ValueError('Invalid head graph size')
        indices = payload['selected_edge_indices'].long()
        if len(indices) != payload['selected_edges'] or len(indices.unique()) != len(indices) or indices.min() < 0 or indices.max() >= payload['total_edges']:
            raise ValueError('Invalid selected head-pair indices')
        self.edge_heads = torch.as_tensor(canonical_edge_heads(self.total_heads)[indices.numpy()], device=device, dtype=torch.long)
        self.frozen = FrozenSDetector(payload, device)

    def validate_backbone(self, config):
        text = config.text_config
        if int(text.num_hidden_layers) * int(text.num_attention_heads) != self.total_heads:
            raise ValueError('Checkpoint and backbone head counts differ')
        expected = self.payload.get('backbone_model_type')
        if expected and config.model_type != expected:
            raise ValueError('Checkpoint and backbone model families differ')

    @torch.inference_mode()
    def score_features(self, features):
        return self.frozen(features.to(self.device))

    @torch.inference_mode()
    def score_attention(self, attention):
        from .pipeline import selected_values
        if attention.shape[-2] != self.total_heads:
            raise ValueError('Attention must have [total_heads, visual_tokens] shape')
        values = selected_values(attention.to(self.device), self.edge_heads)
        return self.score_features(values)

    def score(self, signal, evidence):
        risk = float(self.score_attention(signal.conditional.reshape(self.total_heads, -1)))
        return risk, risk


class RASEDetector(StructuralDetector):
    def __init__(self, s_checkpoint, semantic_checkpoint, device='cpu'):
        super().__init__(s_checkpoint, device)
        payload = torch.load(semantic_checkpoint, map_location='cpu', weights_only=True)
        if payload['schema'] != 'rase-semantic-risk-fusion-v1':
            raise ValueError('Expected RASE semantic and two-risk fusion parameters')
        if not torch.equal(payload['s_edge_indices'], self.payload['selected_edge_indices']):
            raise ValueError('S checkpoint selection differs from the fusion training configuration')
        import hashlib
        if payload.get('s_checkpoint_sha256') != hashlib.sha256(Path(s_checkpoint).read_bytes()).hexdigest():
            raise ValueError('S checkpoint differs from the fitted fusion configuration')
        self.mean = payload['g_mean'].to(device)
        self.scale = payload['g_scale'].to(device)
        if (self.scale <= 0).any() or not torch.isfinite(self.scale).all():
            raise ValueError('Invalid semantic normalization')
        self.g = SemanticMLP(len(self.mean)).to(device)
        self.g.load_state_dict(payload['g_state_dict']); self.g.eval()
        self.fusion = RiskFusion().to(device)
        self.fusion.load_state_dict(payload['fusion_state_dict']); self.fusion.eval()
        self.threshold = float(payload['threshold'])

    @torch.inference_mode()
    def score(self, signal, evidence):
        s = self.score_attention(signal.conditional.reshape(self.total_heads, -1)).reshape(1)
        g = self.g((evidence.g(signal).to(self.device) - self.mean) / self.scale).sigmoid().reshape(1)
        return float(self.fusion(torch.cat((s, g))).sigmoid()), float(s.item())
