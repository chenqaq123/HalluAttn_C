"""Image-level cross-fitting of structural, semantic, and fusion models."""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score
from .learned_s import SmallSMLP, image_folds
from .detector import SemanticMLP, RiskFusion


def operating_point(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    eligible = np.flatnonzero(recall[:-1] >= .9)
    return float(thresholds[eligible[np.argmax(f1[eligible])]])


def select_edges(s, labels, fitting):
    grounded = fitting[labels[fitting] == 0]
    if not len(grounded):
        raise ValueError('Fitting images must contain grounded mentions')
    summed = np.zeros(s.shape[1], dtype=np.float64)
    for chunk in np.array_split(grounded, max(1, math.ceil(len(grounded) / 128))):
        summed += s[chunk].astype(np.float64).sum(0)
    return np.argsort(summed / len(grounded), kind='stable')[:math.ceil(.01 * s.shape[1])].copy()


def fit_model(values, labels, fitting, model, epochs, device, seed, normalize=True):
    torch.manual_seed(seed)
    model = model.to(device)
    x = torch.as_tensor(np.ascontiguousarray(values[fitting]), device=device).float()
    y = torch.as_tensor(labels[fitting], device=device, dtype=torch.float32)
    positives = y.sum()
    if not 0 < positives < len(y):
        raise ValueError('Each fitting split must contain both label classes')
    mean = x.double().mean(0).float() if normalize else torch.zeros(x.shape[1], device=device)
    scale = x.double().var(0, unbiased=False).clamp_min(1e-12).sqrt().float() if normalize else torch.ones_like(mean)
    x = (x - mean) / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    weight = (len(y) - positives) / positives
    for epoch in range(epochs):
        torch.manual_seed(seed + epoch + 1)
        model.train()
        indices = torch.randperm(len(y), device=device)
        for batch in indices.split(512):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[batch]), y[batch], pos_weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    model.eval()
    return {'mean': mean.cpu(), 'scale': scale.cpu(),
            'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def predict(payload, values, model, device):
    model = model.to(device)
    model.load_state_dict(payload['state_dict']); model.eval()
    mean, scale = payload['mean'].to(device), payload['scale'].to(device)
    with torch.inference_mode():
        result = [model((torch.as_tensor(block, device=device).float() - mean) / scale).sigmoid().cpu().numpy()
                  for block in np.array_split(values, max(1, math.ceil(len(values) / 512))) if len(block)]
    return np.concatenate(result) if result else np.empty(0)


def train(s, g, labels, image_ids, output, *, device='cpu', seed=42,
          s_epochs=7, g_epochs=7, fusion_epochs=30, backbone_model_type=None):
    """Fit on strict-upper S matrices and paper-defined G features.

    Each outer image fold receives predictions from a fusion model fitted on
    the other four folds. Its training inputs are branch predictions fitted
    excluding both the outer fold and the scored fold. The final fusion model
    uses five-fold branch predictions; deployment branches use all images.
    """
    s, g, labels = np.asarray(s), np.asarray(g), np.asarray(labels)
    image_ids = np.asarray(image_ids)
    if s.ndim != 2 or g.ndim != 2 or not (len(s) == len(g) == len(labels) == len(image_ids)):
        raise ValueError('S, G, labels, and image IDs must have matching rows')
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(s).all() or not np.isfinite(g).all():
        raise ValueError('Expected finite features and grounded=0, hallucinated=1 labels')
    if g.shape[1] % 3:
        raise ValueError('G contains three features per decoder layer')
    if min(s_epochs, g_epochs, fusion_epochs) < 1:
        raise ValueError('Training epochs must be positive')
    if backbone_model_type:
        layout = {'llava': (32,32), 'llava_next': (32,32), 'qwen2_5_vl': (28,28), 'qwen3_vl': (36,32)}
        layers, heads = layout[backbone_model_type]
        total = layers * heads
        if s.shape[1] != total * (total - 1) // 2 or g.shape[1] != 3 * layers:
            raise ValueError('Training feature dimensions differ from the selected backbone')
    folds = image_folds(image_ids, 5, seed)
    cache = {}

    def branches(included):
        key = tuple(sorted(included))
        if key in cache:
            return cache[key]
        fitting = np.flatnonzero(np.isin(folds, key))
        edges = select_edges(s, labels, fitting)
        values = s[:, edges].astype(np.float16)
        torch.manual_seed(seed)
        sp = fit_model(values, labels, fitting, SmallSMLP(len(edges)), s_epochs, device, seed)
        sp.update(schema='rase-s-mlp-v1', classifier_type='s_mlp_v1', hidden=[128,32], dropout=.1,
                  selected_edges=len(edges), total_edges=s.shape[1], selected_edge_indices=torch.from_numpy(edges),
                  object_query_offset=1, visual_token_scope='all', positive_class='hallucinated')
        if backbone_model_type:
            sp['backbone_model_type'] = backbone_model_type
        torch.manual_seed(seed)
        gp = fit_model(g, labels, fitting, SemanticMLP(g.shape[1]), g_epochs, device, seed)
        scores = np.column_stack((predict(sp, values, SmallSMLP(len(edges)), device),
                                  predict(gp, g, SemanticMLP(g.shape[1]), device)))
        cache[key] = sp, gp, scores
        return cache[key]

    oof = np.empty(len(labels), dtype=np.float32)
    for outer in range(5):
        included = [f for f in range(5) if f != outer]
        fitting = np.flatnonzero(folds != outer)
        heldout = np.flatnonzero(folds == outer)
        joint = np.zeros((len(labels), 2), dtype=np.float32)
        for fold in included:
            rows = folds == fold
            joint[rows] = branches([f for f in included if f != fold])[2][rows]
        torch.manual_seed(seed + outer)
        fp = fit_model(joint, labels, fitting, RiskFusion(), fusion_epochs, device, seed + outer, normalize=False)
        oof[heldout] = predict(fp, branches(included)[2][heldout], RiskFusion(), device)
    threshold = operating_point(labels, oof)
    joint = np.empty((len(labels), 2), dtype=np.float32)
    for fold in range(5):
        rows = folds == fold
        joint[rows] = branches([f for f in range(5) if f != fold])[2][rows]
    torch.manual_seed(seed)
    fp = fit_model(joint, labels, np.arange(len(labels)), RiskFusion(), fusion_epochs, device, seed, normalize=False)
    sp, gp, _ = branches(range(5))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.save(sp, output/'s_mlp.pt')
    torch.save(dict(schema='rase-semantic-risk-fusion-v1', g_mean=gp['mean'], g_scale=gp['scale'],
                    g_state_dict=gp['state_dict'], fusion_state_dict=fp['state_dict'],
                    threshold=threshold, s_edge_indices=sp['selected_edge_indices'],
                    s_checkpoint_sha256=hashlib.sha256((output/'s_mlp.pt').read_bytes()).hexdigest()),
               output/'semantic_fusion.pt')
    np.savez(output/'oof_scores.npz', scores=oof, labels=labels, image_ids=image_ids, folds=folds)
    return dict(threshold=threshold, training_oof_auroc=float(roc_auc_score(labels, oof)),
                images=len(np.unique(image_ids)), mentions=len(labels), s_epochs=s_epochs,
                g_epochs=g_epochs, fusion_epochs=fusion_epochs, seed=seed)
