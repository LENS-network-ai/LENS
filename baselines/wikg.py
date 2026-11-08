"""
baselines/wikg.py

Adapter that plugs the official WiKG model (Li et al., "Dynamic Graph
Representation with Knowledge-aware Attention for Histopathology Whole Slide
Image Analysis", CVPR 2024) into this repo's shared training loop.

WiKG builds its own graph: it takes patch features only (no coordinates, no
adjacency) and picks each patch's top-k neighbours from a dense N x N
head/tail similarity on every forward pass. The candidate 8-NN graph LENS
uses is therefore deliberately ignored, preserving WiKG's original
graph-construction mechanism.

Upstream's model.py is loaded unmodified from baselines/WiKG_original/
(created by baselines/setup_wikg_baseline.sh). It is loaded by file path
because this repo already has a top-level `model` package.
"""

import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.abmil import _NoOpRegularizer

_WIKG_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'WiKG_original', 'model.py')


class _GlobalAttentionShim(nn.Module):
    """
    Stand-in for torch_geometric.nn.GlobalAttention (removed in newer PyG)
    for a single graph: softmax over nodes of gate_nn(x), then a weighted sum.
    """
    def __init__(self, gate_nn):
        super().__init__()
        self.gate_nn = gate_nn

    def forward(self, x, batch=None):
        gate = F.softmax(self.gate_nn(x), dim=0)
        return (gate * x).sum(dim=0, keepdim=True)


def _pyg_pooling_fallback():
    """
    Minimal torch_geometric.nn stand-in for the only three names upstream
    imports (single-graph pooling, batch=None), used when PyG is missing or
    no longer ships GlobalAttention.
    """
    import types
    stub = types.ModuleType('torch_geometric.nn')
    stub.global_mean_pool = lambda x, batch=None: x.mean(dim=-2, keepdim=True)
    stub.global_max_pool = lambda x, batch=None: x.max(dim=-2, keepdim=True)[0]
    stub.GlobalAttention = _GlobalAttentionShim
    return stub


def _load_official_wikg():
    if not os.path.isfile(_WIKG_MODEL_PATH):
        raise FileNotFoundError(
            f"Official WiKG code not found at {_WIKG_MODEL_PATH}. "
            f"Run: bash baselines/setup_wikg_baseline.sh")

    # Real PyG when present (adding GlobalAttention if a newer version
    # dropped it); otherwise a temporary stub, removed again afterwards so
    # it cannot shadow a later real `import torch_geometric`.
    injected = []
    try:
        import torch_geometric.nn as pyg_nn
        if not hasattr(pyg_nn, 'GlobalAttention'):
            pyg_nn.GlobalAttention = _GlobalAttentionShim
    except ImportError:
        import sys
        import types
        sys.modules['torch_geometric'] = types.ModuleType('torch_geometric')
        sys.modules['torch_geometric.nn'] = _pyg_pooling_fallback()
        injected = ['torch_geometric', 'torch_geometric.nn']

    try:
        spec = importlib.util.spec_from_file_location('wikg_official_model', _WIKG_MODEL_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if injected:
            import sys
            for name in injected:
                sys.modules.pop(name, None)

    # Upstream enables anomaly detection at import time, which slows every
    # backward pass substantially.
    torch.autograd.set_detect_anomaly(False)
    return module.WiKG


class WiKGBaseline(nn.Module):
    """
    Official WiKG behind the same interface as baselines/abmil.ABMIL:
      - forward(node_feat, labels, adjs, masks) -> (logits, labels, loss, adjs)
        when labels are given, so training/training_loop.py can train it
        exactly like ImprovedEdgeGNN.
      - forward(node_feat) -> (logits, Y_prob, Y_hat) for standalone inference.

    Each slide in a batch is processed on its own, with padding removed via
    `masks`: WiKG's top-k search and slide-mean mixing are not padding-aware.

    Args:
        feature_dim: patch feature dim (1536 for UNI2-h)
        hidden_dim:  WiKG hidden dim (upstream default 512)
        num_classes: number of classes
        topk:        neighbours per patch (upstream default 6)
        dropout:     message dropout (upstream default 0.3)
        agg_type:    'bi-interaction' | 'gcn' | 'sage' (upstream default)
        pool:        'mean' | 'max' | 'attn' (upstream train.py uses 'mean')
    """

    def __init__(self, feature_dim=1536, hidden_dim=512, num_classes=3, topk=6,
                 dropout=0.3, agg_type='bi-interaction', pool='mean', **kwargs):
        super().__init__()
        WiKG = _load_official_wikg()
        self.net = WiKG(dim_in=feature_dim, dim_hidden=hidden_dim, topk=topk,
                        n_classes=num_classes, agg_type=agg_type,
                        dropout=dropout, pool=pool)
        self.topk = topk
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Compatibility shims for training/training_loop.py (see abmil.py).
        self.current_epoch = 0
        self.regularizer = _NoOpRegularizer()

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_print_stats(self, value):
        pass

    def _forward_one(self, x):
        # topk cannot exceed the number of patches in the slide.
        self.net.topk = min(self.topk, x.shape[0])
        return self.net(x.unsqueeze(0))

    def forward(self, node_feat, labels=None, adjs=None, masks=None):
        single_bag = node_feat.dim() == 2
        if single_bag:
            node_feat = node_feat.unsqueeze(0)
            if masks is not None:
                masks = masks.unsqueeze(0)

        outputs = []
        for b in range(node_feat.shape[0]):
            x = node_feat[b]
            if masks is not None:
                x = x[masks[b] > 0]
            outputs.append(self._forward_one(x))
        logits = torch.cat(outputs, dim=0)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return logits, labels, loss, adjs

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)
        return logits, Y_prob, Y_hat
