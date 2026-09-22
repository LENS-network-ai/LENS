"""
Adapter around the ORIGINAL GraphLSurv implementation
(https://github.com/liupei101/GraphLSurv, cloned into
baselines/GraphLSurv_original/) for a head-to-head comparison against LENS
on the same data, splits, and evaluation protocol.

This imports GraphLSurv's actual model/loss/metric code directly (added to
sys.path below) rather than re-transcribing it, to stay faithful to the
published implementation. The only thing replaced is the input-conversion
step: their `nets.GraphLSurv.forward(data, ...)` expects a PyTorch Geometric
`Data`/`Batch` object and calls `to_dense_matrix(data, norm=True)` to get
dense (x, adj, mask) tensors -- we already have those directly from
utils.survival_dataset.prepare_survival_batch, so `graphlsurv_forward` below
reproduces the rest of their forward pass verbatim (same submodules, same
pooling, same risk clamping) without that PyG round-trip. `to_dense_matrix`
also adds self-loops to the adjacency before normalizing, which is
reproduced explicitly here since our adj_s.pt files don't include them.
"""

import os
import sys

import torch
import torch.nn.functional as F

_ORIGINAL_REPO_MODELING = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'GraphLSurv_original', 'S02-modeling'
)

# GraphLSurv_original/S02-modeling has its own flat modules named nets.py,
# nloss.py, layers.py, utils.py. The last one collides with LENS's own
# utils/ package: sys.modules caches by bare name, so importing their flat
# `utils` module under the name "utils" would make ANY later `import utils`
# anywhere in this process -- including LENS's own utils/ package imports --
# resolve to their flat module instead. Import in an isolated block: stash
# any existing "utils" cache entry, load their modules with the path
# temporarily prepended, then evict their cached entries and restore
# whatever was there before, so nothing leaks into the rest of the process.
_stashed_utils_module = sys.modules.pop('utils', None)
sys.path.insert(0, _ORIGINAL_REPO_MODELING)
try:
    from nets import GraphLSurv  # noqa: E402  (original repo's model, imported verbatim)
    from nloss import nlog_partial_likelihood  # noqa: E402  (original repo's Cox loss, imported verbatim)
    from utils import batch_normalize_adj  # noqa: E402  (original repo's adjacency normalization)
    from utils import concordance_index as graphlsurv_concordance_index  # noqa: E402
finally:
    sys.path.remove(_ORIGINAL_REPO_MODELING)
    for _name in ('utils', 'nets', 'nloss', 'layers'):
        sys.modules.pop(_name, None)
    if _stashed_utils_module is not None:
        sys.modules['utils'] = _stashed_utils_module


def build_graphlsurv(feature_dim, hidden_dim=256, num_layers=1, dropout_ratio=0.25,
                      ratio_anchors=0.2, epsilon=0.9, ratio_init_graph=0.2, graph_hops=2):
    """
    out_dim=1: GraphLSurv predicts a single continuous Cox risk score, not
    discrete hazard bins (unlike LENS's survival head) -- this matches its
    original nlog_partial_likelihood (Cox partial-likelihood) training
    objective, so each method is compared using its own validated loss,
    with concordance index as the common, loss-agnostic metric.

    metric_type must be 'transformer' -- the original AnchorGraphLearner
    only implements that branch; 'weighted_cosine' (used in some derivative
    re-implementations) raises NotImplementedError in the original source.
    """
    args_glearner = {
        'hid_dim': hidden_dim, 'ratio_anchors': ratio_anchors,
        'epsilon': epsilon, 'topk': None, 'metric_type': 'transformer',
    }
    args_gencoder = {
        'graph_hops': graph_hops, 'ratio_init_graph': ratio_init_graph,
        'dropout_ratio': dropout_ratio, 'batch_norm': False,
    }
    return GraphLSurv(
        in_dim=feature_dim, hid_dim=hidden_dim, out_dim=1,
        num_layers=num_layers, dropout_ratio=dropout_ratio,
        args_glearner=args_glearner, args_gencoder=args_gencoder,
    )


def graphlsurv_forward(model, node_feat, adjs, masks, return_retention=False):
    """
    Reproduces nets.GraphLSurv.forward's body exactly, starting from dense
    (node_feat, adjs, masks) tensors instead of a PyG Data object.

    Note: this always uses the FULL, unweighted original adjacency (`adjs`)
    for one of its two message-passing pathways -- GraphLSurv never prunes
    the original candidate graph itself, unlike LENS. `node_anchor_adj` is a
    separate, dynamically-sampled node-to-anchor bipartite structure (only
    ~ratio_anchors of nodes sampled as anchors), already thresholded by
    AnchorGraphLearner's epsilon cutoff. Its post-threshold sparsity is the
    closest thing this architecture has to a "retention" statistic, but it
    is not the same quantity as "% of original edges kept" -- report it
    labeled as node-anchor attention sparsity, not edge retention.

    If return_retention=True, also returns that sparsity fraction (computed
    from the last layer's node_anchor_adj/anchor_mask, restricted to valid
    node/anchor positions so padding doesn't skew it).
    """
    n = adjs.size(-1)
    self_loops = torch.eye(n, device=adjs.device, dtype=adjs.dtype).unsqueeze(0)
    adj_with_self_loops = (adjs + self_loops).clamp(max=1.0)
    init_adj = batch_normalize_adj(adj_with_self_loops, mask=masks)

    prev_x = node_feat
    node_vec = prev_x
    node_anchor_adj, anchor_mask = None, None
    for net_glearner, net_encoder in zip(model.net_glearners, model.net_encoders):
        node_anchor_adj, _, _, anchor_mask = net_glearner(prev_x, masks)
        node_vec = net_encoder(prev_x, init_adj, node_anchor_adj)
        prev_x = node_vec

    out_max = model.graph_pool(node_vec, masks, 'max')
    out_avg = model.graph_pool(node_vec, masks, 'mean')
    out = torch.cat([out_max, out_avg], dim=1)

    out = F.relu(model.lin1(out))
    out = F.dropout(out, p=model.dropout_ratio, training=model.training)
    out = F.relu(model.lin2(out))
    out = model.lin3(out)
    out = torch.where(out > model.MAX_RISK, model.MAX_RISK, out)

    if not return_retention:
        return out

    # Fraction of node-anchor pairs surviving AnchorGraphLearner's epsilon
    # cutoff, restricted to valid (real node, real anchor) positions --
    # NOT the same quantity as LENS's "% of original edges kept" (see
    # docstring above), but the closest analogous statistic this
    # architecture produces.
    valid_pair_mask = masks.unsqueeze(-1) * anchor_mask.float().unsqueeze(1)  # [B, N, num_anchors]
    kept = ((node_anchor_adj != 0).float() * valid_pair_mask).sum()
    total = valid_pair_mask.sum().clamp(min=1.0)
    retention = (kept / total).item()

    return out, retention


def to_cox_label(survival_months, censorship):
    """
    GraphLSurv's label convention (nloss.py, utils.py): abs(y) = observed
    time, sign(y) = event indicator -- positive means the event (death)
    was observed, negative means right-censored. Our censorship field uses
    the opposite polarity (1 = censored, 0 = event), so this flips it.
    """
    event_occurred = (censorship == 0).float()
    sign = torch.where(event_occurred.bool(), torch.ones_like(survival_months),
                        -torch.ones_like(survival_months))
    return sign * survival_months
