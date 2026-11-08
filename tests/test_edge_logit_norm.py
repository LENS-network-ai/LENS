"""
Tests for EdgeScoringNetwork(logit_norm=True) and the --no-edges dataset option.
    pytest tests/test_edge_logit_norm.py
"""

import torch

from model.EdgeScoring import EdgeScoringNetwork
from model.L0Utils import L0RegularizerParams
from utils.dataset import GraphDataset


def _ring_graph(n):
    adj = torch.zeros(1, n, n)
    for i in range(n):
        for k in (1, 2, 3):
            adj[0, i, (i + k) % n] = 1.0
            adj[0, (i + k) % n, i] = 1.0
    return adj


def _scorer(logit_norm, density=0.30):
    torch.manual_seed(0)
    return EdgeScoringNetwork(feature_dim=16, edge_dim=8, l0_params=L0RegularizerParams(),
                              logit_norm=logit_norm, target_density=density).eval()


def test_logit_norm_keeps_target_fraction_per_graph_regardless_of_size():
    for n in (40, 200):
        scorer = _scorer(True)
        feats = torch.nn.functional.normalize(torch.randn(1, n, 16), dim=2)
        adj = _ring_graph(n)
        with torch.no_grad():
            weights, log_alpha = scorer.compute_edge_weights(
                feats, adj, use_l0=True, training=False, temperature=0.67)
        real = adj[0] > 0
        retained = (weights[0][real] > 0).float().mean().item()
        # Both directions share one logit, and the (1 - rho) quantile sits at the
        # keep-probability midpoint, a hair inside the deterministic gate's zero
        # threshold, so just over rho of the edges is nonzero.
        assert 0.28 <= retained <= 0.36, (n, retained)


def test_logit_norm_off_has_no_extra_parameters():
    assert not hasattr(_scorer(False), 'logit_scale')
    assert hasattr(_scorer(True), 'logit_scale')


def test_logit_norm_logits_are_not_constant():
    scorer = _scorer(True)
    feats = torch.nn.functional.normalize(torch.randn(1, 60, 16), dim=2)
    adj = _ring_graph(60)
    with torch.no_grad():
        _, log_alpha = scorer.compute_edge_weights(
            feats, adj, use_l0=True, training=False, temperature=0.67)
    assert log_alpha[0][adj[0] > 0].std().item() > 0.5


def test_no_edges_dataset_returns_empty_adjacency(tmp_path):
    slide = tmp_path / 'C3L-00001-21'
    slide.mkdir()
    torch.save(torch.randn(5, 16), slide / 'features.pt')
    torch.save(_ring_graph(5)[0], slide / 'adj_s.pt')
    ids = ['C3L-00001-21\tluad\n']

    with_edges = GraphDataset(str(tmp_path), ids)[0]['adj_s']
    without = GraphDataset(str(tmp_path), ids, no_edges=True)[0]
    assert with_edges.sum() > 0
    assert without['adj_s'].sum() == 0
    assert without['adj_s'].shape == with_edges.shape
    assert without['image'].shape == (5, 16)
