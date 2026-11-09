# model/EdgeScoring.py
"""
Edge Scoring Network with Hard-Concrete, ARM, and STE
Works with dense adjacency matrices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model.L0Utils import l0_train, l0_test, L0RegularizerParams
from model.L0Utils_ARM import arm_sample_gates, ARML0RegularizerParams


class EdgeScoringNetwork(nn.Module):
    """
    Edge Scoring Network - computes importance scores for edges
    Supports Hard-Concrete, ARM, and STE L0 regularization
    """
    
    def __init__(self, feature_dim, edge_dim=32, dropout=0.2, 
                 l0_params=None, l0_method='hard-concrete',
                 logit_norm=False, target_density=0.30):
        """
        Args:
            feature_dim: Node feature dimension
            edge_dim: Hidden dimension for edge scoring MLP
            dropout: Dropout rate
            l0_params: L0RegularizerParams or ARML0RegularizerParams
            l0_method: 'hard-concrete', 'arm', or 'ste'
            logit_norm: standardize each graph's edge logits (z-score, learned
                scale) and place the target_density quantile at the gate's
                keep-probability midpoint, so about target_density of the
                edges of every graph sit above it by construction. Off by
                default (the paper's scorer); Hard-Concrete only.
            target_density: fraction of edges per graph above the midpoint
                when logit_norm is on
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.l0_params = l0_params
        self.l0_method = l0_method
        self.logit_norm = logit_norm
        self.target_density = target_density
        if logit_norm:
            # softplus(logit_scale) is the spread between kept and removed edges
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
        
        # Feature attention tower: 2d -> edge_dim -> d with a softmax over the d
        # channels. A bottleneck of edge_dim (128 in the paper) rather than d
        # keeps this tower at ~0.59M parameters for d=1536 (it was d-wide,
        # ~7.1M, and dominated scorer compute: it runs once per candidate edge).
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim * 2, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, feature_dim),
            nn.Softmax(dim=1)
        )
        
        # Edge scoring MLP: [src_feat || tgt_feat || attended_diff]
        self.edge_mlp = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim *3, edge_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(edge_dim, edge_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Deliberately NOT spectral-normalized. Spectral norm on every layer
            # makes the whole scorer 1-Lipschitz in its input, so logits could
            # differ between any two edges by at most ~||e_1 - e_2|| (about 1
            # for unit-norm endpoint features) -- far less than the ~3.2 logit
            # range a Hard-Concrete gate at tau=0.67 needs to go from exactly 0
            # to exactly 1. Gates could then only move together (via the bias),
            # never separate. The first two layers keep spectral norm; this
            # output layer sets the logit scale.
            nn.Linear(edge_dim, 1)
        )
        
        # Kaiming initialization
        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Positive bias for final layer → gates start open
        final_module = self.edge_mlp[-1]
        final_layer = final_module.module if hasattr(final_module, 'module') else final_module
        nn.init.constant_(final_layer.bias, 1.0)
        nn.init.xavier_uniform_(final_layer.weight, gain=0.1)
        
        # Store last logAlpha for ARM gradient computation
        self.last_logAlpha = None
        
        print(f"[EdgeScoring] Initialized: method={l0_method}, feature_dim={feature_dim}, edge_dim={edge_dim}")
    
    def _symmetrize_logits(self, logAlpha, batch_ids, src_nodes, tgt_nodes,
                           batch_size, num_nodes):
        """Average the logits of the two orientations of every undirected edge,
        so each undirected edge has ONE logit. Applied before gating: the gate,
        the keep probability pi and the penalty/density terms then all describe
        the same quantity, and at inference both orientations get the same
        deterministic gate (averaging gates after gating would keep an edge
        whenever either orientation's gate is nonzero). Edges present in only
        one direction keep their own logit."""
        dense = torch.zeros(batch_size, num_nodes, num_nodes, device=logAlpha.device)
        has = torch.zeros_like(dense)
        dense[batch_ids, src_nodes, tgt_nodes] = logAlpha
        has[batch_ids, src_nodes, tgt_nodes] = 1.0
        both = (has * has.transpose(-1, -2)).bool()
        dense = torch.where(both, 0.5 * (dense + dense.transpose(-1, -2)), dense)
        return dense[batch_ids, src_nodes, tgt_nodes]

    def _standardize_logits(self, logAlpha, batch_ids, batch_size, params):
        """Per-graph z-score of the (already symmetrized) logits, shifted so the
        (1 - target_density) quantile of each graph lands on the keep-probability
        midpoint (pi = 0.5, logit params.const1). The quantile is per graph, so the
        retained fraction does not depend on graph size or on a global bias."""
        out = torch.empty_like(logAlpha)
        scale = F.softplus(self.logit_scale)
        for b in range(batch_size):
            mask = batch_ids == b
            if not mask.any():
                continue
            z = logAlpha[mask]
            z = (z - z.mean()) / (z.std(unbiased=False) + 1e-6)
            q = torch.quantile(z.detach(), 1.0 - self.target_density)
            out[mask] = params.const1 + scale * (z - q)
        return out

    def compute_edge_weights(
        self, node_feat, adj_matrix,
        current_epoch=0, warmup_epochs=5, temperature=None,
        graph_size_adaptation=True, min_edges_per_node=2,
        regularizer=None, use_l0=False, print_stats=False,
        l0_params=None, training=True
    ):
        """
        Compute edge weights using L0 method for sparse graphs
        
        Args:
            node_feat: Node features [B, N, D]
            adj_matrix: Dense adjacency matrix [B, N, N]
            current_epoch: Current training epoch
            warmup_epochs: Number of warmup epochs
            temperature: Hard-Concrete temperature tau
            graph_size_adaptation: Adapt to graph size
            min_edges_per_node: Minimum edges per node
            regularizer: L0Regularization instance
            use_l0: Whether using L0 regularization
            print_stats: Print statistics
            l0_params: L0 parameters
            training: Training mode
        
        Returns:
            edge_weights_dense: Dense edge weights [B, N, N]
            logAlpha_dense: Dense logits [B, N, N]
        """
        batch_size, num_nodes, feat_dim = node_feat.shape
        device = node_feat.device
        params = l0_params if l0_params is not None else self.l0_params
        
        # Ensure dense adjacency
        if adj_matrix.is_sparse:
            adj_matrix = adj_matrix.to_dense()
        
        # Build sparse edge index
        edge_index_list = []
        for b in range(batch_size):
            edges_b = (adj_matrix[b] > 0).nonzero(as_tuple=False)
            if edges_b.numel() == 0:
                continue
            batch_col = torch.full((edges_b.size(0), 1), b, device=device)
            edge_index_list.append(torch.cat([batch_col, edges_b], dim=1))
        
        if len(edge_index_list) == 0:
            return torch.zeros_like(adj_matrix), torch.zeros_like(adj_matrix)
        
        edge_index = torch.cat(edge_index_list, dim=0)
        batch_ids, src_nodes, tgt_nodes = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        
        # Gather node features
        src_feat = node_feat[batch_ids, src_nodes]
        tgt_feat = node_feat[batch_ids, tgt_nodes]
        
        # Normalize features
        src_feat = F.normalize(src_feat, p=2, dim=1)
        tgt_feat = F.normalize(tgt_feat, p=2, dim=1)
        
        # Compute feature differences
        feat_diff = torch.abs(src_feat - tgt_feat)
        
        # Apply feature attention
        edge_feats_raw = torch.cat([src_feat, tgt_feat], dim=1)
        attention_weights = self.feature_attention(edge_feats_raw)
        # attention_weights sums to 1 over feature_dim channels, so a uniform
        # attention is 1/feature_dim per channel and would shrink the contrast
        # block ~feature_dim-fold relative to the unit-norm endpoint blocks.
        # Scaling by feature_dim gives mean weight 1 (uniform attention ->
        # plain |x_i - x_j|), so the contrast block keeps a usable magnitude.
        attended_diff = feat_diff * attention_weights * self.feature_dim
        
        # Final edge features: [src || tgt || attended_diff]
        edge_features = torch.cat([src_feat, tgt_feat, attended_diff], dim=-1)
        
        # Compute edge logits
        logAlpha = self.edge_mlp(edge_features).squeeze(-1)
        logAlpha = self._symmetrize_logits(logAlpha, batch_ids, src_nodes, tgt_nodes,
                                           batch_size, num_nodes)
        if self.logit_norm and params is not None and self.l0_method == 'hard-concrete':
            logAlpha = self._standardize_logits(logAlpha, batch_ids, batch_size, params)
        self.last_logAlpha = logAlpha
        
        # Store logits in regularizer
        if use_l0 and regularizer is not None:
            regularizer.clear_logits()
            for b in range(batch_size):
                mask_b = batch_ids == b
                regularizer.store_logits(b, logAlpha[mask_b])
        
        # Apply L0 gating
        if use_l0 and params is not None:
            if self.l0_method == 'hard-concrete':
                edge_weights = l0_train(logAlpha, params=params, temperature=temperature) if training else l0_test(logAlpha, params=params, temperature=temperature)
            
            elif self.l0_method == 'arm':
                edge_weights, edge_weights_anti = arm_sample_gates(logAlpha, params, training=training)
            
            elif self.l0_method == 'ste':
                from model.L0Utils_STE import ste_sample_gates
                if training:
                    edge_weights, _ = ste_sample_gates(logAlpha, temperature=temperature)
                else:
                    probs = torch.sigmoid(logAlpha / temperature)
                    edge_weights = (probs > 0.5).float()
            else:
                raise ValueError(f"Unknown l0_method: {self.l0_method}")
        else:
            # No L0 gating requested (e.g. reg_mode='none'): keep every
            # existing edge with a full weight of 1 (no sparsification).
            edge_weights = torch.ones_like(logAlpha)

        # Scatter edge weights and logits back to dense [B, N, N]
        edge_weights_dense = torch.zeros((batch_size, num_nodes, num_nodes), device=device)
        logAlpha_dense = torch.zeros((batch_size, num_nodes, num_nodes), device=device)

        edge_weights_dense[batch_ids, src_nodes, tgt_nodes] = edge_weights
        logAlpha_dense[batch_ids, src_nodes, tgt_nodes] = logAlpha

        # Symmetrize for undirected graphs. edge_mlp's Linear layers are NOT
        # invariant to swapping [src_feat, tgt_feat]'s concatenation order --
        # the (i,j) and (j,i) rows in edge_index get different edge_features
        # (attention_weights/attended_diff differ by direction too), so
        # logAlpha_ij != logAlpha_ji in general, computed as two independent
        # values. Left unfixed, this means node i's and node j's own
        # aggregate() calls use genuinely different weights for the same
        # physical edge during training (aggregate() reads row i / row j of
        # edge_weights separately) -- a real asymmetry for graphs that are
        # undirected by construction (spatial adjacency has no directionality
        # to justify an edge "existing" from i's side but not j's, unlike
        # e.g. GAT's deliberately asymmetric attention weights on an
        # otherwise-fixed graph). Previously only the frozen inference-time
        # gate was explicitly symmetrized (z_ij averaged with z_ji), leaving
        # a train/inference discrepancy. Average both directions here, before
        # anything downstream (GNN aggregation, L0 penalty, density control)
        # consumes them, so training and inference operate on the same
        # symmetric quantity throughout. Masked to positions where both
        # directions are real edges, so this is a no-op wherever a graph
        # only has one direction -- shouldn't happen with this repo's own
        # adjacency construction (symmetric by construction in both
        # preprocessing/graph_construction.py and
        # preprocessing/build_survival_graphs.py), but kept safe rather than
        # assumed.
        edge_mask = (adj_matrix > 0).float()
        bidirectional_mask = (edge_mask * edge_mask.transpose(-1, -2)).bool()

        edge_weights_sym = 0.5 * (edge_weights_dense + edge_weights_dense.transpose(-1, -2))
        edge_weights_dense = torch.where(bidirectional_mask, edge_weights_sym, edge_weights_dense)

        logAlpha_sym = 0.5 * (logAlpha_dense + logAlpha_dense.transpose(-1, -2))
        logAlpha_dense = torch.where(bidirectional_mask, logAlpha_sym, logAlpha_dense)

        if use_l0 and params is not None and print_stats:
            # Moved here (post-symmetrization) so this reports what's
            # actually used downstream, not the pre-symmetrization values.
            real_mask = edge_mask.bool()
            n_real = real_mask.sum().item()
            active_edges = (edge_weights_dense[real_mask] > 0.5).float().sum().item()
            print(f"   [EdgeScoring-{self.l0_method.upper()}] Active edges: {active_edges}/{n_real} "
                  f"({100*active_edges/max(n_real,1):.1f}%)")

        return edge_weights_dense, logAlpha_dense
