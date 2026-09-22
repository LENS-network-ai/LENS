"""
LENS adapted for discrete-time survival prediction.

Reuses the exact same edge-sparsification core as ImprovedEdgeGNN
(model/LENS2.py) -- EdgeScoringNetwork + L0 regularization + MultiLayerGNN +
attention pooling -- since that shared mechanism is what's under evaluation;
only the task head and loss differ (hazards over discrete time bins + a
survival NLL loss, instead of class logits + cross-entropy).
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from model.multilayerGNN import MultiLayerGNN
from model.attPooling import MultiHeadAttentionPooling
from model.L0_Reg import L0Regularization, compute_density
from model.EdgeScoring import EdgeScoringNetwork
from model.StatsTracker import StatsTracker
from model.L0Utils import get_loss2, l0_test, L0RegularizerParams
from model.survival_loss import nll_survival_loss


class ImprovedEdgeGNNSurvival(nn.Module):
    """LENS with a discrete-time survival head."""

    def __init__(self, feature_dim, hidden_dim, n_bins=4,
                 num_gnn_layers=3, num_attention_heads=4, use_attention_pooling=True,
                 lambda_reg=0.01, lambda_density=0.03, target_density=0.30,
                 l0_method='hard-concrete',
                 edge_dim=32, dropout=0.2,
                 warmup_epochs=15, ramp_epochs=20,
                 graph_size_adaptation=True, min_edges_per_node=2,
                 l0_gamma=-0.1, l0_zeta=1.1, l0_beta=0.66, initial_temp=5.0,
                 enable_adaptive_lambda=True, enable_density_loss=True,
                 alpha_min=0.2, alpha_max=2.0, nll_alpha=0.0):
        super().__init__()

        self.n_bins = n_bins
        self.use_l0 = True
        self.l0_method = l0_method
        self.num_gnn_layers = num_gnn_layers
        self.use_attention_pooling = use_attention_pooling
        self.nll_alpha = nll_alpha

        self.l0_params = L0RegularizerParams(gamma=l0_gamma, zeta=l0_zeta, beta_l0=l0_beta)

        self.edge_scorer = EdgeScoringNetwork(
            feature_dim=feature_dim, edge_dim=edge_dim,
            l0_method=l0_method, l0_params=self.l0_params,
        )

        self.gnn = MultiLayerGNN(
            feature_dim=feature_dim, hidden_dim=hidden_dim,
            num_layers=num_gnn_layers, dropout=dropout, use_spectral_norm=True,
        )

        if use_attention_pooling:
            self.pooling = MultiHeadAttentionPooling(
                hidden_dim=hidden_dim, num_heads=num_attention_heads,
                dropout=dropout, use_edge_masking=True,
            )
        else:
            from model.GraphPooling import EdgeWeightedAttentionPooling
            self.pooling = EdgeWeightedAttentionPooling()

        self.regularizer = L0Regularization(
            lambda_reg=lambda_reg, lambda_density=lambda_density, target_density=target_density,
            warmup_epochs=warmup_epochs, ramp_epochs=ramp_epochs,
            l0_params=self.l0_params, l0_method=l0_method,
            alpha_min=alpha_min, alpha_max=alpha_max,
            enable_adaptive_lambda=enable_adaptive_lambda, enable_density_loss=enable_density_loss,
            # l0_loss (the L0 penalty) is one-directional -- it only ever
            # pushes density down -- while density_loss is the only genuinely
            # bidirectional restoring force (its gradient pushes density back
            # UP once it's below target). Gating density_loss behind the same
            # warmup_epochs as the L0 ramp (the original shared default)
            # means density can already collapse well past target before any
            # correction is allowed to engage. Activate it from epoch 0 here.
            density_loss_warmup_epochs=0,
            # Restores per-edge gradient parity with l0_loss (see
            # L0Regularization's docstring) -- confirmed necessary empirically:
            # without this, density held a stable plateau near target_density
            # only until l0_loss's cumulative pressure crossed density_loss's
            # (E-diluted) ceiling, then collapse resumed.
            scale_density_loss_by_edges=True,
            # Matches the paper's documented anneal target (was silently
            # hardcoded to 1.0 in the shared L0_Reg.py). Sharpens the
            # deterministic gate near its plateau: an edge needs
            # logAlpha < ~-1.6 to hit exact zero here, vs ~-2.4 at 1.0 --
            # directly relevant to gates getting stuck in the ambiguous
            # middle instead of reaching a confident 0 or 1.
            temperature_min=0.67,
        )

        self.stats_tracker = StatsTracker()

        # Survival head: one hazard logit per discrete time bin
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim // 2, n_bins)),
        )

        self.graph_size_adaptation = graph_size_adaptation
        self.min_edges_per_node = min_edges_per_node

        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.initial_temp = initial_temp
        self.temperature = self.initial_temp

        print(f"\n[LENS-Survival] Model initialized: {num_gnn_layers}-layer GNN, "
              f"n_bins={n_bins}, L0 method={l0_method}")

    def set_print_stats(self, value):
        self.stats_tracker.print_stats = value

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.regularizer.current_epoch = epoch
        schedules = self.regularizer.update_all_schedules(
            current_epoch=epoch, initial_temp=self.initial_temp
        )
        self.temperature = schedules['temperature']

    def forward(self, node_feat, y_disc, censorship, adjs, masks=None):
        """
        Args:
            node_feat : [B, N, feature_dim]
            y_disc    : [B]  ground-truth discrete time-bin index
            censorship: [B]  1 if censored, 0 if event observed
            adjs      : [B, N, N]
            masks     : [B, N]

        Returns:
            hazards, risk_score, loss, weighted_adj
        """
        node_feat = torch.nn.functional.normalize(node_feat, p=2, dim=2)

        edge_weights, logAlpha = self.edge_scorer.compute_edge_weights(
            node_feat=node_feat, adj_matrix=adjs,
            current_epoch=self.current_epoch, warmup_epochs=self.warmup_epochs,
            temperature=self.temperature,
            graph_size_adaptation=self.graph_size_adaptation, min_edges_per_node=self.min_edges_per_node,
            regularizer=self.regularizer, use_l0=True,
            print_stats=self.stats_tracker.print_stats,
            l0_params=self.l0_params, training=self.training,
        )

        h = self.gnn(node_feat, edge_weights, adjs)

        if self.use_attention_pooling:
            graph_rep = self.pooling(h, edge_weights, adjs, masks)
        else:
            graph_rep = self.pooling.edge_weighted_attention_pooling(h, edge_weights, adjs, masks)

        hazard_logits = self.classifier(graph_rep)          # [B, n_bins]
        hazards = torch.sigmoid(hazard_logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk_score = -torch.sum(survival, dim=1)             # higher survival -> lower risk

        survival_loss = nll_survival_loss(hazards, y_disc, censorship, alpha=self.nll_alpha)

        # Sparsity control (density/adaptive-lambda) is driven by the
        # DETERMINISTIC gate (l0_test -- the same function eval-mode already
        # uses), not the stochastic training-time gate (edge_weights above,
        # from l0_train). Confirmed empirically these disagree badly here:
        # stochastic sampling let density hold at 30% during training while
        # the deterministic gate at test time showed 0% of edges above 0.5
        # (every logAlpha <= 0, but never pushed past the ~-2.4 threshold
        # needed to hit exact zero) -- the model learned to mildly suppress
        # every edge under noise without ever learning to discriminate which
        # edges matter. Computing density/density_loss from the deterministic
        # gate means the training signal targets what deployment actually
        # sees, pressuring logAlpha toward the extremes needed for real
        # separation instead of a uniform mild negative. The GNN/pooling/
        # classifier forward pass above still uses the stochastic
        # `edge_weights` unchanged -- only this control-loop input changes.
        deterministic_edge_weights = l0_test(logAlpha, params=self.l0_params, temperature=self.temperature)
        current_density = compute_density(deterministic_edge_weights, adjs)

        # L0 penalty summed over real edges only (masked), NOT averaged.
        # logAlpha is the *dense* [B, N, N] tensor scattered from the sparse
        # per-edge logits, zero-filled at every non-edge position.
        #   - Summing over the full dense tensor (the original code) included
        #     get_loss2(0) for every one of the O(N^2 - E) non-edge entries,
        #     which for a sparse ~8-per-node grid graph (E ~ 8N) dwarfs the
        #     real O(E) edge term -- that's what made the penalty scale with
        #     N^2 instead of edge count and caused every fold to collapse.
        #   - Averaging instead of summing (an earlier version of this fix)
        #     over-corrected: d(mean)/d(logAlpha_i) = d(sum)/d(logAlpha_i)/E,
        #     so dividing by edge count doesn't just shrink the reported loss
        #     value, it divides the GRADIENT on every individual edge's logit
        #     by E (~2000-3000 here) -- diluting the actual per-edge push far
        #     more than any reasonable increase to lambda_reg could offset.
        # Masking without dividing keeps each edge's gradient contribution at
        # its natural sigmoid-derivative scale, independent of graph size,
        # while still excluding the spurious non-edge terms.
        edge_mask = (adjs > 0).float()
        l0_penalty = (get_loss2(logAlpha, params=self.l0_params) * edge_mask).sum()

        # Ambiguity: mean_{real edges}[gate * (1 - gate)] on the deterministic
        # gate. Pure monitoring stat -- NOT part of total_loss (a confidence
        # loss using this exact quantity was tried and removed; it didn't
        # move the plateau even at a properly-calibrated scale, see
        # LENS_survival.py git history). Kept here only so sweeps/reports can
        # see whether a given hyperparameter setting produces genuine
        # bimodal separation (ambiguity -> 0) versus a uniform "adequate"
        # cluster (ambiguity -> 0.21-0.25, its value for any unimodal
        # distribution sitting near target_density).
        with torch.no_grad():
            num_real = edge_mask.sum().clamp(min=1.0)
            ambiguity = (deterministic_edge_weights * (1.0 - deterministic_edge_weights) * edge_mask).sum() / num_real

        reg_loss, reg_stats = self.regularizer.compute_regularization_with_l0(
            l0_penalty=l0_penalty, edge_weights=deterministic_edge_weights, adj_matrix=adjs, return_stats=True
        )

        total_loss = survival_loss + reg_loss

        lambda_eff = reg_stats.get('lambda_eff', self.regularizer.current_lambda)
        self.stats_tracker.update_stats(
            edge_weights, adjs, survival_loss, reg_loss, self.current_epoch, lambda_eff
        )

        if self.stats_tracker.print_stats:
            with torch.no_grad():
                real_mask = edge_mask.bool()
                real_logits = logAlpha[real_mask]
                stoch = edge_weights[real_mask]
                det = deterministic_edge_weights[real_mask]
                n_real = real_logits.numel()

                def _pct(x, p):
                    return torch.quantile(x, p).item() if x.numel() > 0 else float('nan')

                print(f"    [DEBUG] epoch={self.current_epoch} temp={self.temperature:.3f} "
                      f"lambda_eff={lambda_eff:.6f} density={current_density.item()*100:.1f}% "
                      f"n_real_edges={n_real}")
                print(f"    [DEBUG] loss: survival={survival_loss.item():.4f} "
                      f"reg={reg_loss.item():.4f} l0_penalty={l0_penalty.item():.2f}")
                print(f"    [DEBUG] logAlpha:      "
                      f"min={real_logits.min().item():.3f} p10={_pct(real_logits, 0.1):.3f} "
                      f"p50={_pct(real_logits, 0.5):.3f} p90={_pct(real_logits, 0.9):.3f} "
                      f"max={real_logits.max().item():.3f} std={real_logits.std().item():.3f}")
                print(f"    [DEBUG] stochastic gate  (used for GNN forward pass, l0_train): "
                      f"min={stoch.min().item():.3f} mean={stoch.mean().item():.3f} max={stoch.max().item():.3f} "
                      f"exact0={(stoch == 0).float().mean().item()*100:.1f}% "
                      f"exact1={(stoch == 1).float().mean().item()*100:.1f}%")
                print(f"    [DEBUG] deterministic gate (drives density control, l0_test): "
                      f"min={det.min().item():.3f} mean={det.mean().item():.3f} max={det.max().item():.3f} "
                      f"exact0={(det == 0).float().mean().item()*100:.1f}% "
                      f">0.5={(det > 0.5).float().mean().item()*100:.1f}% "
                      f"ambiguous(0,0.5]={((det > 0) & (det <= 0.5)).float().mean().item()*100:.1f}%")

        # Per-batch lambda/density history, read by training/train_survival.py
        # to report the L0 schedule and edge-retention rate over training.
        if not hasattr(self.stats_tracker, 'current_density_history'):
            self.stats_tracker.current_density_history = []
            self.stats_tracker.lambda_eff_history = []
            self.stats_tracker.ambiguity_history = []
        self.stats_tracker.current_density_history.append(
            current_density.item() if isinstance(current_density, torch.Tensor) else current_density
        )
        self.stats_tracker.lambda_eff_history.append(lambda_eff)
        self.stats_tracker.ambiguity_history.append(ambiguity.item())

        return hazards, risk_score, total_loss, adjs * edge_weights

    def __repr__(self):
        return (f"ImprovedEdgeGNNSurvival:\n"
                f"  Architecture: {self.num_gnn_layers}-layer GNN + "
                f"{'Multi-Head Attention' if self.use_attention_pooling else 'Standard'} Pooling\n"
                f"  n_bins={self.n_bins}, L0 Method={self.l0_method}")
