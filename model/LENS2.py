"""
LENS+ Model with Multi-Layer GNN and Attention Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model.multilayerGNN import MultiLayerGNN
from model.attPooling import MultiHeadAttentionPooling
from model.L0_Reg import L0Regularization
from model.EdgeScoring import EdgeScoringNetwork
from model.StatsTracker import StatsTracker
from model.L0Utils import l0_train, l0_test, L0RegularizerParams, get_loss2
from model.L0Utils_ARM import ARML0RegularizerParams, arm_sample_gates


class ImprovedEdgeGNN(nn.Module):
    """LENS with Multi-Layer GNN and Multi-Head Attention Pooling"""
    
    def __init__(self, feature_dim, hidden_dim, num_classes, 
                 # GNN parameters
                 num_gnn_layers=3,
                 num_attention_heads=4,
                 use_attention_pooling=True,
                 
                 # Regularization parameters
                 lambda_reg=0.01, 
                 lambda_density=0.03,
                 target_density=0.30,
                 reg_mode='l0', 
                 l0_method='hard-concrete',
                 
                 # Architecture parameters
                 edge_dim=32, 
                 dropout=0.2,
                 
                 # Training parameters
                 warmup_epochs=15,
                 ramp_epochs=20,
                 graph_size_adaptation=True, 
                 min_edges_per_node=2, 
                 
                 # L0 parameters
                 l0_gamma=-0.1, 
                 l0_zeta=1.1, 
                 l0_beta=0.66, 
                 baseline_ema=0.9, 
                 initial_temp=5.0,
                 
                 # Penalty mode parameters
                 enable_adaptive_lambda=True,
                 enable_density_loss=True,
                 alpha_min=0.2,
                 alpha_max=2.0,
                 edge_logit_norm=False):
        
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_mode = reg_mode
        self.l0_method = l0_method
        self.use_l0 = (reg_mode == 'l0')
        self.num_gnn_layers = num_gnn_layers
        self.use_attention_pooling = use_attention_pooling
        
        # Create L0 parameters
        self.l0_params = None
        if self.use_l0:
            if l0_method == 'hard-concrete':
                self.l0_params = L0RegularizerParams(
                    gamma=l0_gamma, zeta=l0_zeta, beta_l0=l0_beta
                )
            elif l0_method == 'arm':
                self.l0_params = ARML0RegularizerParams(
                    gamma=l0_gamma, zeta=l0_zeta, baseline_ema=baseline_ema
                )
        
        # Edge scoring network
        self.edge_scorer = EdgeScoringNetwork(
            feature_dim=feature_dim,
            edge_dim=edge_dim,
            l0_method=l0_method,
            l0_params=self.l0_params,
            logit_norm=edge_logit_norm,
            target_density=target_density,
        )
        
        # Multi-Layer GNN
        self.gnn = MultiLayerGNN(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            use_spectral_norm=True
        )
        
        # Attention Pooling
        if use_attention_pooling:
            self.pooling = MultiHeadAttentionPooling(
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                use_edge_masking=True
            )
        else:
            from model.GraphPooling import EdgeWeightedAttentionPooling
            self.pooling = EdgeWeightedAttentionPooling()
        
        # Regularizer. density_loss_warmup_epochs/scale_density_loss_by_edges/
        # temperature_min match model/LENS_survival.py's settings -- ported
        # here after confirming classification exhibits the exact same
        # collapse (logAlpha driven to <-7 by epoch 3, deterministic gate
        # 100% ambiguous) that these fixes were originally written for. See
        # L0Regularization's docstring for what each one does and why.
        self.regularizer = L0Regularization(
            lambda_reg=lambda_reg,
            lambda_density=lambda_density,
            target_density=target_density,
            warmup_epochs=warmup_epochs,
            ramp_epochs=ramp_epochs,
            l0_params=self.l0_params,
            l0_method=l0_method,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            enable_adaptive_lambda=enable_adaptive_lambda,
            enable_density_loss=enable_density_loss,
            density_loss_warmup_epochs=0,
            scale_density_loss_by_edges=True,
            temperature_min=0.67,
        )
        
        # Stats tracker
        self.stats_tracker = StatsTracker()
        
        # Classifier
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            spectral_norm(nn.Linear(hidden_dim // 2, hidden_dim // 4)),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            spectral_norm(nn.Linear(hidden_dim // 4, num_classes))
        )
        
        # Initialize classifier
        for m in self.classifier:
            if isinstance(m, nn.Linear) or hasattr(m, 'module'):
                weight = m.weight if not hasattr(m, 'module') else m.module.weight
                nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
                bias = m.bias if not hasattr(m, 'module') else m.module.bias
                if bias is not None:
                    nn.init.constant_(bias, 0)
        
        # Graph adaptation
        self.graph_size_adaptation = graph_size_adaptation
        self.min_edges_per_node = min_edges_per_node
        
        # Training state
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.initial_temp = initial_temp
        self.temperature = self.initial_temp
        
        # ARM-specific
        self.last_edge_weights_anti = None
        
        print(f"\n[LENS] Model initialized:")
        print(f"  Architecture: Multi-Layer GNN + {'Multi-Head Attention' if use_attention_pooling else 'Standard'} Pooling")
        print(f"  GNN Layers: {num_gnn_layers}")
        if use_attention_pooling:
            print(f"  Attention Heads: {num_attention_heads}")
        print(f"  Hidden Dim: {hidden_dim}")
        print(f"  Regularization: {reg_mode}")
        print(f"  L0 Method: {l0_method if self.use_l0 else 'N/A'}")
    
    def set_print_stats(self, value):
        """Control stats printing"""
        self.stats_tracker.print_stats = value
    
    def update_temperature_and_lambda(self):
        """Update temperature and lambda schedules"""
        if hasattr(self.regularizer, 'update_all_schedules'):
            schedules = self.regularizer.update_all_schedules(
                current_epoch=self.current_epoch,
                initial_temp=self.initial_temp
            )
            self.temperature = schedules['temperature']
    
    def set_epoch(self, epoch):
        """Set current epoch and update schedules"""
        self.current_epoch = epoch
        self.regularizer.current_epoch = epoch
        self.update_temperature_and_lambda()
    
    def forward(self, node_feat, labels, adjs, masks=None, 
                return_edge_weights_anti=False):
        """
        Forward pass with multi-layer GNN and attention pooling
        
        Args:
            node_feat: Node features [B, N, D]
            labels: Ground truth labels [B]
            adjs: Adjacency matrices [B, N, N]
            masks: Node masks [B, N]
            return_edge_weights_anti: Return antithetic samples (ARM only)
        
        Returns:
            logits, labels, total_loss, weighted_adj[, edge_weights_anti]
        """
        # Normalize node features
        node_feat = F.normalize(node_feat, p=2, dim=2)
        batch_size = node_feat.shape[0]
        
        # Compute edge weights
        edge_scorer_outputs = self.edge_scorer.compute_edge_weights(
            node_feat=node_feat,
            adj_matrix=adjs,
            current_epoch=self.current_epoch,
            warmup_epochs=self.warmup_epochs,
            temperature=self.temperature,
            graph_size_adaptation=self.graph_size_adaptation,
            min_edges_per_node=self.min_edges_per_node,
            regularizer=self.regularizer if self.use_l0 else None,
            use_l0=self.use_l0,
            print_stats=self.stats_tracker.print_stats,
            l0_params=self.l0_params,
            training=self.training
        )
        
        # Handle different return formats
        if self.use_l0 and self.l0_method == 'arm' and self.training:
            edge_weights, edge_weights_anti, logAlpha = edge_scorer_outputs
            self.last_edge_weights_anti = edge_weights_anti
        else:
            if isinstance(edge_scorer_outputs, tuple) and len(edge_scorer_outputs) == 2:
                edge_weights, logAlpha = edge_scorer_outputs
            else:
                edge_weights = edge_scorer_outputs
                logAlpha = None
            edge_weights_anti = None
        
        # Multi-layer GNN
        h = self.gnn(node_feat, edge_weights, adjs)
        
        # Attention pooling
        if self.use_attention_pooling:
            graph_rep = self.pooling(h, edge_weights, adjs, masks)
        else:
            graph_rep = self.pooling.edge_weighted_attention_pooling(h, edge_weights, adjs, masks)
        
        # Classification
        logits = self.classifier(graph_rep)
        cls_loss = F.cross_entropy(logits, labels)
        
        # Regularization
        if self.use_l0 and logAlpha is not None:
            from model.L0_Reg import compute_density

            if self.l0_method == 'hard-concrete':
                # Both fixes ported from model/LENS_survival.py after
                # confirming classification hit the identical collapse:
                #
                # 1. l0_penalty masked to real edges only (edge_mask),
                #    NOT summed over the full dense [B,N,N] logAlpha --
                #    logAlpha is zero-filled at every non-edge position, and
                #    an unmasked .sum() includes get_loss2(0) for every one
                #    of the O(N^2 - E) non-edge entries. For these WSI
                #    graphs (hundreds+ nodes, ~8 edges/node) that inflated
                #    the penalty ~89x over the real edge count (confirmed:
                #    l0_penalty=451344 against n_real_edges=5086), swamping
                #    cls_loss by 100-1000x and driving logAlpha to <-7 by
                #    epoch 3.
                # 2. Density control driven by the deterministic gate
                #    (l0_test), not the stochastic training-time gate
                #    (edge_weights, from l0_train) -- they can disagree
                #    badly, same as survival found. The GNN/pooling/
                #    classifier forward pass above still uses the
                #    stochastic `edge_weights` unchanged.
                edge_mask = (adjs > 0).float()
                from model.L0Utils import get_loss2
                keep_prob = get_loss2(logAlpha, params=self.l0_params) * edge_mask
                l0_penalty = keep_prob.sum()

                # Deterministic gate kept for the [DEBUG] diagnostics only.
                deterministic_edge_weights = l0_test(logAlpha, params=self.l0_params, temperature=self.temperature)
                # Density = expected fraction of KEPT edges, mean_e P(z_e > 0)
                # (the mean keep probability pi), not the mean gate value; see
                # model/LENS_survival.py for why a mean-gate target can be met
                # with no edges removed.
                current_density = compute_density(keep_prob, adjs)
                density_control_weights = keep_prob
            elif self.l0_method == 'arm':
                # Not covered by the survival-side fix (ARM has no l0_test
                # analog) -- left as before.
                current_density = compute_density(edge_weights, adjs)
                density_control_weights = edge_weights
                from model.L0Utils_ARM import get_expected_l0_arm
                l0_penalty = get_expected_l0_arm(logAlpha, self.l0_params)
            else:
                raise ValueError(f"Unknown l0_method: {self.l0_method}")

            # Compute regularization
            reg_loss, reg_stats = self.regularizer.compute_regularization_with_l0(
                l0_penalty=l0_penalty,
                edge_weights=density_control_weights,
                adj_matrix=adjs,
                return_stats=True
            )

            l0_loss = reg_stats.get('l0_loss', 0)
            density_loss = reg_stats.get('density_loss', 0)
            lambda_eff = reg_stats.get('lambda_eff', self.regularizer.current_lambda)
        else:
            reg_loss = self.regularizer.compute_regularization(edge_weights, adjs, return_stats=False)
            l0_loss = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            density_loss = 0
            lambda_eff = self.regularizer.current_lambda
            current_density = 0
        
        # Total loss
        total_loss = cls_loss + reg_loss
        
        # Update stats
        self.stats_tracker.update_stats(
            edge_weights, adjs, cls_loss, reg_loss,
            self.current_epoch, lambda_eff
        )

        # Same rich per-batch [DEBUG] block as model/LENS_survival.py's
        # forward() -- logAlpha/gate distributions, not just epoch
        # averages. Hard-concrete only (ARM/STE have no l0_test analog).
        # Reuses edge_mask/deterministic_edge_weights computed above for
        # the actual density-control mechanism, rather than recomputing --
        # this IS the gate driving current_density/reg_loss now, not just
        # a diagnostic-only observation.
        if self.stats_tracker.print_stats and self.use_l0 and logAlpha is not None \
                and self.l0_method == 'hard-concrete':
            with torch.no_grad():
                real_mask = edge_mask.bool()
                real_logits = logAlpha[real_mask]
                stoch = edge_weights[real_mask]
                det = deterministic_edge_weights[real_mask]
                n_real = real_logits.numel()

                def _pct(x, p):
                    return torch.quantile(x, p).item() if x.numel() > 0 else float('nan')

                density_pct = current_density.item() * 100 if isinstance(current_density, torch.Tensor) else current_density * 100
                l0_penalty_val = l0_penalty.item() if isinstance(l0_penalty, torch.Tensor) else l0_penalty

                print(f"    [DEBUG] epoch={self.current_epoch} temp={self.temperature:.3f} "
                      f"lambda_eff={lambda_eff:.6f} density={density_pct:.1f}% "
                      f"n_real_edges={n_real}")
                print(f"    [DEBUG] loss: cls={cls_loss.item():.4f} "
                      f"reg={reg_loss.item():.4f} l0_penalty={l0_penalty_val:.2f}")
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

        # Add density control stats
        if not hasattr(self.stats_tracker, 'density_loss_history'):
            self.stats_tracker.density_loss_history = []
            self.stats_tracker.current_density_history = []
            self.stats_tracker.lambda_eff_history = []
        
        if self.use_l0:
            self.stats_tracker.density_loss_history.append(density_loss)
            self.stats_tracker.current_density_history.append(
                current_density.item() if isinstance(current_density, torch.Tensor) else current_density
            )
            self.stats_tracker.lambda_eff_history.append(lambda_eff)
        
        # Return
        if return_edge_weights_anti and edge_weights_anti is not None:
            return logits, labels, total_loss, adjs * edge_weights, edge_weights_anti
        else:
            return logits, labels, total_loss, adjs * edge_weights
    
    def __repr__(self):
        """String representation"""
        return (f"ImprovedEdgeGNN(LENS-v2):\n"
                f"  Architecture: {self.num_gnn_layers}-layer GNN + "
                f"{'Multi-Head Attention' if self.use_attention_pooling else 'Standard'} Pooling\n"
                f"  Regularization: {self.reg_mode}\n"
                f"  L0 Method: {self.l0_method if self.use_l0 else 'N/A'}\n"
                f"  Classes: {self.num_classes}")
