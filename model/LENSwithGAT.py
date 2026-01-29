import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model.L0_Reg import EGLassoRegularization
from model.EdgeScoring import EdgeScoringNetwork
from model.GraphPooling import EdgeWeightedAttentionPooling
from model.StatsTracker import StatsTracker
from model.GATLayer import MultiLayerGAT

class LENSWithGAT(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, beta=0.01, egl_mode='egl', edge_dim=32,
                 warmup_epochs=5, graph_size_adaptation=True, min_edges_per_node=2, dropout=0.2,
                 gat_heads=8, gat_layers=2, gat_hidden_dim=64, use_l0_gat=False, gat_bias_l0=-0.5):
        
        super().__init__()
        self.num_classes = num_classes        
        
        # Create component modules
        self.edge_scorer = EdgeScoringNetwork(feature_dim, edge_dim, dropout)
        self.pooling = EdgeWeightedAttentionPooling()
        self.regularizer = EGLassoRegularization(beta, egl_mode, warmup_epochs)
        self.stats_tracker = StatsTracker()
        
        # Add GAT layers
        self.use_gat = True
        self.gat = MultiLayerGAT(
            in_dim=feature_dim,
            hidden_dim=gat_hidden_dim,
            out_dim=hidden_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout,
            attn_dropout=dropout,
            alpha=0.2,  # LeakyReLU slope
            bias_l0=gat_bias_l0,
            residual=True,
            l0=1 if use_l0_gat else 0  # Whether to use L0 regularization in GAT
        )
        
        # Classifier with spectral normalization
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            spectral_norm(nn.Linear(hidden_dim // 2, num_classes))
        )
        
        # Graph adaptation parameters
        self.graph_size_adaptation = graph_size_adaptation
        self.min_edges_per_node = min_edges_per_node
        
        # Store epoch information
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.initial_temp = 5.0
        self.temperature = self.initial_temp
        
        print(f"Initialized LENSWithGAT with {egl_mode} regularization (base_beta={beta})")
        print(f"Using GAT with {gat_layers} layers, {gat_heads} heads, L0 regularization: {use_l0_gat}")
        print(f"Using warmup for {warmup_epochs} epochs")
        print(f"Using graph size adaptation: {graph_size_adaptation}")
        print(f"Using min edges per node: {min_edges_per_node}")
        print(f"Using Gumbel-Sigmoid with initial temperature {self.temperature}")
    
    def set_print_stats(self, value):
        """Control whether to print stats during forward pass"""
        self.stats_tracker.print_stats = value
    
    def update_temperature_and_beta(self):
        """Update temperature and beta based on current epoch"""
        self.temperature = self.regularizer.update_temperature(
            self.current_epoch, self.warmup_epochs, self.initial_temp
        )
        self.regularizer.update_beta(self.current_epoch, self.warmup_epochs)
        
        if self.stats_tracker.print_stats:
            print(f"Epoch {self.current_epoch}: temperature={self.temperature:.3f}, beta={self.regularizer.current_beta:.6f}")
    
    def set_epoch(self, epoch):
        """Set the current epoch number and update temperature/beta"""
        self.current_epoch = epoch
        self.regularizer.current_epoch = epoch
        self.update_temperature_and_beta()
    
    def forward(self, node_feat, labels, adjs, masks=None):
        """Forward pass with per-graph analysis and edge-weighted attention pooling"""
        node_feat = F.normalize(node_feat, p=2, dim=2) 
        batch_size = node_feat.shape[0]
        
        # Compute edge weights
        edge_weights = self.edge_scorer.compute_edge_weights(
            node_feat, adjs, self.current_epoch, self.warmup_epochs,
            self.temperature, self.graph_size_adaptation, self.min_edges_per_node,
            self.stats_tracker.print_stats
        )
        
        # Apply GAT layers with the computed edge weights
        # We use the original adjacency matrix for structural connectivity
        # and the learned edge weights for attention modulation
        gat_output, attention_weights, l0_loss = self.gat(node_feat, adjs)
        
        # Apply edge-weighted attention pooling to get graph-level representation
        graph_rep = self.pooling.edge_weighted_attention_pooling(gat_output, edge_weights, adjs, masks)
        
        # Classification
        logits = self.classifier(graph_rep)
        
        # Compute classification loss
        cls_loss = F.cross_entropy(logits, labels)
        
        # Compute regularization loss (edge pruning + GAT L0 regularization if enabled)
        edge_reg_loss = self.regularizer.compute_regularization(edge_weights, adjs)
        total_reg_loss = edge_reg_loss + l0_loss * 0.01  # Scale L0 loss 
        
        # Total loss
        total_loss = cls_loss + total_reg_loss
        
        # Track statistics
        self.stats_tracker.update_stats(edge_weights, adjs, cls_loss, total_reg_loss, 
                                        self.current_epoch, self.regularizer.current_beta)
        
        # Return same format as expected
        return logits, labels, total_loss, adjs * edge_weights
    
    # Delegate visualization methods to the stats tracker
    def save_graph_analysis(self, epoch, batch_idx, save_dir='./'):
        return self.stats_tracker.save_graph_analysis(
            epoch, batch_idx, save_dir, self.regularizer.current_beta, 
            self.temperature, self.warmup_epochs
        )
    
    def plot_edge_weight_distribution(self, weighted_adj, epoch, batch_idx=0, save_dir='./'):
        return self.stats_tracker.plot_edge_weight_distribution(
            weighted_adj, epoch, batch_idx, save_dir, self.regularizer.current_beta, 
            self.temperature, self.current_epoch, self.warmup_epochs
        )
    
    def plot_stats(self, save_path='stats.png'):
        return self.stats_tracker.plot_stats(
            save_path, self.regularizer.egl_mode, self.regularizer.base_beta, 
            self.warmup_epochs
        )
    
    def save_sparsification_report(self, epoch, save_dir='./'):
        return self.stats_tracker.save_sparsification_report(
            epoch, save_dir, self.regularizer.current_beta, self.temperature,
            self.warmup_epochs
        )
