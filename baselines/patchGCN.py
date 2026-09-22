"""
PatchGCN and DeepGraphConv baseline models adapted for 3-class classification.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GINConv, GENConv, DeepGCNLayer
from torch_geometric.utils import dense_to_sparse


class Attn_Net_Gated(nn.Module):
    """Attention Network with Sigmoid Gating (3 fc layers) - from original model_utils.py"""
    
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)
    
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class PatchGCN(nn.Module):
    """
    PatchGCN adapted for 3-class classification.
    Based on the exact PatchGCN_Surv architecture from the original codebase.
    """
    
    def __init__(self, 
                 input_dim: int = 1024,
                 hidden_dim: int = 128,  # Original uses 128
                 num_classes: int = 3,   # normal, luad, lscc
                 num_layers: int = 4,    # Original default
                 dropout: float = 0.25,  # Original default
                 resample: float = 0.0):
        super(PatchGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.resample = resample
        
        # Feature compression (matching original)
        if self.resample > 0:
            self.fc = nn.Sequential(*[
                nn.Dropout(self.resample),
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.25)
            ])
        else:
            self.fc = nn.Sequential(*[
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.25)
            ])
        
        # Deep GCN layers (matching original architecture)
        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers):  # num_layers - 1
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                          t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)
        
        # Path processing (matching original)
        self.path_phi = nn.Sequential(*[
            nn.Linear(hidden_dim * 4, hidden_dim * 4), 
            nn.ReLU(), 
            nn.Dropout(0.25)
        ])
        
        # Attention mechanism (matching original)
        self.path_attention_head = Attn_Net_Gated(
            L=hidden_dim * 4, 
            D=hidden_dim * 4, 
            dropout=dropout, 
            n_classes=1
        )
        self.path_rho = nn.Sequential(*[
            nn.Linear(hidden_dim * 4, hidden_dim * 4), 
            nn.ReLU()
        ])
        
        # Classification head (adapted from survival to 3-class)
        self.classifier = torch.nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, sample):
        """Forward pass adapted for LENS dataset format."""
        features = sample['image']  # [num_nodes, feature_dim]
        adj_matrix = sample['adj_s']  # [num_nodes, num_nodes]
        
        # Convert dense adjacency to edge_index
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        
        # Feature compression
        x = self.fc(features)
        x_ = x
        
        # First layer
        x = self.layers[0].conv(x_, edge_index, edge_weight)
        x_ = torch.cat([x_, x], axis=1)
        
        # Remaining layers
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_weight)
            x_ = torch.cat([x_, x], axis=1)
        
        # Path processing
        h_path = x_
        h_path = self.path_phi(h_path)
        
        # Attention-based pooling
        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        
        # Classification (adapted from survival)
        logits = self.classifier(h)
        
        return logits


class DeepGraphConv(nn.Module):
    """
    DeepGraphConv adapted for 3-class classification.
    Based on the exact DeepGraphConv_Surv architecture from the original codebase.
    """
    
    def __init__(self, 
                 input_dim: int = 1024,
                 hidden_dim: int = 256,  # Original uses 256
                 num_classes: int = 3,   # normal, luad, lscc
                 dropout: float = 0.25,  # Original default
                 resample: float = 0.0):
        super(DeepGraphConv, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.resample = resample
        
        # Resampling layer (matching original)
        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample)])
        
        # GIN Conv layers (matching original architecture)
        self.conv1 = GINConv(Seq(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.conv2 = GINConv(Seq(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.conv3 = GINConv(Seq(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        ))
        
        # Attention mechanism (matching original)
        self.path_attention_head = Attn_Net_Gated(
            L=hidden_dim, 
            D=hidden_dim, 
            dropout=dropout, 
            n_classes=1
        )
        self.path_rho = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU()
        ])
        
        # Classification head (adapted from survival to 3-class)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, sample):
        """Forward pass adapted for LENS dataset format."""
        features = sample['image']  # [num_nodes, feature_dim]
        adj_matrix = sample['adj_s']  # [num_nodes, num_nodes]
        
        # Convert dense adjacency to edge_index
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        
        x = features
        
        # Apply resampling if specified
        if self.resample > 0:
            x = self.fc(x)
        
        # GIN convolution layers
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        
        h_path = x3
        
        # Attention-based pooling
        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho(h_path).squeeze()
        
        # Classification (adapted from survival)
        logits = self.classifier(h_path)
        
        return logits
