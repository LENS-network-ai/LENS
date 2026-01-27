import torch
import torch.nn.functional as F


class EdgeWeightedAttentionPooling:
    """Edge-weighted attention pooling using edge importance for node attention scores"""
    
    def __init__(self):
        pass
    
    def edge_weighted_attention_pooling(self, node_feat, edge_weights, adj_matrix, masks=None):
        """
        Compute graph-level representation using edge-weighted attention
        
        Args:
            node_feat: Node features [batch_size, num_nodes, feature_dim]
            edge_weights: Edge weights [batch_size, num_nodes, num_nodes]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            masks: Optional node masks [batch_size, num_nodes]
            
        Returns:
            graph_rep: Graph-level representation [batch_size, feature_dim]
        """
        batch_size = node_feat.shape[0]
        graph_rep = torch.zeros(batch_size, node_feat.shape[2], device=node_feat.device)
        
        for b in range(batch_size):
            # Get valid nodes
            if masks is not None:
                valid_indices = torch.where(masks[b] > 0)[0]
            else:
                valid_indices = torch.arange(node_feat.shape[1], device=node_feat.device)
            
            if len(valid_indices) == 0:
                continue
            
            # Calculate node importance from edge weights
            edge_mask = (adj_matrix[b] > 0).float()
            node_importance = torch.sum(edge_weights[b, :, valid_indices] * edge_mask[:, valid_indices], dim=0)
            
            # Apply softmax attention
            if torch.sum(node_importance) > 0:
                attention_weights = F.softmax(node_importance, dim=0)
                weighted_features = node_feat[b, valid_indices] * attention_weights.unsqueeze(1)
                graph_rep[b] = torch.sum(weighted_features, dim=0)
            else:
                # Fallback to mean pooling
                graph_rep[b] = torch.mean(node_feat[b, valid_indices], dim=0)
        
        return graph_rep
