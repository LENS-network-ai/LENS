"""
baselines/abmil.py

ABMIL: Attention-Based Multiple Instance Learning baseline for LENS.
Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.

Designed to integrate cleanly into the LENS repository:
  - Input : features.pt  ->  Tensor [N x 512]  (ResNet18-SimCLR, same as LENS)
  - Output: logits, probabilities, predicted class, attention weights [N]
  - Attention weights are directly usable by visualize_heatmap.py

Architecture mirrors LENS classifier head dimensions for fair comparison:
  projector      : 512 -> 512  (linear + ReLU + dropout)
  gated attention: N patches -> 1 slide vector
  classifier MLP : 512 -> 256 -> num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPool(nn.Module):
    """
    Gated attention pooling from Ilse et al. (2018).

    Two parallel branches (tanh + sigmoid) are multiplied element-wise
    to gate the attention signal, then projected to a scalar weight per patch.
    Softmax over all N patches yields a normalized importance distribution.

    Args:
        input_dim  (int)  : patch embedding dimension (default: 512)
        hidden_dim (int)  : attention hidden dimension (default: 256)
        dropout    (float): dropout rate inside each branch (default: 0.25)
    """
    def __init__(
        self,
        input_dim:  int   = 512,
        hidden_dim: int   = 256,
        dropout:    float = 0.25
    ):
        super(GatedAttentionPool, self).__init__()

        # tanh branch  (matrix V in Ilse et al.)
        self.V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        # sigmoid branch  (matrix U in Ilse et al.)
        self.U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        # weight vector  (w in Ilse et al.)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H: torch.Tensor):
        """
        Args:
            H : patch embeddings  [N x input_dim]
        Returns:
            A : attention weights [1 x N]    softmax over N patches
            Z : slide embedding   [1 x input_dim]
        """
        A = self.w(self.V(H) * self.U(H))   # N x 1
        A = F.softmax(A.T, dim=1)           # 1 x N   (softmax over patches)
        Z = torch.mm(A, H)                  # 1 x input_dim
        return A, Z


class ABMIL(nn.Module):
    """
    Attention-Based MIL for multi-class WSI classification.

    Plugs into the LENS training infrastructure identically to ImprovedEdgeGNN:
      - Accepts raw patch features [N x feature_dim] loaded from features.pt
      - Returns (logits, Y_prob, Y_hat, A) matching LENS forward() signature
      - get_attention_weights() returns [N] for visualize_heatmap.py

    Args:
        feature_dim  (int)  : input patch feature dim  (default: 512, SimCLR)
        hidden_dim   (int)  : projection + classifier hidden dim (default: 256)
        num_classes  (int)  : number of output classes  (default: 3)
        dropout      (float): dropout rate (default: 0.25)
    """
    def __init__(
        self,
        feature_dim: int   = 512,
        hidden_dim:  int   = 256,
        num_classes: int   = 3,
        dropout:     float = 0.25,
        # Accept and silently ignore LENS-specific kwargs so the same
        # instantiation call works for both LENS and ABMIL
        **kwargs
    ):
        super(ABMIL, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Feature projector — keeps same dim as input (512 -> 512)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention pooling
        self.attention = GatedAttentionPool(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Classifier head — mirrors LENS classifier dimensions
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, adj=None):
        """
        Args:
            x   : patch features [N x feature_dim]  from features.pt
            adj : ignored (accepted for API compatibility with LENS loaders
                  that pass both features and adjacency)
        Returns:
            logits  : [1 x num_classes]
            Y_prob  : [1 x num_classes]  softmax probabilities
            Y_hat   : scalar LongTensor  predicted class index
            A       : [1 x N]            attention weights (for heatmap)
        """
        H      = self.projector(x)              # N x feature_dim
        A, Z   = self.attention(H)              # A: 1xN,  Z: 1 x feature_dim
        logits = self.classifier(Z)             # 1 x num_classes
        Y_prob = F.softmax(logits, dim=1)       # 1 x num_classes
        Y_hat  = torch.argmax(Y_prob, dim=1)    # scalar
        return logits, Y_prob, Y_hat, A

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns patch-level attention weights of shape [N].
        Compatible with LENS visualize_heatmap.py pipeline:
            weights = model.get_attention_weights(features)
            # weights[i] -> importance of patch i for heatmap rendering
        """
        _, _, _, A = self.forward(x)
        return A.squeeze(0)                     # N


# ---------------------------------------------------------------------------
# Sanity check — run: python baselines/abmil.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    N = 3000                                    # representative WSI patch count
    x = torch.randn(N, 512)                     # SimCLR 512-dim features

    model = ABMIL(feature_dim=512, hidden_dim=256, num_classes=3)
    model.eval()

    with torch.no_grad():
        logits, probs, pred, attn = model(x)

    print(f"Input          : {x.shape}")
    print(f"Logits         : {logits.shape}")
    print(f"Probabilities  : {probs.squeeze().tolist()}")
    print(f"Prediction     : class {pred.item()}")
    print(f"Attention      : {attn.shape}   sum={attn.sum():.6f}")
    print(f"Attention min/max: {attn.min():.6f} / {attn.max():.6f}")
    print("\nSanity check passed.")
