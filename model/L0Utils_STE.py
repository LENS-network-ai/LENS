# model/L0Utils_STE.py
"""
Straight-Through Estimator (STE) for L0 Regularization
Hard binary gates during training with straight-through gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STEBinarize(torch.autograd.Function):
    """
    Straight-Through Estimator for binary gates
    
    Forward: Hard threshold (binary output)
    Backward: Straight-through gradient (ignores threshold)
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        Forward: Binary gate based on threshold
        
        Args:
            input: Probabilities [B, N, N] or logits
        
        Returns:
            output: Binary gates {0, 1}
        """
        # Hard threshold at 0.5
        output = (input > 0.5).float()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Pass gradient straight through (no modification)
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            grad_input: Same gradient (straight-through)
        """
        # Straight-through: gradient flows as if threshold didn't exist
        return grad_output


def ste_sample_gates(logAlpha, l0_params, temperature=1.0):
    """
    Sample binary gates using STE
    
    Args:
        logAlpha: Edge logits [B, N, N]
        l0_params: L0 parameters (uses gamma, zeta)
        temperature: Temperature for sigmoid
    
    Returns:
        gates: Binary gates {0, 1} [B, N, N]
        probs: Gate probabilities [B, N, N] (for regularization)
    """
    # Compute probabilities
    probs = torch.sigmoid(logAlpha / temperature)
    
    # Apply stretching (like Hard-Concrete)
    stretched_probs = probs * (l0_params.zeta - l0_params.gamma) + l0_params.gamma
    stretched_probs = torch.clamp(stretched_probs, 0, 1)
    
    # Binarize with STE
    gates = STEBinarize.apply(stretched_probs)
    
    return gates, stretched_probs


def get_expected_l0_ste(logAlpha, l0_params):
    """
    Compute expected L0 penalty for STE
    
    Uses same formula as Hard-Concrete (expected number of active gates)
    
    Args:
        logAlpha: Edge logits [B, N, N]
        l0_params: L0 parameters
    
    Returns:
        l0_penalty: Expected L0 penalty (scalar)
    """
    # Probability of gate being active
    probs = torch.sigmoid(logAlpha)
    
    # Apply stretching
    stretched_probs = probs * (l0_params.zeta - l0_params.gamma) + l0_params.gamma
    stretched_probs = torch.clamp(stretched_probs, 0, 1)
    
    # Expected L0 = sum of probabilities
    l0_penalty = stretched_probs.sum()
    
    return l0_penalty


class STERegularizerParams:
    """
    Parameters for STE regularization
    Same as Hard-Concrete but used differently
    """
    def __init__(self, gamma=-0.1, zeta=1.1):
        self.gamma = gamma  # Lower stretch bound
        self.zeta = zeta    # Upper stretch bound
    
    def update_params(self, gamma=None, zeta=None):
        if gamma is not None:
            self.gamma = gamma
        if zeta is not None:
            self.zeta = zeta
