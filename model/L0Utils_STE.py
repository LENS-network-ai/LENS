# model/L0Utils_STE.py
"""
Straight-Through Estimator (STE) for L0 Regularization
Binary gates with straight-through gradients
"""

import torch
import torch.nn as nn


class STEBinarize(torch.autograd.Function):
    """
    Straight-Through Estimator for binary gates
    Forward: Hard threshold (binary output)
    Backward: Straight-through gradient
    """
    
    @staticmethod
    def forward(ctx, input):
        """Binary gate based on threshold"""
        output = (input > 0.5).float()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradient straight through"""
        return grad_output


def ste_sample_gates(logAlpha, temperature=1.0):
    """
    Sample binary gates using STE
    
    Args:
        logAlpha: Edge logits [B, N, N]
        temperature: Temperature for sigmoid
    
    Returns:
        gates: Binary gates {0, 1}
        probs: Gate probabilities (for L0 penalty)
    """
    probs = torch.sigmoid(logAlpha / temperature)
    gates = STEBinarize.apply(probs)
    
    return gates, probs


def get_expected_l0_ste(logAlpha, temperature=1.0):
    """
    Compute expected L0 penalty for STE
    
    Args:
        logAlpha: Edge logits
        temperature: Temperature
    
    Returns:
        l0_penalty: Expected number of active gates
    """
    probs = torch.sigmoid(logAlpha / temperature)
    l0_penalty = probs.sum()
    
    return l0_penalty


class STERegularizerParams:
    """Parameters for STE regularization (empty for API consistency)"""
    
    def __init__(self):
        pass
