import torch
import torch.nn as nn
import numpy as np


class L0RegularizerParams:
    """Parameters for L0 regularization with Hard-Concrete distribution"""
    
    def __init__(self, gamma=-0.1, zeta=1.1, beta_l0=0.66, eps=1e-20):
        self.sig = nn.Sigmoid()
        self.gamma = gamma
        self.zeta = zeta
        self.beta_l0 = beta_l0
        self.eps = eps
        self._update_const1()
    
    def _update_const1(self):
        """Update const1 when parameters change"""
        self.const1 = self.beta_l0 * np.log(-self.gamma/self.zeta + self.eps)
    
    def update_params(self, gamma=None, zeta=None, beta_l0=None):
        """Update parameters and recalculate dependent values"""
        if gamma is not None:
            self.gamma = gamma
        if zeta is not None:
            self.zeta = zeta
        if beta_l0 is not None:
            self.beta_l0 = beta_l0
        self._update_const1()
        return self


# Default instance
l0_params = L0RegularizerParams()


def l0_train(logAlpha, min=0, max=1, params=None, temperature=None):
    """L0 regularization for training - stochastic gates"""
    if params is None:
        params = l0_params
    
    effective_beta = temperature if temperature is not None else params.beta_l0
    
    U = torch.rand(logAlpha.size()).type_as(logAlpha) + params.eps
    s = params.sig((torch.log(U / (1 - U)) + logAlpha) / effective_beta)
    s_bar = s * (params.zeta - params.gamma) + params.gamma
    mask = torch.clamp(s_bar, min, max)
    
    return mask


def l0_test(logAlpha, min=0, max=1, params=None, temperature=None):
    """L0 regularization for testing - deterministic"""
    if params is None:
        params = l0_params
    
    effective_beta = temperature if temperature is not None else params.beta_l0
    
    s = params.sig(logAlpha / effective_beta)
    s_bar = s * (params.zeta - params.gamma) + params.gamma
    mask = torch.clamp(s_bar, min, max)
    
    return mask


def get_loss2(logAlpha, params=None):
    """Calculate L0 regularization penalty"""
    if params is None:
        params = l0_params
    
    return params.sig(logAlpha - params.const1)


# Backward compatibility
gamma = l0_params.gamma
zeta = l0_params.zeta
beta_l0 = l0_params.beta_l0
eps = l0_params.eps
sig = l0_params.sig
const1 = l0_params.const1
