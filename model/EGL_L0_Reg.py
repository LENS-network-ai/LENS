"""
L0 Regularization with Target Density Control

L0 regularization (Hard-Concrete or ARM) with adaptive lambda and density loss
L_total = λ_eff * L_L0 + λ_ρ * (ρ - ρ_target)²
"""

import torch
import torch.nn as nn
import numpy as np
from model.L0Utils import get_loss2, L0RegularizerParams
from model.L0Utils_ARM import get_expected_l0_arm, ARML0RegularizerParams


def compute_density(edge_weights, adj_matrix):
    """Compute graph density: fraction of edges retained"""
    edge_mask = (adj_matrix > 0).float()
    num_edges = edge_mask.sum()
    active_edges = (edge_weights * edge_mask).sum()
    density = active_edges / (num_edges + 1e-8)
    return density


def compute_keep_probabilities(logits, l0_params, l0_method='hard-concrete'):
    """Compute keep probabilities from edge logits"""
    if l0_method == 'hard-concrete':
        if isinstance(l0_params, L0RegularizerParams):
            keep_probs = l0_params.sig(logits - l0_params.const1)
        else:
            keep_probs = torch.sigmoid(logits)
    elif l0_method == 'arm':
        keep_probs = torch.sigmoid(logits)
    else:
        keep_probs = torch.sigmoid(logits)
    
    return keep_probs


class L0Regularization:
    """
    L0 regularization with adaptive lambda and density control
    
    Loss: λ_eff * L_L0 + λ_ρ * (ρ - ρ_target)²
    - λ_eff adapts based on current density
    - Scheduled warmup and ramp-up for stability
    """
    
    def __init__(self, 
                 lambda_reg=0.0001, 
                 lambda_density=0.03,
                 target_density=0.30,
                 warmup_epochs=15, 
                 ramp_epochs=20,
                 l0_params=None, 
                 l0_method='hard-concrete',
                 alpha_min=0.2,
                 alpha_max=2.0,
                 enable_adaptive_lambda=True,
                 enable_density_loss=True):
        
        # Core parameters
        self.base_lambda = lambda_reg
        self.current_lambda = 0.0
        self.l0_method = l0_method
        self.logits_storage = {}
        
        # Density control
        self.target_density = target_density
        self.base_lambda_density = lambda_density
        self.current_lambda_density = 0.0
        self.enable_adaptive_lambda = enable_adaptive_lambda
        self.enable_density_loss = enable_density_loss
        
        # Adaptive lambda bounds
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Schedule
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        
        # L0 params
        if l0_params is not None:
            self.l0_params = l0_params
        else:
            if l0_method == 'hard-concrete':
                self.l0_params = L0RegularizerParams()
            elif l0_method == 'arm':
                self.l0_params = ARML0RegularizerParams()
            else:
                self.l0_params = L0RegularizerParams()
        
        # Print config
        print(f"\n{'='*70}")
        print(f"[L0Reg] Penalty Mode with Density Control")
        print(f"{'='*70}")
        print(f"  L0 Method: {l0_method}")
        print(f"\n  Parameters:")
        print(f"  Base λ: {lambda_reg:.6f}")
        print(f"  Base λ_ρ: {lambda_density:.6f}")
        print(f"  Target Density: {target_density*100:.1f}%")
        print(f"  Adaptive λ Range: [{alpha_min}, {alpha_max}]")
        print(f"  Adaptive λ: {'Enabled' if enable_adaptive_lambda else 'Disabled'}")
        print(f"  Density Loss: {'Enabled' if enable_density_loss else 'Disabled'}")
        print(f"\n  Warmup Epochs: {warmup_epochs}")
        print(f"  Ramp Epochs: {ramp_epochs}")
        print(f"{'='*70}\n")
    
    def clear_logits(self):
        """Clear stored logits"""
        self.logits_storage = {}
    
    def store_logits(self, batch_idx, logits):
        """Store edge logits for batch"""
        self.logits_storage[batch_idx] = logits
    
    def update_lambda(self, current_epoch):
        """
        Three-phase lambda schedule:
        - Warmup: 0 → 0.1 * λ_base
        - Ramp: 0.1 * λ_base → λ_base
        - Plateau: λ_base
        """
        self.current_epoch = current_epoch
        
        if current_epoch < self.warmup_epochs:
            progress = current_epoch / self.warmup_epochs
            self.current_lambda = progress * 0.1 * self.base_lambda
        elif current_epoch < self.warmup_epochs + self.ramp_epochs:
            post_warmup_epochs = current_epoch - self.warmup_epochs
            progress = post_warmup_epochs / self.ramp_epochs
            self.current_lambda = self.base_lambda * (0.1 + 0.9 * progress)
        else:
            self.current_lambda = self.base_lambda
    
    def update_lambda_density(self, current_epoch):
        """
        Two-phase density lambda schedule:
        - Warmup: 0 → 0.5 * λ_ρ
        - Ramp: 0.5 * λ_ρ → λ_ρ
        """
        if current_epoch < self.warmup_epochs:
            progress = current_epoch / self.warmup_epochs
            self.current_lambda_density = progress * 0.5 * self.base_lambda_density
        else:
            post_warmup_epochs = current_epoch - self.warmup_epochs
            ramp_duration = self.ramp_epochs / 2
            
            if post_warmup_epochs < ramp_duration:
                progress = post_warmup_epochs / ramp_duration
                scale = 0.5 + 0.5 * progress
            else:
                scale = 1.0
            
            self.current_lambda_density = scale * self.base_lambda_density
    
    def update_temperature(self, current_epoch, initial_temp=5.0):
        """
        Three-phase temperature schedule with cosine annealing:
        - Warmup: constant at τ_init
        - Anneal: cosine decay to τ_min
        - Plateau: τ_min
        """
        tau_init = initial_temp
        tau_min = 1.0
        t_warmup = self.warmup_epochs
        t_anneal = self.warmup_epochs + self.ramp_epochs
        mu_min = 0.1
        
        if current_epoch < t_warmup:
            temperature = tau_init
        elif current_epoch <= t_anneal:
            progress = (current_epoch - t_warmup) / (t_anneal - t_warmup)
            progress = min(1.0, max(0.0, progress))
            mu = mu_min + (1 - mu_min) * (np.cos(progress * np.pi) + 1) / 2
            temperature = max(tau_min, tau_init * mu)
        else:
            temperature = tau_min
        
        return temperature
    
    def update_all_schedules(self, current_epoch, initial_temp=5.0):
        """Update all schedules at once"""
        self.current_epoch = current_epoch
        temperature = self.update_temperature(current_epoch, initial_temp)
        self.update_lambda(current_epoch)
        self.update_lambda_density(current_epoch)
        
        return {
            'lambda': self.current_lambda,
            'lambda_density': self.current_lambda_density,
            'temperature': temperature,
            'epoch': current_epoch,
        }
    
    def compute_adaptive_lambda(self, current_density):
        """
        Adaptive lambda based on density error:
        α = clip(1 + (ρ - ρ_target), α_min, α_max)
        λ_eff = λ_base * α
        """
        if not self.enable_adaptive_lambda:
            return self.current_lambda, 1.0
        
        if isinstance(current_density, torch.Tensor):
            density_value = current_density.item()
        else:
            density_value = current_density
        
        density_error = density_value - self.target_density
        alpha = 1.0 + density_error
        alpha = max(self.alpha_min, min(self.alpha_max, alpha))
        lambda_eff = self.current_lambda * alpha
        
        return lambda_eff, alpha
    
    def compute_density_loss(self, current_density):
        """Density loss: λ_ρ * (ρ - ρ_target)²"""
        if not self.enable_density_loss or self.current_lambda_density == 0.0:
            device = current_density.device if isinstance(current_density, torch.Tensor) else 'cpu'
            return torch.tensor(0.0, device=device)
        
        density_error = current_density - self.target_density
        density_loss = self.current_lambda_density * (density_error ** 2)
        
        return density_loss
    
    def compute_l0_loss(self):
        """Compute L0 penalty from stored logits"""
        if len(self.logits_storage) == 0:
            return torch.tensor(0.0)
        
        first_key = next(iter(self.logits_storage))
        device = self.logits_storage[first_key].device
        l0_loss = torch.tensor(0.0, device=device)
        
        for batch_idx, logits in self.logits_storage.items():
            if self.l0_method == 'hard-concrete':
                batch_l0 = get_loss2(logits, params=self.l0_params).sum()
            elif self.l0_method == 'arm':
                batch_l0 = get_expected_l0_arm(logits, self.l0_params)
            else:
                batch_l0 = get_loss2(logits, params=self.l0_params).sum()
            
            l0_loss = l0_loss + batch_l0
        
        l0_loss = l0_loss / len(self.logits_storage)
        return l0_loss
    
    def compute_regularization_with_l0(self, l0_penalty, edge_weights, adj_matrix, 
                                        return_stats=False):
        """
        Compute regularization: λ_eff * L_L0 + λ_ρ * (ρ - ρ_target)²
        """
        device = edge_weights.device
        current_density = compute_density(edge_weights, adj_matrix)
        
        # Adaptive lambda
        if self.enable_adaptive_lambda:
            lambda_eff, alpha = self.compute_adaptive_lambda(current_density)
        else:
            lambda_eff = self.current_lambda
            alpha = 1.0
        
        # L0 loss
        l0_loss = lambda_eff * l0_penalty
        
        # Density loss
        if self.enable_density_loss and self.current_epoch >= self.warmup_epochs:
            density_deviation = torch.abs(current_density - self.target_density)
            density_loss = self.current_lambda_density * density_deviation
        else:
            density_loss = torch.tensor(0.0, device=device)
        
        # Total
        reg_loss = l0_loss + density_loss
        
        if return_stats:
            stats = {
                'l0_loss': l0_loss.item() if isinstance(l0_loss, torch.Tensor) else l0_loss,
                'density_loss': density_loss.item() if isinstance(density_loss, torch.Tensor) else density_loss,
                'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
                'lambda_eff': lambda_eff if isinstance(lambda_eff, float) else lambda_eff.item(),
                'alpha': alpha,
                'current_density': current_density.item() if isinstance(current_density, torch.Tensor) else current_density,
            }
            return reg_loss, stats
        
        return reg_loss
    
    def compute_regularization(self, edge_weights, adj_matrix, return_stats=True):
        """Compute complete regularization loss"""
        if self.current_lambda == 0.0 or not edge_weights.requires_grad:
            if return_stats:
                return torch.tensor(0.0, device=edge_weights.device), {
                    'total_reg_loss': 0.0,
                    'l0_loss': 0.0,
                    'density_loss': 0.0,
                    'lambda_eff': 0.0,
                    'alpha': 1.0,
                    'current_density': 0.0,
                    'target_density': self.target_density,
                    'density_deviation': 0.0,
                }
            else:
                return torch.tensor(0.0, device=edge_weights.device)
        
        l0_loss = self.compute_l0_loss()
        
        return self.compute_regularization_with_l0(
            l0_loss, edge_weights, adj_matrix, return_stats
        )
    
    def get_keep_probabilities(self, logits):
        """Get keep probabilities from logits"""
        return compute_keep_probabilities(logits, self.l0_params, self.l0_method)
    
    def get_statistics(self):
        """Get current state for logging"""
        return {
            'current_epoch': self.current_epoch,
            'l0_method': self.l0_method,
            'warmup_epochs': self.warmup_epochs,
            'ramp_epochs': self.ramp_epochs,
            'base_lambda': self.base_lambda,
            'current_lambda': self.current_lambda,
            'base_lambda_density': self.base_lambda_density,
            'current_lambda_density': self.current_lambda_density,
            'target_density': self.target_density,
            'alpha_min': self.alpha_min,
            'alpha_max': self.alpha_max,
            'enable_adaptive_lambda': self.enable_adaptive_lambda,
            'enable_density_loss': self.enable_density_loss,
        }
    
    def print_configuration(self):
        """Print current config"""
        stats = self.get_statistics()
        print(f"\n{'='*70}")
        print(f"L0 Regularization Configuration")
        print(f"{'='*70}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*70}\n")
    
    def __repr__(self):
        return (f"L0Regularization:\n"
                f"  l0_method={self.l0_method},\n"
                f"  base_lambda={self.base_lambda:.6f},\n"
                f"  current_lambda={self.current_lambda:.6f},\n"
                f"  target_density={self.target_density:.2f},\n"
                f"  current_lambda_density={self.current_lambda_density:.6f},\n"
                f"  adaptive_lambda={'enabled' if self.enable_adaptive_lambda else 'disabled'},\n"
                f"  density_loss={'enabled' if self.enable_density_loss else 'disabled'},\n"
                f"  current_epoch={self.current_epoch},\n"
                f"  warmup_epochs={self.warmup_epochs}\n"
                f")")


# Standalone schedules
class LambdaSchedule:
    """Standalone lambda schedule"""
    
    def __init__(self, base_lambda=0.0001, warmup_epochs=15, ramp_epochs=20):
        self.base_lambda = base_lambda
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
    
    def get_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            return progress * 0.1 * self.base_lambda
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            post_warmup = epoch - self.warmup_epochs
            progress = post_warmup / self.ramp_epochs
            return self.base_lambda * (0.1 + 0.9 * progress)
        else:
            return self.base_lambda


class DensityLambdaSchedule:
    """Standalone density lambda schedule"""
    
    def __init__(self, base_lambda_density=0.03, warmup_epochs=15, ramp_epochs=10):
        self.base_lambda_density = base_lambda_density
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
    
    def get_lambda_density(self, epoch):
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            return progress * 0.5 * self.base_lambda_density
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            post_warmup = epoch - self.warmup_epochs
            progress = post_warmup / self.ramp_epochs
            scale = 0.5 + 0.5 * progress
            return scale * self.base_lambda_density
        else:
            return self.base_lambda_density


class TemperatureSchedule:
    """Standalone temperature schedule with cosine annealing"""
    
    def __init__(self, tau_init=5.0, tau_min=1.0, t_warmup=15, t_anneal=35):
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.t_warmup = t_warmup
        self.t_anneal = t_anneal
        self.mu_min = 0.1
    
    def get_temperature(self, epoch):
        if epoch < self.t_warmup:
            return self.tau_init
        elif epoch <= self.t_anneal:
            progress = (epoch - self.t_warmup) / (self.t_anneal - self.t_warmup)
            progress = min(1.0, max(0.0, progress))
            mu = self.mu_min + (1 - self.mu_min) * (np.cos(progress * np.pi) + 1) / 2
            return max(self.tau_min, self.tau_init * mu)
        else:
            return self.tau_min
