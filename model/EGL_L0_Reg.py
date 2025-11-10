# model/EGL_L0_Reg.py
"""
EGLasso Regularization with Target Density Control

Supports:
- L0 regularization with Hard-Concrete (default)
- L0 regularization with ARM
- Exclusive Group Lasso (EGL)
- Target density control with adaptive lambda
- Density loss for explicit sparsity targets

Author: Updated with density control mechanism
"""

import torch
import torch.nn as nn
import numpy as np
from model.L0Utils import get_loss2, L0RegularizerParams
from model.L0Utils_ARM import get_expected_l0_arm, ARML0RegularizerParams


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
# model/EGL_L0_Reg.py - ADD THIS METHOD

def compute_density(edge_weights, adj_matrix):
    """
    Compute current graph density: ρ = (1/|E|) Σ_{(i,j)∈E} z_ij
    
    Args:
        edge_weights: Edge gate values [batch_size, num_nodes, num_nodes]
        adj_matrix: Original adjacency matrix [batch_size, num_nodes, num_nodes]
    
    Returns:
        density: Scalar tensor - fraction of edges retained (0 to 1)
    """
    # Create mask for valid edges
    edge_mask = (adj_matrix > 0).float()
    
    # Count total possible edges
    num_edges = edge_mask.sum()
    
    # Count active edges (weighted sum)
    active_edges = (edge_weights * edge_mask).sum()
    
    # Compute density (with epsilon for numerical stability)
    density = active_edges / (num_edges + 1e-8)
    
    return density


def compute_keep_probabilities(logits, l0_params, l0_method='hard-concrete'):
    """
    Compute keep probabilities π_ij from logits
    
    Args:
        logits: Edge logits [batch_size, num_nodes, num_nodes]
        l0_params: L0 regularizer parameters
        l0_method: 'hard-concrete' or 'arm'
    
    Returns:
        keep_probs: Keep probabilities [batch_size, num_nodes, num_nodes]
    """
    if l0_method == 'hard-concrete':
        if isinstance(l0_params, L0RegularizerParams):
            # π_ij = σ(log α - τ log(-γ/ζ))
            keep_probs = l0_params.sig(logits - l0_params.const1)
        else:
            keep_probs = torch.sigmoid(logits)
    
    elif l0_method == 'arm':
        # For ARM, keep probability is sigmoid
        keep_probs = torch.sigmoid(logits)
    
    else:
        # Fallback
        keep_probs = torch.sigmoid(logits)
    
    return keep_probs


# ============================================================================
# MAIN REGULARIZATION CLASS
# ============================================================================

class EGLassoRegularization:
    """
    Edge-Graph Lasso Regularization with Target Density Control
    
    Features:
    - L0 regularization (Hard-Concrete or ARM)
    - Exclusive Group Lasso (EGL)
    - Target density control with adaptive lambda
    - Scheduled warmup for lambda and temperature
    - Automatic density correction via opposing gradient forces
    """
    
    def __init__(self, 
                 lambda_reg=0.0001, 
                 lambda_density=0.03,
                 target_density=0.30,
                 reg_mode='l0', 
                 warmup_epochs=15, 
                 ramp_epochs=20,
                 l0_params=None, 
                 l0_method='hard-concrete',
                 alpha_min=0.2,
                 alpha_max=2.0,
                 enable_adaptive_lambda=True,
                 enable_density_loss=True):
        """
        Initialize the regularization module with density control
        
        Args:
            lambda_reg: Base L0 regularization strength (λ_base)
            lambda_density: Density loss weight (λ_ρ)
            target_density: Target edge retention rate (ρ_target) in [0, 1]
            reg_mode: Regularization type ('l0' or 'egl')
            warmup_epochs: Number of warmup epochs (E_warmup)
            ramp_epochs: Number of ramp-up epochs after warmup
            l0_params: L0RegularizerParams or ARML0RegularizerParams instance
            l0_method: 'hard-concrete' or 'arm'
            alpha_min: Minimum adaptive scaling factor
            alpha_max: Maximum adaptive scaling factor
            enable_adaptive_lambda: Whether to use adaptive lambda mechanism
            enable_density_loss: Whether to use density loss
        """
        # ====================================================================
        # CORE REGULARIZATION PARAMETERS
        # ====================================================================
        self.base_lambda = lambda_reg
        self.current_lambda = 0.0  # Will increase during training
        self.reg_mode = reg_mode
        self.l0_method = l0_method
        self.logits_storage = {}  # For L0 regularization
        
        # ====================================================================
        # DENSITY CONTROL PARAMETERS
        # ====================================================================
        self.target_density = target_density  # ρ_target
        self.base_lambda_density = lambda_density  # λ_ρ^base
        self.current_lambda_density = 0.0  # Will increase during training
        self.enable_adaptive_lambda = enable_adaptive_lambda
        self.enable_density_loss = enable_density_loss
        
        # Adaptive lambda bounds
        self.alpha_min = alpha_min  # Minimum scaling (reduces pruning)
        self.alpha_max = alpha_max  # Maximum scaling (increases pruning)
        
        # ====================================================================
        # SCHEDULE PARAMETERS
        # ====================================================================
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        
        # ====================================================================
        # L0 REGULARIZER PARAMETERS
        # ====================================================================
        if l0_params is not None:
            self.l0_params = l0_params
        else:
            # Create default params based on method
            if l0_method == 'hard-concrete':
                self.l0_params = L0RegularizerParams()
            elif l0_method == 'arm':
                self.l0_params = ARML0RegularizerParams()
            else:
                self.l0_params = L0RegularizerParams()
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"[EGLassoReg] Initialized with Target Density Control")
        print(f"{'='*70}")
        print(f"  Regularization Mode: {reg_mode}")
        print(f"  L0 Method: {l0_method}")
        print(f"  Base λ: {lambda_reg:.6f}")
        print(f"  Base λ_ρ: {lambda_density:.6f}")
        print(f"  Target Density: {target_density*100:.1f}%")
        print(f"  Warmup Epochs: {warmup_epochs}")
        print(f"  Ramp Epochs: {ramp_epochs}")
        print(f"  Adaptive λ Range: [{alpha_min}, {alpha_max}]")
        print(f"  Adaptive λ Enabled: {enable_adaptive_lambda}")
        print(f"  Density Loss Enabled: {enable_density_loss}")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # LOGIT STORAGE FOR L0 REGULARIZATION
    # ========================================================================
    def compute_regularization_with_l0(self, l0_penalty, edge_weights, adj_matrix, 
                                    return_stats=False):
     """
     Compute regularization when L0 penalty is already computed
    
     This is used for Hard-Concrete, ARM, and STE where L0 penalty
     is computed differently for each method
    
     Args:
        l0_penalty: Pre-computed L0 penalty (scalar tensor)
        edge_weights: Edge weights [B, N, N]
        adj_matrix: Original adjacency [B, N, N]
        return_stats: If True, return detailed statistics
    
     Returns:
        reg_loss: Total regularization loss
        stats: Dictionary with loss components (if return_stats=True)
     """
     # Compute current density
     current_density = compute_density(edge_weights, adj_matrix)
    
     # Compute effective lambda (with adaptive mechanism if enabled)
     if self.enable_adaptive_lambda:
        lambda_eff, alpha = self.compute_adaptive_lambda(current_density)
     else:
        lambda_eff = self.current_lambda
        alpha = 1.0
    
     # L0 regularization loss
     l0_loss = lambda_eff * l0_penalty
    
     # Density loss (if enabled)
     if self.enable_density_loss and self.current_epoch >= self.warmup_epochs:
        density_deviation = torch.abs(current_density - self.target_density)
        density_loss = self.current_lambda_density * density_deviation
     else:
        density_loss = torch.tensor(0.0, device=edge_weights.device)
    
     # Total regularization
     reg_loss = l0_loss + density_loss
    
     if return_stats:
        stats = {
            'l0_loss': l0_loss.item() if isinstance(l0_loss, torch.Tensor) else l0_loss,
            'density_loss': density_loss.item() if isinstance(density_loss, torch.Tensor) else density_loss,
            'lambda_eff': lambda_eff if isinstance(lambda_eff, float) else lambda_eff.item(),
            'alpha': alpha,
            'current_density': current_density.item() if isinstance(current_density, torch.Tensor) else current_density,
        }
        return reg_loss, stats
    
     return reg_loss
    def clear_logits(self):
        """Clear stored logits for L0 regularization"""
        self.logits_storage = {}
    
    def store_logits(self, batch_idx, logits):
        """
        Store logits for a batch for L0 regularization
        
        Args:
            batch_idx: Batch index
            logits: Edge logits [num_nodes, num_nodes]
        """
        self.logits_storage[batch_idx] = logits
    
    # ========================================================================
    # SCHEDULE UPDATES
    # ========================================================================
    
    def update_lambda(self, current_epoch):
        """
        Update L0 lambda regularization strength: λ(e)
        
        Three-phase linear schedule:
        - Phase 1 (warmup): Linear from 0 to 0.1 × λ_base over E_warmup epochs
        - Phase 2 (ramp): Linear from 0.1 × λ_base to λ_base over E_ramp epochs
        - Phase 3 (plateau): Fixed at λ_base
        
        Args:
            current_epoch: Current training epoch
        """
        self.current_epoch = current_epoch
        
        if current_epoch < self.warmup_epochs:
            # Phase 1: Warmup - linear increase to 10% of base
            progress = current_epoch / self.warmup_epochs
            self.current_lambda = progress * 0.1 * self.base_lambda
        
        elif current_epoch < self.warmup_epochs + self.ramp_epochs:
            # Phase 2: Ramp - linear increase from 10% to 100%
            post_warmup_epochs = current_epoch - self.warmup_epochs
            progress = post_warmup_epochs / self.ramp_epochs
            self.current_lambda = self.base_lambda * (0.1 + 0.9 * progress)
        
        else:
            # Phase 3: Plateau - fixed at full strength
            self.current_lambda = self.base_lambda
    
    def update_lambda_density(self, current_epoch):
        """
        Update density lambda: λ_ρ(e)
        
        Two-phase schedule:
        - Phase 1 (warmup): Linear from 0 to 0.5 × λ_ρ^base over E_warmup epochs
        - Phase 2 (ramp): Linear from 0.5 × λ_ρ^base to λ_ρ^base over E_ramp/2 epochs
        
        Args:
            current_epoch: Current training epoch
        """
        if current_epoch < self.warmup_epochs:
            # Phase 1: Warmup - linear increase to 50% of base
            progress = current_epoch / self.warmup_epochs
            self.current_lambda_density = progress * 0.5 * self.base_lambda_density
        
        else:
            # Phase 2: Ramp - linear increase from 50% to 100%
            post_warmup_epochs = current_epoch - self.warmup_epochs
            ramp_duration = self.ramp_epochs / 2  # Faster ramp for density
            
            if post_warmup_epochs < ramp_duration:
                progress = post_warmup_epochs / ramp_duration
                scale = 0.5 + 0.5 * progress
            else:
                scale = 1.0
            
            self.current_lambda_density = scale * self.base_lambda_density
    
    def update_temperature(self, current_epoch, initial_temp=5.0):
        """
        Update temperature based on current epoch: τ(e)
        
        Three-phase cosine annealing:
        - Phase 1: Constant at τ_init during warmup
        - Phase 2: Cosine decay from τ_init to τ_min
        - Phase 3: Plateau at τ_min
        
        Args:
            current_epoch: Current training epoch
            initial_temp: Initial temperature (τ_init)
        
        Returns:
            temperature: Current temperature value
        """
        tau_init = initial_temp
        tau_min = 1.0
        t_warmup = self.warmup_epochs
        t_anneal = self.warmup_epochs + self.ramp_epochs
        mu_min = 0.1
        
        if current_epoch < t_warmup:
            # Phase 1: Constant during warmup
            temperature = tau_init
        
        elif current_epoch <= t_anneal:
            # Phase 2: Cosine annealing
            progress = (current_epoch - t_warmup) / (t_anneal - t_warmup)
            progress = min(1.0, max(0.0, progress))
            mu = mu_min + (1 - mu_min) * (np.cos(progress * np.pi) + 1) / 2
            temperature = max(tau_min, tau_init * mu)
        
        else:
            # Phase 3: Plateau at minimum
            temperature = tau_min
        
        return temperature
    
    def update_all_schedules(self, current_epoch, initial_temp=5.0):
        """
        Convenience function to update all schedules at once
        
        Args:
            current_epoch: Current training epoch
            initial_temp: Initial temperature
        
        Returns:
            dict with updated values
        """
        self.update_lambda(current_epoch)
        self.update_lambda_density(current_epoch)
        temperature = self.update_temperature(current_epoch, initial_temp)
        
        return {
            'lambda': self.current_lambda,
            'lambda_density': self.current_lambda_density,
            'temperature': temperature,
            'epoch': current_epoch
        }
    
    # ========================================================================
    # ADAPTIVE LAMBDA MECHANISM
    # ========================================================================
    
    def compute_adaptive_lambda(self, current_density):
        """
        Compute adaptive lambda: λ_eff(t) = λ_base · α(t)
        
        Adaptive scaling factor:
            α(t) = clip(1 + [ρ(t) - ρ_target], α_min, α_max)
        
        Behavior:
        - When ρ > ρ_target (too dense): α > 1, increase pruning
        - When ρ < ρ_target (too sparse): α < 1, decrease pruning
        - When ρ ≈ ρ_target (at target): α ≈ 1, maintain equilibrium
        
        Args:
            current_density: Current graph density ρ(t) ∈ [0, 1]
        
        Returns:
            lambda_eff: Effective lambda value (scalar or tensor)
            alpha: Adaptive scaling factor (float)
        """
        if not self.enable_adaptive_lambda:
            # Adaptive mechanism disabled - use current lambda directly
            return self.current_lambda, 1.0
        
        # Convert density to scalar if it's a tensor
        if isinstance(current_density, torch.Tensor):
            density_value = current_density.item()
        else:
            density_value = current_density
        
        # Compute density error
        density_error = density_value - self.target_density
        
        # Compute adaptive scaling factor
        alpha = 1.0 + density_error
        
        # Clip to bounds
        alpha = max(self.alpha_min, min(self.alpha_max, alpha))
        
        # Compute effective lambda
        lambda_eff = self.current_lambda * alpha
        
        return lambda_eff, alpha
    
    # ========================================================================
    # DENSITY LOSS
    # ========================================================================
    
    def compute_density_loss(self, current_density):
        """
        Compute density loss: L_density = λ_ρ · (ρ - ρ_target)²
        
        Gradient behavior:
        - When ρ > ρ_target: positive gradient → pushes edges down (pruning)
        - When ρ < ρ_target: negative gradient → pulls edges up (retention)
        
        Args:
            current_density: Current graph density ρ(t)
        
        Returns:
            density_loss: Density loss value (tensor)
        """
        if not self.enable_density_loss or self.current_lambda_density == 0.0:
            return torch.tensor(0.0, device=current_density.device if isinstance(current_density, torch.Tensor) else 'cpu')
        
        # Compute density error
        density_error = current_density - self.target_density
        
        # Quadratic penalty
        density_loss = self.current_lambda_density * (density_error ** 2)
        
        return density_loss
    
    # ========================================================================
    # L0 REGULARIZATION
    # ========================================================================
    
    def compute_l0_loss(self):
        """
        Compute L0 penalty from stored logits (without lambda scaling)
        
        Returns:
            l0_loss: L0 penalty (tensor)
        """
        if len(self.logits_storage) == 0:
            return torch.tensor(0.0)
        
        # Get device from first stored logits
        first_key = next(iter(self.logits_storage))
        device = self.logits_storage[first_key].device
        
        l0_loss = torch.tensor(0.0, device=device)
        
        for batch_idx, logits in self.logits_storage.items():
            # Calculate L0 loss based on method
            if self.l0_method == 'hard-concrete':
                batch_l0 = get_loss2(logits, params=self.l0_params).sum()
            
            elif self.l0_method == 'arm':
                batch_l0 = get_expected_l0_arm(logits, self.l0_params)
            
            else:
                # Fallback to hard-concrete
                batch_l0 = get_loss2(logits, params=self.l0_params).sum()
            
            l0_loss = l0_loss + batch_l0
        
        # Average over batches
        l0_loss = l0_loss / len(self.logits_storage)
        
        return l0_loss
    
    # ========================================================================
    # MAIN REGULARIZATION COMPUTATION
    # ========================================================================
    
    def compute_regularization(self, edge_weights, adj_matrix, 
                               return_stats=True):
        """
        Compute complete regularization loss with density control
        
        Total loss:
            L_total = λ_eff · L_L0 + L_density
        
        where:
            - λ_eff = λ_base · α(ρ) (adaptive)
            - L_L0 = Σ π_ij (expected number of edges)
            - L_density = λ_ρ · (ρ - ρ_target)²
        
        Args:
            edge_weights: Edge gate values [batch_size, num_nodes, num_nodes]
            adj_matrix: Original adjacency [batch_size, num_nodes, num_nodes]
            return_stats: Whether to return detailed statistics
        
        Returns:
            If return_stats=False:
                reg_loss: Total regularization loss (tensor)
            If return_stats=True:
                (reg_loss, stats_dict): Loss and statistics dictionary
        """
        # Check if regularization is active
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
                    'density_deviation': 0.0
                }
            else:
                return torch.tensor(0.0, device=edge_weights.device)
        
        device = edge_weights.device
        
        # ====================================================================
        # 1. COMPUTE CURRENT DENSITY
        # ====================================================================
        current_density = compute_density(edge_weights, adj_matrix)
        
        # ====================================================================
        # 2. COMPUTE ADAPTIVE LAMBDA
        # ====================================================================
        lambda_eff, alpha = self.compute_adaptive_lambda(current_density)
        
        # ====================================================================
        # 3. COMPUTE L0 LOSS
        # ====================================================================
        if self.reg_mode == 'l0':
            l0_loss = self.compute_l0_loss()
        
        elif self.reg_mode == 'egl':
            # Exclusive Group Lasso - group by nodes
            edge_mask = (adj_matrix > 0).float()
            
            # Sum weights per source node
            source_sum = torch.sum(edge_weights * edge_mask, dim=2)
            source_reg = torch.sum(source_sum ** 2)
            
            # Sum weights per target node
            target_sum = torch.sum(edge_weights * edge_mask, dim=1)
            target_reg = torch.sum(target_sum ** 2)
            
            # Average
            batch_size = adj_matrix.shape[0]
            l0_loss = (source_reg + target_reg) / (2 * batch_size)
        
        else:
            raise ValueError(f"Unsupported regularization mode: {self.reg_mode}")
        
        # ====================================================================
        # 4. COMPUTE DENSITY LOSS
        # ====================================================================
        density_loss = self.compute_density_loss(current_density)
        
        # ====================================================================
        # 5. COMBINE LOSSES
        # ====================================================================
        # Apply adaptive lambda to L0 loss
        weighted_l0_loss = lambda_eff * l0_loss
        
        # Total regularization
        total_reg_loss = weighted_l0_loss + density_loss
        
        # ====================================================================
        # 6. PREPARE STATISTICS
        # ====================================================================
        if return_stats:
            stats = {
                'total_reg_loss': total_reg_loss.item(),
                'l0_loss': l0_loss.item(),
                'weighted_l0_loss': weighted_l0_loss.item(),
                'density_loss': density_loss.item(),
                'lambda_base': self.current_lambda,
                'lambda_eff': lambda_eff if isinstance(lambda_eff, float) else lambda_eff.item(),
                'lambda_density': self.current_lambda_density,
                'alpha': alpha,
                'current_density': current_density.item(),
                'target_density': self.target_density,
                'density_deviation': abs(current_density.item() - self.target_density),
                'density_error_pct': abs(current_density.item() - self.target_density) * 100,
            }
            return total_reg_loss, stats
        else:
            return total_reg_loss
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def get_keep_probabilities(self, logits):
        """
        Get keep probabilities from logits: π_ij
        
        Args:
            logits: Edge logits [batch_size, num_nodes, num_nodes]
        
        Returns:
            keep_probs: Keep probabilities [batch_size, num_nodes, num_nodes]
        """
        return compute_keep_probabilities(logits, self.l0_params, self.l0_method)
    
    def get_statistics(self):
        """
        Get current regularization statistics for logging
        
        Returns:
            stats: Dictionary of current state
        """
        return {
            'current_epoch': self.current_epoch,
            'reg_mode': self.reg_mode,
            'l0_method': self.l0_method,
            # Lambda values
            'base_lambda': self.base_lambda,
            'current_lambda': self.current_lambda,
            'base_lambda_density': self.base_lambda_density,
            'current_lambda_density': self.current_lambda_density,
            # Density control
            'target_density': self.target_density,
            'alpha_min': self.alpha_min,
            'alpha_max': self.alpha_max,
            'enable_adaptive_lambda': self.enable_adaptive_lambda,
            'enable_density_loss': self.enable_density_loss,
            # Schedule
            'warmup_epochs': self.warmup_epochs,
            'ramp_epochs': self.ramp_epochs,
        }
    
    def print_configuration(self):
        """Print current configuration"""
        stats = self.get_statistics()
        print(f"\n{'='*70}")
        print(f"EGLasso Regularization Configuration")
        print(f"{'='*70}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*70}\n")
    
    def __repr__(self):
        return (f"EGLassoRegularization(\n"
                f"  mode={self.reg_mode},\n"
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


# ============================================================================
# STANDALONE SCHEDULE CLASSES (for convenience)
# ============================================================================

class LambdaSchedule:
    """
    Standalone lambda schedule for use in training scripts
    
    Three-phase schedule:
    - Phase 1: Linear warmup to 10% of base
    - Phase 2: Linear ramp to 100% of base
    - Phase 3: Plateau at base value
    """
    
    def __init__(self, base_lambda=0.0001, warmup_epochs=15, ramp_epochs=20):
        """
        Args:
            base_lambda: Target lambda value (λ_base)
            warmup_epochs: Warmup duration (E_warmup)
            ramp_epochs: Ramp duration after warmup (E_ramp)
        """
        self.base_lambda = base_lambda
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
    
    def get_lambda(self, epoch):
        """Get lambda for given epoch"""
        if epoch < self.warmup_epochs:
            # Phase 1: Warmup to 10%
            progress = epoch / self.warmup_epochs
            return progress * 0.1 * self.base_lambda
        
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            # Phase 2: Ramp from 10% to 100%
            post_warmup = epoch - self.warmup_epochs
            progress = post_warmup / self.ramp_epochs
            return self.base_lambda * (0.1 + 0.9 * progress)
        
        else:
            # Phase 3: Plateau at 100%
            return self.base_lambda


class DensityLambdaSchedule:
    """
    Standalone density lambda schedule
    
    Two-phase schedule:
    - Phase 1: Linear warmup to 50% of base
    - Phase 2: Linear ramp to 100% of base
    """
    
    def __init__(self, base_lambda_density=0.03, warmup_epochs=15, ramp_epochs=10):
        """
        Args:
            base_lambda_density: Target density lambda (λ_ρ^base)
            warmup_epochs: Warmup duration (E_warmup)
            ramp_epochs: Ramp duration after warmup
        """
        self.base_lambda_density = base_lambda_density
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
    
    def get_lambda_density(self, epoch):
        """Get density lambda for given epoch"""
        if epoch < self.warmup_epochs:
            # Phase 1: Warmup to 50%
            progress = epoch / self.warmup_epochs
            return progress * 0.5 * self.base_lambda_density
        
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            # Phase 2: Ramp from 50% to 100%
            post_warmup = epoch - self.warmup_epochs
            progress = post_warmup / self.ramp_epochs
            scale = 0.5 + 0.5 * progress
            return scale * self.base_lambda_density
        
        else:
            # Plateau at 100%
            return self.base_lambda_density


class TemperatureSchedule:
    """
    Standalone temperature schedule with cosine annealing
    
    Three-phase schedule:
    - Phase 1: Constant at initial temperature
    - Phase 2: Cosine annealing to minimum
    - Phase 3: Plateau at minimum
    """
    
    def __init__(self, tau_init=5.0, tau_min=1.0, t_warmup=15, t_anneal=35):
        """
        Args:
            tau_init: Initial temperature (τ_init)
            tau_min: Minimum temperature (τ_min)
            t_warmup: Warmup end epoch
            t_anneal: Annealing end epoch
        """
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.t_warmup = t_warmup
        self.t_anneal = t_anneal
        self.mu_min = 0.1
    
    def get_temperature(self, epoch):
        """Get temperature for given epoch"""
        if epoch < self.t_warmup:
            # Phase 1: Constant
            return self.tau_init
        
        elif epoch <= self.t_anneal:
            # Phase 2: Cosine annealing
            progress = (epoch - self.t_warmup) / (self.t_anneal - self.t_warmup)
            progress = min(1.0, max(0.0, progress))
            mu = self.mu_min + (1 - self.mu_min) * (np.cos(progress * np.pi) + 1) / 2
            return max(self.tau_min, self.tau_init * mu)
        
        else:
            # Phase 3: Plateau
            return self.tau_min


