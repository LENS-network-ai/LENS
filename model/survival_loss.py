"""
Discrete-time negative log-likelihood survival loss (Zadeh & Schmid, 2020),
in the MCAT/PORPOISE parameterization used across WSI survival literature.

Ported from the validated implementation in the healnet project
(healnet/healnet/models/survival_loss.py) so LENS's survival head trains
against the same, already-published loss formulation.
"""

import torch


def nll_survival_loss(hazards, y_disc, censorship, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    Args:
        hazards     : [B, n_bins]  sigmoid(logits) -> hazard per discrete time bin
        y_disc      : [B] or [B, 1]  ground-truth time-bin index (0..n_bins-1)
        censorship  : [B] or [B, 1]  1 if censored (event not observed), 0 if event occurred
        alpha       : interpolates between the full NLL and its uncensored-only term
        eps         : numerical floor to avoid log(0)
        reduction   : 'mean' or 'sum'

    Returns:
        scalar loss tensor
    """
    batch_size = hazards.shape[0]
    y = y_disc.view(batch_size, 1).long()
    c = censorship.view(batch_size, 1).float()

    # Survival function S(t) = prod_{i<=t} (1 - hazard_i)
    survival = torch.cumprod(1 - hazards, dim=1)
    # S(-1) = 1 by definition (everyone alive before t=0): pad a leading column of ones
    survival_padded = torch.cat([torch.ones_like(c), survival], dim=1)

    s_prev = torch.gather(survival_padded, 1, y).clamp(min=eps)        # S(y-1)
    hazard_y = torch.gather(hazards, 1, y).clamp(min=eps)              # h(y)
    s_curr = torch.gather(survival_padded, 1, y + 1).clamp(min=eps)    # S(y)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(hazard_y))
    censored_loss = -c * torch.log(s_curr)

    neg_log_likelihood = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_log_likelihood + alpha * uncensored_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"Unknown reduction: {reduction}")
