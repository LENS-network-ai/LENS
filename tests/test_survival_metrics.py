"""
Regression tests for the discrete-time survival loss and concordance index
used by the survival-prediction pipeline (training/train_survival.py).

Requires the project's own environment (torch, numpy) -- run with:
    pytest tests/test_survival_metrics.py
"""

import math

import pytest
import torch

from model.survival_loss import nll_survival_loss
from utils.metrics import concordance_index


def test_nll_survival_loss_hand_computed():
    # 2 samples, 2 time bins, hazards=0.5 in every bin for both.
    # survival = cumprod(1-hazards) = [0.5, 0.25]; survival_padded = [1, 0.5, 0.25]
    # Sample A (uncensored, y=0): loss = -(log(S(-1)=1) + log(hazard(0)=0.5)) = -log(0.5) = ln(2)
    # Sample B (censored, y=0):   loss = -log(S(0)=0.5) = ln(2)
    # Both terms hand-computed to ln(2) regardless of censorship for this
    # symmetric input -- a coincidence of the chosen hazards, not a
    # simplification of the formula (verified by working through
    # model/survival_loss.py's definition directly).
    hazards = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    y_disc = torch.tensor([0, 0])
    censorship = torch.tensor([0, 1])  # sample A uncensored, sample B censored

    loss = nll_survival_loss(hazards, y_disc, censorship, alpha=0.0, reduction='mean')
    assert loss.item() == pytest.approx(math.log(2), abs=1e-5)

    loss_sum = nll_survival_loss(hazards, y_disc, censorship, alpha=0.0, reduction='sum')
    assert loss_sum.item() == pytest.approx(2 * math.log(2), abs=1e-5)


def test_nll_survival_loss_rejects_unknown_reduction():
    hazards = torch.tensor([[0.5, 0.5]])
    y_disc = torch.tensor([0])
    censorship = torch.tensor([0])
    with pytest.raises(ValueError):
        nll_survival_loss(hazards, y_disc, censorship, reduction='bogus')


def test_concordance_index_perfect_ranking():
    # Higher risk should correspond to shorter (uncensored) survival time.
    risk_scores = [3.0, 1.0, 0.0]
    event_times = [5.0, 10.0, 15.0]
    censorships = [0, 0, 1]  # patient C's 15-month time is censored
    c_index = concordance_index(risk_scores, event_times, censorships)
    assert c_index == pytest.approx(1.0)


def test_concordance_index_no_comparable_pairs_returns_half():
    # Every patient censored -> no pair has an observed-event "shorter time"
    # side, so there's nothing to compare.
    risk_scores = [1.0, 2.0, 3.0]
    event_times = [5.0, 10.0, 15.0]
    censorships = [1, 1, 1]
    assert concordance_index(risk_scores, event_times, censorships) == 0.5


def test_concordance_index_single_sample_returns_half():
    assert concordance_index([1.0], [5.0], [0]) == 0.5
