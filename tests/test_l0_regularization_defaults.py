"""
Regression test guarding the backward-compatibility contract of
model/L0_Reg.py's extended parameters (density_loss_warmup_epochs,
scale_density_loss_by_edges, temperature_min). These were added for the
survival-prediction pipeline with defaults chosen so every EXISTING
classification caller (model/LENS.py, model/LENS2.py, model/LENSwithGAT.py --
none of which pass these new kwargs) gets byte-for-byte identical behavior to
before the extension. This test pins that contract so a future default
change can't silently alter classification training.

Requires the project's own environment -- run with:
    pytest tests/test_l0_regularization_defaults.py
"""

from model.L0_Reg import L0Regularization


def test_defaults_match_pre_extension_classification_behavior():
    reg = L0Regularization(lambda_reg=0.001, warmup_epochs=15, ramp_epochs=20)

    # density_loss_warmup_epochs must default to warmup_epochs (the original
    # single-schedule behavior), not some independent value.
    assert reg.density_loss_warmup_epochs == reg.warmup_epochs == 15

    # scale_density_loss_by_edges must default off -- classification callers
    # never asked for edge-count-scaled density loss.
    assert reg.scale_density_loss_by_edges is False

    # temperature_min must default to 1.0, the original hardcoded plateau
    # (not the survival pipeline's 0.67).
    assert reg.temperature_min == 1.0


def test_survival_style_kwargs_are_opt_in_only():
    # Passing the survival-pipeline's values must not be the default -- they
    # only take effect when explicitly requested.
    reg = L0Regularization(
        lambda_reg=0.001, warmup_epochs=15, ramp_epochs=20,
        density_loss_warmup_epochs=0, scale_density_loss_by_edges=True,
        temperature_min=0.67,
    )
    assert reg.density_loss_warmup_epochs == 0
    assert reg.scale_density_loss_by_edges is True
    assert reg.temperature_min == 0.67
