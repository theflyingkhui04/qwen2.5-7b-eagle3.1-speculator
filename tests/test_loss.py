import pytest
from training.loss import EagleDualLoss

def test_dual_loss_weights_and_defaults():
    loss_fn = EagleDualLoss(smooth_l1_weight=0.5, ce_weight=1.0)
    assert loss_fn.smooth_l1_weight == 0.5
    assert loss_fn.ce_weight == 1.0

def test_dual_loss_custom_weights():
    loss_fn = EagleDualLoss(smooth_l1_weight=0.2, ce_weight=2.0)
    assert loss_fn.smooth_l1_weight == 0.2
    assert loss_fn.ce_weight == 2.0
