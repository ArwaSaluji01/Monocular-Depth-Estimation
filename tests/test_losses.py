import pytest
import torch

from src.depth_estimation.losses import scale_invariant_depth_loss


def test_identical_depth_has_zero_loss():
    depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    loss = scale_invariant_depth_loss(depth, depth.clone())

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_scale_invariant():
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    prediction = target * 5.0

    loss = scale_invariant_depth_loss(prediction, target)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_invalid_ground_truth_is_ignored():
    target = torch.tensor([[[[1.0, 0.0], [3.0, 4.0]]]])
    prediction = target.clone()

    loss = scale_invariant_depth_loss(prediction, target)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)