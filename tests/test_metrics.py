import numpy as np
import pytest

from src.depth_estimation.metrics import evaluate_depth


def test_perfect_prediction():
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = evaluate_depth(gt, gt.copy())

    assert result["abs_rel"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["rmse_log"] == pytest.approx(0.0)
    assert result["delta1"] == pytest.approx(1.0)
    assert result["valid_pixels"] == 4


def test_inverse_depth_conversion():
    gt = np.array([[1.0, 2.0, 4.0]])
    inverse_depth = 1.0 / gt

    result = evaluate_depth(
        inverse_depth,
        gt,
        prediction_type="inverse_depth",
    )

    assert result["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_median_alignment():
    gt = np.array([[2.0, 4.0, 6.0]])
    pred = gt * 3.0

    result = evaluate_depth(pred, gt, median_align=True)

    assert result["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_invalid_ground_truth_pixels_are_ignored():
    gt = np.array([[1.0, 0.0], [np.nan, 4.0]])
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = evaluate_depth(pred, gt)

    assert result["valid_pixels"] == 2
    assert result["rmse"] == pytest.approx(0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate_depth(
            np.ones((2, 2)),
            np.ones((3, 3)),
        )


def test_empty_valid_mask_raises():
    with pytest.raises(ValueError, match="No valid pixels"):
        evaluate_depth(
            np.ones((2, 2)),
            np.zeros((2, 2)),
        )


def test_unknown_prediction_type_raises():
    with pytest.raises(ValueError, match="prediction_type"):
        evaluate_depth(
            np.ones((2, 2)),
            np.ones((2, 2)),
            prediction_type="unknown",
        )