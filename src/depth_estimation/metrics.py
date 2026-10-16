import numpy as np


def evaluate_depth(
    prediction,
    ground_truth,
    max_depth=None,
    prediction_type="depth",
    median_align=False,
    eps=1e-6,
):
    """
    Evaluate predicted depth against ground-truth depth.

    prediction_type:
        "depth"        - larger values mean farther away
        "inverse_depth" - larger values mean closer
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)

    # Remove only an optional leading/trailing channel dimension.
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    elif pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]

    if gt.ndim == 3 and gt.shape[0] == 1:
        gt = gt[0]
    elif gt.ndim == 3 and gt.shape[-1] == 1:
        gt = gt[..., 0]

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred.shape}, GT {gt.shape}"
        )

    if pred.ndim != 2:
        raise ValueError("Expected 2D depth maps.")

    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)

    if prediction_type == "inverse_depth":
        valid &= pred > 0
        pred = 1.0 / np.maximum(pred, eps)
    elif prediction_type != "depth":
        raise ValueError(
            "prediction_type must be 'depth' or 'inverse_depth'."
        )

    if max_depth is not None:
        valid &= gt <= max_depth

    pred_valid = pred[valid]
    gt_valid = gt[valid]

    if pred_valid.size == 0:
        raise ValueError("No valid pixels available for evaluation.")

    if not np.isfinite(pred_valid).all() or (pred_valid <= 0).any():
        raise ValueError(
            "Predicted depth must be finite and positive at valid pixels."
        )

    if median_align:
        pred_median = np.median(pred_valid)
        gt_median = np.median(gt_valid)

        if pred_median <= eps:
            raise ValueError("Cannot median-align zero-valued predictions.")

        pred_valid = pred_valid * (gt_median / pred_median)

    diff = pred_valid - gt_valid
    ratio = np.maximum(
        pred_valid / gt_valid,
        gt_valid / pred_valid,
    )

    return {
        "abs_rel": float(np.mean(np.abs(diff) / gt_valid)),
        "sq_rel": float(np.mean(diff**2 / gt_valid)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(
            np.sqrt(
                np.mean(
                    (np.log(pred_valid) - np.log(gt_valid)) ** 2
                )
            )
        ),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "valid_pixels": int(pred_valid.size),
    }