import torch
import torch.nn.functional as F


def scale_invariant_depth_loss(prediction, target, eps=1e-6):
    """
    Compare relative depth after normalizing each image by its
    median valid depth. Invalid ground-truth pixels are ignored.
    """
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")

    losses = []

    for pred, gt in zip(prediction, target):
        valid = (
            torch.isfinite(pred)
            & torch.isfinite(gt)
            & (pred > eps)
            & (gt > eps)
        )

        if not valid.any():
            continue

        pred_valid = pred[valid]
        gt_valid = gt[valid]

        pred_median = pred_valid.median().clamp_min(eps)
        gt_median = gt_valid.median().clamp_min(eps)

        pred_normalized = pred_valid / pred_median
        gt_normalized = gt_valid / gt_median

        log_difference = (
            torch.log(pred_normalized.clamp_min(eps))
            - torch.log(gt_normalized.clamp_min(eps))
        )

        losses.append(F.smooth_l1_loss(
            log_difference,
            torch.zeros_like(log_difference),
        ))

    if not losses:
        raise ValueError("Batch contains no valid depth pixels.")

    return torch.stack(losses).mean()