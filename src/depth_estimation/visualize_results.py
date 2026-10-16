import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.depth_estimation.metrics import evaluate_depth


def normalize_depth(depth):
    """Normalize finite values for visualization."""
    valid = np.isfinite(depth)
    result = np.full(depth.shape, np.nan, dtype=np.float32)

    if valid.any():
        low = depth[valid].min()
        high = depth[valid].max()

        if high > low:
            result[valid] = (depth[valid] - low) / (high - low)
        else:
            result[valid] = 0.0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument(
        "--prediction-type",
        choices=["depth", "inverse_depth"],
        default="depth",
    )
    parser.add_argument("--median-align", action="store_true")
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument(
        "--output",
        default="outputs/visualizations/depth_analysis.png",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    prediction_path = Path(args.prediction)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)

    image = np.asarray(Image.open(image_path).convert("RGB"))
    prediction = np.load(prediction_path).squeeze()

    if prediction.ndim != 2:
        raise ValueError("Prediction must be a 2D depth map.")

    if prediction.shape != image.shape[:2]:
        raise ValueError(
            f"Image size {image.shape[:2]} does not match "
            f"prediction size {prediction.shape}."
        )

    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values.")

    if np.all(prediction <= 0):
        raise ValueError(
            "Prediction has no positive values. Check model output."
        )

    ground_truth = None
    metrics = None

    if args.ground_truth:
        ground_truth = np.load(args.ground_truth).squeeze()

        if ground_truth.shape != prediction.shape:
            raise ValueError(
                f"Ground truth {ground_truth.shape} does not match "
                f"prediction {prediction.shape}. Align them first."
            )

        metrics = evaluate_depth(
            prediction,
            ground_truth,
            max_depth=args.max_depth,
            prediction_type=args.prediction_type,
            median_align=args.median_align,
        )

    columns = 2 if ground_truth is None else 4
    fig, axes = plt.subplots(
        1, columns, figsize=(5 * columns, 5)
    )
    axes = np.atleast_1d(axes)

    axes[0].imshow(image)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    axes[1].imshow(normalize_depth(prediction), cmap="inferno")
    axes[1].set_title("Predicted Depth (normalized)")
    axes[1].axis("off")

    if ground_truth is not None:
        axes[2].imshow(
            normalize_depth(ground_truth), cmap="inferno"
        )
        axes[2].set_title("Ground-Truth Depth (normalized)")
        axes[2].axis("off")

        valid = (
            np.isfinite(ground_truth)
            & (ground_truth > 0)
            & np.isfinite(prediction)
            & (prediction > 0)
        )

        error = np.full(ground_truth.shape, np.nan)
        error[valid] = (
            np.abs(prediction[valid] - ground_truth[valid])
            / ground_truth[valid]
        )

        axes[3].imshow(error, cmap="magma", vmin=0, vmax=1)
        axes[3].set_title("Absolute Relative Error (clipped at 1)")
        axes[3].axis("off")

        fig.suptitle(
            f"Abs Rel: {metrics['abs_rel']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | "
            f"δ1: {metrics['delta1']:.4f}"
        )

        report_path = Path("outputs/reports/depth_metrics.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        report_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved: {output_path}")

    if metrics is not None:
        print("Evaluation metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()