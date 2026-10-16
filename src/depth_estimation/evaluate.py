import argparse
import json
from pathlib import Path

import numpy as np

from src.depth_estimation.metrics import evaluate_depth


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate monocular depth predictions."
    )
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument(
        "--prediction-type",
        choices=["depth", "inverse_depth"],
        default="inverse_depth",
    )
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--median-align", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    prediction = np.load(args.prediction)
    ground_truth = np.load(args.ground_truth)

    metrics = evaluate_depth(
        prediction=prediction,
        ground_truth=ground_truth,
        max_depth=args.max_depth,
        prediction_type=args.prediction_type,
        median_align=args.median_align,
    )

    print(json.dumps(metrics, indent=2))

    report_path = (
        Path(args.report)
        if args.report
        else Path("outputs/reports/depth_metrics.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()