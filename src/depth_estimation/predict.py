import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.depth_estimation.model import MonocularDepthEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    image = Image.open(image_path).convert("RGB")

    estimator = MonocularDepthEstimator()
    depth = estimator.predict(image)

    if not np.isfinite(depth).all():
        raise ValueError("Prediction contains non-finite values.")

    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    np.save(output_dir / f"{stem}_depth.npy", depth)

    low, high = depth.min(), depth.max()
    normalized = (depth - low) / max(float(high - low), 1e-8)

    grayscale = (normalized * 255).astype(np.uint8)
    Image.fromarray(grayscale).save(
        output_dir / f"{stem}_depth_gray.png"
    )

    plt.imsave(
        output_dir / f"{stem}_depth_color.png",
        normalized,
        cmap="inferno",
    )

    print(f"Input image: {image_path}")
    print(f"Image size: {image.width} x {image.height}")
    print(f"Prediction shape: {depth.shape}")
    print(f"Raw prediction range: {low:.4f} to {high:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()