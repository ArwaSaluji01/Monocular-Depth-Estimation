from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.depth_estimation.nyu_dataset import NYUDepthDataset


def main():
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = NYUDepthDataset()

    try:
        print(f"Dataset samples: {len(dataset)}")

        for index in [0, 10, 50]:
            if index >= len(dataset):
                continue

            sample = dataset[index]

            image = sample["image"].permute(1, 2, 0).numpy()
            depth = sample["depth"].squeeze(0).numpy()
            valid = sample["valid_mask"].squeeze(0).numpy()

            print(
                f"Sample {index}: "
                f"RGB={image.shape}, depth={depth.shape}, "
                f"valid pixels={valid.mean():.2%}"
            )

            if valid.any():
                print(
                    f"Valid depth range: "
                    f"{depth[valid].min():.3f} to "
                    f"{depth[valid].max():.3f}"
                )

            display_depth = np.where(valid, depth, np.nan)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            axes[0].imshow(image)
            axes[0].set_title("RGB Image")
            axes[0].axis("off")

            axes[1].imshow(display_depth, cmap="plasma")
            axes[1].set_title("Ground-Truth Depth")
            axes[1].axis("off")

            fig.tight_layout()
            fig.savefig(
                output_dir / f"nyu_sample_{index:04d}.png",
                dpi=150,
            )
            plt.close(fig)

        print(f"Visualizations saved to {output_dir}")

    finally:
        dataset.close()


if __name__ == "__main__":
    main()