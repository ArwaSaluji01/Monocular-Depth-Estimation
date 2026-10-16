from src.depth_estimation.nyu_dataset import NYUDepthDataset


def main():
    dataset = NYUDepthDataset()

    try:
        print("Number of samples:", len(dataset))

        for index in range(min(3, len(dataset))):
            sample = dataset[index]
            image = sample["image"]
            depth = sample["depth"]
            valid = sample["valid_mask"]

            print(f"\nSample {index}")
            print("RGB shape:", tuple(image.shape))
            print("Depth shape:", tuple(depth.shape))
            print("Valid pixels:", f"{valid.float().mean().item():.2%}")

            if valid.any():
                values = depth[valid]
                print("Depth minimum:", values.min().item())
                print("Depth maximum:", values.max().item())

    finally:
        dataset.close()


if __name__ == "__main__":
    main()