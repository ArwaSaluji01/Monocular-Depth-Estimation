import torch

from src.depth_estimation.config import load_config


def main():
    config = load_config()

    print("=" * 45)
    print("MONOCULAR DEPTH ESTIMATION")
    print("=" * 45)

    print(f"Project: {config['project']['name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU (local development)")
    print("Configuration loaded successfully.")


if __name__ == "__main__":
    main()