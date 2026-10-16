import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)

from src.depth_estimation.losses import scale_invariant_depth_loss
from src.depth_estimation.nyu_dataset import NYUDepthDataset


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_samples(samples):
    return {
        "images": [to_pil_image(s["image"]) for s in samples],
        "depths": torch.stack([s["depth"] for s in samples]),
        "masks": torch.stack([s["valid_mask"] for s in samples]),
    }


def run_epoch(model, processor, loader, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    batches = 0

    for batch in loader:
        inputs = processor(
            images=batch["images"],
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(device)

        target = batch["depths"].to(device)

        with torch.set_grad_enabled(training):
            outputs = model(pixel_values=pixel_values)
            prediction = outputs.predicted_depth.unsqueeze(1)

            target = torch.nn.functional.interpolate(
                target,
                size=prediction.shape[-2:],
                mode="nearest",
            )

            loss = scale_invariant_depth_loss(prediction, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

        total_loss += loss.item()
        batches += 1

    if batches == 0:
        raise RuntimeError("No batches were processed.")

    return total_loss / batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    if args.epochs < 1 or args.max_samples < 4 or args.batch_size < 1:
        raise ValueError("Use epochs >= 1, samples >= 4, batch size >= 1.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = NYUDepthDataset()

    try:
        count = min(args.max_samples, len(dataset))
        if count < 4:
            raise ValueError("Dataset must contain at least four samples.")

        subset, _ = random_split(
            dataset,
            [count, len(dataset) - count],
            generator=torch.Generator().manual_seed(args.seed),
        )

        train_count = max(1, int(count * 0.8))
        val_count = count - train_count

        train_set, val_set = random_split(
            subset,
            [train_count, val_count],
            generator=torch.Generator().manual_seed(args.seed),
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_samples,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_samples,
        )

        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
        )

        best_val_loss = float("inf")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(args.epochs):
            train_loss = run_epoch(
                model, processor, train_loader, optimizer, device, True
            )
            with torch.no_grad():
                val_loss = run_epoch(
                    model, processor, val_loader, optimizer, device, False
                )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train loss: {train_loss:.6f} | "
                f"Validation loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "validation_loss": val_loss,
                        "model_id": MODEL_ID,
                        "seed": args.seed,
                    },
                    output_dir / "depth_finetuned_best.pt",
                )

        print(f"Best checkpoint saved in {output_dir}")

    finally:
        dataset.close()


if __name__ == "__main__":
    main()