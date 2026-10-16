from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class NYUDepthDataset(Dataset):
    def __init__(
        self,
        mat_path="data/raw/nyu_depth_v2_labeled.mat",
        image_size=(256, 320),
    ):
        self.mat_path = Path(mat_path)
        self.image_size = image_size

        if not self.mat_path.exists():
            raise FileNotFoundError(self.mat_path)

        self.file = h5py.File(self.mat_path, "r")

        if "images" not in self.file or "depths" not in self.file:
            self.file.close()
            raise KeyError("Expected 'images' and 'depths' in MAT file.")

        self.images = self.file["images"]
        self.depths = self.file["depths"]

        if len(self.images) != len(self.depths):
            self.file.close()
            raise ValueError("RGB and depth sample counts do not match.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Convert MATLAB/HDF5 storage layout to HWC and HW.
        image = np.asarray(self.images[index]).transpose(2, 1, 0)
        depth = np.asarray(self.depths[index]).T.astype(np.float32)

        image = image.astype(np.uint8)

        height, width = self.image_size

        image = cv2.resize(
            image, (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        depth = cv2.resize(
            depth, (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        # Invalid depth values are represented as zero.
        depth[~np.isfinite(depth)] = 0
        depth[depth <= 0] = 0

        image_tensor = torch.from_numpy(
            image.copy()
        ).permute(2, 0, 1).float() / 255.0

        depth_tensor = torch.from_numpy(
            depth.copy()
        ).unsqueeze(0)

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": depth_tensor > 0,
        }

    def close(self):
        if self.file:
            self.file.close()