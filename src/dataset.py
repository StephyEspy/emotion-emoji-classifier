# src/dataset.py

from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class FER2013ImageDataset(Dataset):
    """
    PyTorch Dataset for FER2013-style image folders.

    Each item is:
        (image_tensor, label)

    Where:
        - image_tensor: torch.Tensor of shape (1, H, W), normalized to ~[-1, 1]
        - label: int in [0..6] following FER2013 convention
    """

    def __init__(
        self,
        items: List[Tuple[str, int]],
        transform: Optional[object] = None,
    ) -> None:
        """
        Args:
            items: list of (image_path, label) pairs.
            transform: optional transform pipeline applied to the image.
        """
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]

        # Open image as grayscale ('L')
        img = Image.open(img_path).convert("L")
        img_np = np.array(img)  # (H, W), uint8

        if self.transform is not None:
            # Most transforms expect numpy array or PIL; we pass numpy here
            img_tensor = self.transform(img_np)
        else:
            # Fallback: basic tensor conversion and scaling to [0, 1]
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).float() / 255.0

        # Safety: if transform doesn't return a tensor for some reason
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).float() / 255.0

        return img_tensor, int(label)
