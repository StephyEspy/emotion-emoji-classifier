# src/preprocessing.py

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from .dataset import FER2013ImageDataset


# ----- Class / label mapping -----

# FER2013 label convention:
# 0: angry, 1: disgust, 2: fear, 3: happy, 4: sad, 5: surprise, 6: neutral
CLASS_NAMES = [
    "angry",    # 0
    "disgust",  # 1
    "fear",     # 2
    "happy",    # 3
    "sad",      # 4
    "surprise", # 5
    "neutral",  # 6
]

CLASS_TO_LABEL: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ----- Label → emoji mapping -----

LABEL_TO_EMOJI: Dict[int, str] = {
    0: "😠",  # angry
    1: "🤢",  # disgust
    2: "😨",  # fear
    3: "😊",  # happy
    4: "😢",  # sad
    5: "😲",  # surprise
    6: "😐",  # neutral
}

EMOJI_TO_LABEL: Dict[str, int] = {emoji: label for label, emoji in LABEL_TO_EMOJI.items()}


def save_emoji_map(json_path: str = "data/processed/emoji_map.json") -> None:
    """
    Save the label→emoji mapping to a JSON file so it can be reused
    in evaluation and for the final demo.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_TO_EMOJI, f, ensure_ascii=False, indent=2)


# ----- Helper functions to index image folders -----

def _is_image_file(filename: str) -> bool:
    ext = filename.lower().rsplit(".", 1)[-1]
    return ext in {"jpg", "jpeg", "png", "bmp", "png"}


def collect_items(split_dir: str) -> List[Tuple[str, int]]:
    """
    Walks a split directory like:
        split_dir/
          angry/
          disgust/
          ...
    and returns a list of (image_path, label).
    """
    items: List[Tuple[str, int]] = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")

        for fname in os.listdir(class_dir):
            if not _is_image_file(fname):
                continue
            img_path = os.path.join(class_dir, fname)
            items.append((img_path, label))

    return items


def split_train_val(
    train_items: List[Tuple[str, int]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Deterministically split training items into train and validation lists.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(train_items))
    rng.shuffle(indices)

    n_val = int(len(indices) * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_split = [train_items[i] for i in train_idx]
    val_split = [train_items[i] for i in val_idx]

    return train_split, val_split


# ----- Transforms (augmentations + normalization) -----

def get_train_transform() -> transforms.Compose:
    """
    Data augmentation + normalization for the training set.
    """
    return transforms.Compose([
        transforms.ToPILImage(),                # (H, W) numpy → PIL
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),                 # → (1, H, W) float in [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_eval_transform() -> transforms.Compose:
    """
    Normalization only for validation and test sets.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ----- Master factory: create train/val/test datasets -----

def create_datasets(
    root_dir: str = "data/raw/fer2013/archive",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Creates train/val/test Dataset objects from the folder structure.

    Expected layout:
        root_dir/
          train/
            angry/
            disgust/
            ...
          test/
            angry/
            disgust/
            ...

    Args:
        root_dir: path to the 'archive' directory containing train/ and test/.
        val_ratio: fraction of training data to use for validation.
        seed: random seed for reproducible splitting.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")

    # 1. Collect all items from train and test folders
    all_train_items = collect_items(train_dir)
    test_items = collect_items(test_dir)

    # 2. Split training into train + val
    train_items, val_items = split_train_val(
        all_train_items, val_ratio=val_ratio, seed=seed
    )

    # 3. Build Dataset objects with appropriate transforms
    train_ds = FER2013ImageDataset(train_items, transform=get_train_transform())
    val_ds = FER2013ImageDataset(val_items, transform=get_eval_transform())
    test_ds = FER2013ImageDataset(test_items, transform=get_eval_transform())

    return train_ds, val_ds, test_ds
