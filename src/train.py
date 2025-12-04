import os
import json
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from src.preprocessing import create_datasets
from src.model import get_resnet18


def get_dataloaders(
    root_dir: str = "data/raw/fer2013/archive",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for train, val, and test splits.
    """
    train_ds, val_ds, test_ds = create_datasets(root_dir=root_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """
    Train for a single epoch.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Train", leave=False)

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)
        loop.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate on validation or test set.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Eval", leave=False)

    with torch.no_grad():
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            avg_loss = running_loss / max(total, 1)
            acc = correct / max(total, 1)
            loop.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def save_checkpoint(model: nn.Module, path: str) -> None:
    """
    Save model weights to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_history(history: Dict[str, Any], path: str) -> None:
    """
    Save training history (loss/accuracy curves) as JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def main():
    # -----------------------
    # Hyperparameters
    # -----------------------
    root_dir = "data/raw/fer2013/archive"
    batch_size = 64
    num_workers = 2
    num_epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-4
    use_amp = True          # mixed precision on GPU
    patience = 7            # early stopping patience (epochs)

    # -----------------------
    # Device
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # Data
    # -----------------------
    print("Loading datasets and creating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # -----------------------
    # Model, loss, optimizer, scheduler
    # -----------------------
    print("Initializing model...")
    model = get_resnet18(num_classes=7, pretrained=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = "models/best_resnet18.pth"
    history_path = "models/history.json"

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{num_epochs} - lr={current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        # Log to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Step LR scheduler
        scheduler.step()

        # Check for improvement (early stopping on val accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, best_model_path)
            print(f"  ✓ New best model saved with val acc = {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break

    # Save history after training
    save_history(history, history_path)
    print(f"\nTraining history saved to: {history_path}")

    # -----------------------
    # Final test evaluation
    # -----------------------
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
