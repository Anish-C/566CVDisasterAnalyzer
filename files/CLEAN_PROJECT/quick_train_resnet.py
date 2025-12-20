"""
Quick Fine-Tuning for ResNet-50 (Time-Efficient)
- Freezes backbone; trains only final FC layer
- Uses a subset of the dataset for speed
- Runs a small number of epochs
- Saves checkpoint to finetuned_resnet50_quick.pt
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def build_model(device: str = "cpu"):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Replace final layer for 4 classes
    model.fc = nn.Linear(model.fc.in_features, 4)
    # Freeze all layers except final FC for fast training
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False
    return model.to(device)


def make_dataloaders(data_dir: str, batch_size: int = 32, max_per_class: int = 500):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Subset by class to cap total size for speed
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(idx)

    selected_indices = []
    for label, indices in class_to_indices.items():
        selected_indices.extend(indices[:max_per_class])

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, dataset.classes


def train_quick(data_dir: str, out_path: str, epochs: int = 2, batch_size: int = 32, lr: float = 1e-3,
                 max_per_class: int = 500, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader, classes = make_dataloaders(data_dir, batch_size=batch_size, max_per_class=max_per_class)
    print(f"Classes: {classes}")
    print(f"Training subset size: {len(loader.dataset)}")

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.3f}")

    # Save checkpoint
    ckpt = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_per_class": max_per_class,
        }
    }
    torch.save(ckpt, out_path)
    print(f"[OK] Saved checkpoint: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick fine-tune ResNet-50 on a subset for speed")
    parser.add_argument("--data", required=True, help="Path to ImageFolder dataset root (class subfolders)")
    parser.add_argument("--out", default="finetuned_resnet50_quick.pt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--max-per-class", type=int, default=500, help="Cap images per class (default: 500)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    train_quick(
        data_dir=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_per_class=args.max_per_class,
    )
