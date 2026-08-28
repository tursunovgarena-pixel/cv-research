import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.datasets.cifar10_dataset import CIFAR10Dataset
from src.models.resnet import create_resnet18
from src.training.trainer import Trainer


# Read seed from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed = args.seed


# Reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 50)
print("Training ResNet18")
print("=" * 50)
print("Device:", device)
print("Seed:", seed)


# Dataset
dataset = CIFAR10Dataset(
    train=True,
    degradation=None
)


# 90% train / 10% validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
)


# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)


# Model
model = create_resnet18().to(device)


# Loss
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Trainer
trainer = Trainer(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device
)


# Training
epochs = 10

best_val_accuracy = 0.0

Path("models/checkpoints").mkdir(
    parents=True,
    exist_ok=True
)


for epoch in range(epochs):

    train_loss, train_accuracy = trainer.train_epoch()

    val_loss, val_accuracy = trainer.validate()

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            f"models/checkpoints/resnet18_seed{seed}_best.pth"
        )

        print(
            f"  ✓ Best model saved "
            f"({val_accuracy:.2f}%)"
        )


print("\n" + "=" * 50)
print("Training complete")
print("=" * 50)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)