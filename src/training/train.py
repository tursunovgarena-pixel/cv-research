import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.cifar10_dataset import CIFAR10Dataset
from src.models.resnet import create_resnet18
from src.training.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CIFAR10Dataset(
    train=True,
    degradation=None
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

model = create_resnet18().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

trainer = Trainer(
    model,
    loader,
    criterion,
    optimizer,
    device
)

epochs = 5

for epoch in range(epochs):

    loss, accuracy = trainer.train_epoch()

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

from pathlib import Path

Path("models/checkpoints").mkdir(parents=True, exist_ok=True)

torch.save(
    model.state_dict(),
    "models/checkpoints/resnet18_baseline.pth"
)

print("\nModel saved successfully!")