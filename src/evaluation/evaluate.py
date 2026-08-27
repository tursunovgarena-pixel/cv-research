import torch
from torch.utils.data import DataLoader

from src.datasets.cifar10_dataset import CIFAR10Dataset
from src.models.resnet import create_resnet18


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_resnet18().to(device)

model.load_state_dict(
    torch.load(
        "models/checkpoints/resnet18_best.pth",
        map_location=device
    )
)

model.eval()

dataset = CIFAR10Dataset(
    train=False,
    degradation=None
)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False
)

correct = 0
total = 0

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

accuracy = 100 * correct / total

print("=" * 50)
print("Evaluation Results")
print("=" * 50)
print(f"Test Accuracy: {accuracy:.2f}%")