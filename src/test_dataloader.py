from torch.utils.data import DataLoader

from src.datasets.cifar10_dataset import CIFAR10Dataset

dataset = CIFAR10Dataset(
    train=True,
    degradation="noise"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

images, labels = next(iter(loader))

print("=" * 50)
print("Batch loaded successfully")
print("=" * 50)

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"First label: {labels[0].item()}")