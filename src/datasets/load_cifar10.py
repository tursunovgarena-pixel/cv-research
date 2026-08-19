from torchvision import datasets
from torchvision import transforms

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data/raw",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data/raw",
    train=False,
    download=True,
    transform=transform
)

print("=" * 50)
print("CIFAR-10 loaded successfully")
print("=" * 50)

print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")

print("\nClasses:")

for i, name in enumerate(train_dataset.classes):
    print(f"{i}: {name}")