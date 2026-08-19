import random

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root="./data/raw",
    train=True,
    download=False,
    transform=transform
)

fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for ax in axes.flatten():
    idx = random.randint(0, len(dataset) - 1)

    image, label = dataset[idx]

    image = image.permute(1, 2, 0)

    ax.imshow(image)
    ax.set_title(dataset.classes[label])
    ax.axis("off")

plt.tight_layout()
plt.show()