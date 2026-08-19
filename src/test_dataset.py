import matplotlib.pyplot as plt

from src.datasets.cifar10_dataset import CIFAR10Dataset

dataset = CIFAR10Dataset(
    degradation="noise"
)

image, label = dataset[0]

image = image.permute(1, 2, 0)

plt.imshow(image)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()