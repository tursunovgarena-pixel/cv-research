import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

from src.preprocessing.image_degradation import ImageDegradation

dataset = CIFAR10(
    root="./data/raw",
    train=True,
    download=False,
)

image, label = dataset[0]

methods = [
    ("Original", None),
    ("Brightness", "brightness"),
    ("Contrast", "contrast"),
    ("Blur", "blur"),
    ("Noise", "noise"),
    ("JPEG", "jpeg"),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (title, method) in zip(axes.flatten(), methods):

    if method is None:
        img = image
    else:
        img = ImageDegradation.apply(image, method)

    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()