import random

import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

from src.preprocessing.image_degradation import ImageDegradation

dataset = CIFAR10(
    root="./data/raw",
    train=True,
    download=False,
)

image, label = dataset[random.randint(0, len(dataset) - 1)]

bright = ImageDegradation.change_brightness(image)
contrast = ImageDegradation.change_contrast(image)

fig, ax = plt.subplots(1, 3, figsize=(10, 4))

ax[0].imshow(image)
ax[0].set_title("Original")

ax[1].imshow(bright)
ax[1].set_title("Brightness")

ax[2].imshow(contrast)
ax[2].set_title("Contrast")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()