from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

from src.preprocessing.image_degradation import ImageDegradation

# Создаем папку для результатов
output_dir = Path("figures/day6")
output_dir.mkdir(parents=True, exist_ok=True)

# Загружаем датасет
dataset = CIFAR10(
    root="./data/raw",
    train=True,
    download=False,
)

# Берем первое изображение
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

save_path = output_dir / "experiment_1.png"
plt.savefig(save_path, dpi=300)

print(f"Saved to: {save_path}")

plt.show()