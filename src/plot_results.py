import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = "results/robustness_results.csv"
OUTPUT_DIR = Path("figures")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Load results
data = {}

with open(INPUT_FILE, newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        model = row["Model"]
        degradation = row["Degradation"]
        accuracy = float(row["Accuracy"])

        if degradation not in data:
            data[degradation] = []

        data[degradation].append(accuracy)


# Order
degradations = [
    "original",
    "brightness",
    "contrast",
    "blur",
    "noise",
    "jpeg",
]


means = [
    np.mean(data[name])
    for name in degradations
]

stds = [
    np.std(data[name], ddof=1)
    for name in degradations
]


# --------------------------------------------------
# Figure 1: Accuracy
# --------------------------------------------------

plt.figure(figsize=(10, 6))

x = np.arange(len(degradations))

plt.bar(
    x,
    means,
    yerr=stds,
    capsize=5
)

plt.xticks(
    x,
    degradations,
    rotation=20
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Image condition")
plt.title(
    "ResNet18 Robustness to Image Degradation"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "robustness_accuracy.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Figure 2: Accuracy drop
# --------------------------------------------------

baseline = means[0]

drops = [
    baseline - mean
    for mean in means
]


plt.figure(figsize=(10, 6))

plt.bar(
    x,
    drops
)

plt.xticks(
    x,
    degradations,
    rotation=20
)

plt.ylabel("Accuracy drop (percentage points)")
plt.xlabel("Image condition")
plt.title(
    "Accuracy Drop Relative to Original Images"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "robustness_drop.png",
    dpi=300
)

plt.close()


print("Figures generated successfully!")

print(
    f"Mean accuracy: {baseline:.2f}%"
)

print(
    f"Largest degradation: "
    f"{degradations[np.argmax(drops)]}"
)