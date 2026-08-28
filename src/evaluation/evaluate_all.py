import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.cifar10_dataset import CIFAR10Dataset
from src.models.resnet import create_resnet18


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


checkpoints = [
    "resnet18_seed42_best.pth",
    "resnet18_seed123_best.pth",
    "resnet18_seed2026_best.pth",
]


degradations = [
    None,
    "brightness",
    "contrast",
    "blur",
    "noise",
    "jpeg",
]


results = []


for checkpoint in checkpoints:

    print("\n" + "=" * 50)
    print(f"Evaluating: {checkpoint}")
    print("=" * 50)

    model = create_resnet18().to(device)

    model.load_state_dict(
        torch.load(
            f"models/checkpoints/{checkpoint}",
            map_location=device,
        )
    )

    model.eval()

    model_results = {}

    for degradation in degradations:

        dataset = CIFAR10Dataset(
            train=False,
            degradation=degradation,
        )

        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
        )

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predictions = outputs.argmax(dim=1)

                correct += (
                    (predictions == labels)
                    .sum()
                    .item()
                )

                total += labels.size(0)

        accuracy = 100 * correct / total

        name = (
            "original"
            if degradation is None
            else degradation
        )

        model_results[name] = accuracy

        results.append(
            [checkpoint, name, accuracy]
        )

        print(
            f"{name:12} {accuracy:.2f}%"
        )


# Save results
Path("results").mkdir(
    parents=True,
    exist_ok=True
)

output_file = "results/robustness_results.csv"

with open(
    output_file,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "Model",
            "Degradation",
            "Accuracy",
        ]
    )

    writer.writerows(results)


print("\n" + "=" * 50)
print("All experiments completed")
print("=" * 50)

print(
    f"Results saved to {output_file}"
)