# Experiment Log

## Experiment: Multi-Seed Robustness Evaluation

### Objective

Evaluate the robustness of a ResNet18 image classification model to common image degradations and measure the stability of the results across different random seeds.

### Dataset

CIFAR-10

- Training images: 50,000
- Test images: 10,000
- Classes: 10

### Model

ResNet18

### Training configuration

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 10
- Train/validation split: 90% / 10%
- Validation split seed: 42
- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU
- PyTorch: 2.13.0+cu126

### Random Seeds

Three independent training runs were performed:

- Seed 42
- Seed 123
- Seed 2026

### Best Validation Accuracy

| Seed | Best Validation Accuracy |
|---:|---:|
| 42 | 74.28% |
| 123 | 74.94% |
| 2026 | 73.94% |

Mean validation accuracy:

**74.39% ± 0.50 percentage points**

### Robustness Evaluation

Each trained model was evaluated on:

- Original images
- Brightness degradation
- Contrast degradation
- Blur
- Noise
- JPEG compression

### Robustness Results

| Condition | Mean Accuracy | Std |
|---|---:|---:|
| Original | 74.18% | 0.98 |
| Brightness | 70.20% | 0.68 |
| Contrast | 69.44% | 1.44 |
| Blur | 44.76% | 2.46 |
| Noise | 67.98% | 0.98 |
| JPEG | 69.53% | 1.11 |

### Accuracy Drop

Compared with the original images:

| Condition | Accuracy Drop |
|---|---:|
| Brightness | 3.98 pp |
| Contrast | 4.74 pp |
| Blur | 29.42 pp |
| Noise | 6.20 pp |
| JPEG | 4.65 pp |

### Main Finding

Blur was the most damaging image degradation.

The average classification accuracy decreased from 74.18% on original images to 44.76% under blur, representing a decrease of 29.42 percentage points.

Brightness, contrast, noise, and JPEG compression caused substantially smaller performance decreases.

The results across three random seeds were relatively consistent, suggesting that the observed robustness pattern is not caused by a single training run.

### Generated Figures

- `figures/robustness_accuracy.png`
- `figures/robustness_drop.png`

### Results File

- `results/robustness_results.csv`