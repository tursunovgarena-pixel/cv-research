import torch

from src.models.resnet import create_resnet18

model = create_resnet18()

x = torch.randn(8, 3, 32, 32)

y = model(x)

print("=" * 50)
print(model.__class__.__name__)
print("=" * 50)
print("Output shape:", y.shape)