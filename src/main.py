import torch

print("=" * 50)
print("Computer Vision Research Project")
print("=" * 50)

print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not found")