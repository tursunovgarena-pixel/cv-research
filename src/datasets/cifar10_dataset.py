import random

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from src.preprocessing.image_degradation import ImageDegradation


class CIFAR10Dataset(Dataset):

    def __init__(
        self,
        train=True,
        degradation=None
    ):

        self.dataset = CIFAR10(
            root="./data/raw",
            train=train,
            download=False,
        )

        self.transform = ToTensor()

        self.degradation = degradation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, label = self.dataset[index]

        if self.degradation is not None:
            image = ImageDegradation.apply(
                image,
                self.degradation
            )

        image = self.transform(image)

        return image, label