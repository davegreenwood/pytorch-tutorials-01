"""Datasets and data loaders.
Showing some examplkes of manipulating data loading."""

import os
import torch
import torchvision
import torchvision.transforms as transforms


class ReshapeTransform:
    """An example of a callable for the transform stack."""
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class Data:
    """Torchvision datasets base class.
    """

    def __init__(self, **kwargs):

        # get kwargs or use defaults
        self.batch_size = kwargs.get("batch_size", 10)
        self.num_workers = kwargs.get("num_workers", 1)
        self.shuffle = kwargs.get("shuffle", True)
        self.download = kwargs.get("download", True)
        self.root = kwargs.get("root", os.path.abspath("./data"))
        self.dataset = kwargs.get("dataset", torchvision.datasets.MNIST)
        self.transform = kwargs.get(
            "transform", transforms.Compose([transforms.ToTensor()])
        )

        os.makedirs(self.root, exist_ok=True)
        dsargs = dict(root=self.root, download=True, transform=self.transform)
        dlargs = dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

        self.trainset = self.dataset(train=True, **dsargs)
        self.testset = self.dataset(train=False, **dsargs)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, **dlargs)
        self.testloader = torch.utils.data.DataLoader(self.testset, **dlargs)

    def get_loaders(self):
        """get the tarin and test loaders"""
        return self.trainloader, self.testloader


class MNISTMLP(Data):
    """The Mnist Data returned as flattened examples."""
    def __init__(self, **kwargs):
        reshape = ReshapeTransform((-1, ))
        toten = transforms.ToTensor()
        kwargs["transform"] = transforms.Compose([toten, reshape])
        super().__init__(**kwargs)

