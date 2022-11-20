import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Callable, Optional, Any, Tuple
import matplotlib.pyplot as plt

class CIFAR10WithID(torchvision.datasets.CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = super().__getitem__(index)
        return image, label, index

if __name__ == "__main__":

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # trainset1 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # a = trainset1.__getitem__(0)
    # print(a)
    # trainloader = torch.utils.data.DataLoader(trainset1, batch_size=10, shuffle=False, num_workers=1)
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    #     inputs, targets = inputs, targets
    #     print(targets)

    #Great. below shows this returned the index of the samples as well.
    trainset2 = CIFAR10WithID(root='./data', train=True, download=True, transform = transform_train)
    # b = trainset2.__getitem__(0)
    # print(b)
    trainloader = torch.utils.data.DataLoader(trainset2, batch_size=10, shuffle=False, num_workers=1)
    for batch_idx, (inputs, targets, index) in enumerate(trainloader):
        inputs, targets = inputs, targets
        print(index)







