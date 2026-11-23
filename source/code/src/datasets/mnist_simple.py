import functools

import torch
import torchvision.datasets


class SmallMNIST:
    """Tiny offline stand-in for ``torchvision.datasets.MNIST``.

    The small run uses it so that grading needs no download: random 28 x 28
    ``uint8`` images, 512 for training and 128 for validation, with labels
    0, 1, ..., 9 repeated. ``root`` and ``download`` are accepted, like in
    ``torchvision.datasets.MNIST``, and ignored.
    """

    def __init__(self, root=None, train=True, download=False, seed=0):
        generator = torch.Generator().manual_seed(seed + (0 if train else 1000))
        size = 512 if train else 128
        self.data = torch.randint(0, 256, (size, 28, 28), generator=generator,
                                  dtype=torch.uint8)
        self.targets = torch.arange(size) % 10


class MNISTSimpleDataset:
    def __init__(self, train=True, small=False, seed=0):
        if small:
            mnist_class = functools.partial(SmallMNIST, seed=seed)
        else:
            mnist_class = torchvision.datasets.MNIST

        dataset = mnist_class(root='~/', train=train, download=True)
        self.X = dataset.data
        self.y = dataset.targets


    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, index):
        image = self.X[index].float() / 127.5 - 1.0
        label = self.y[index].long()
        return {
            'image': image,
            'label': label,
        }
