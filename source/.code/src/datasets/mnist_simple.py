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
        # A small run reads the tiny offline stand-in instead of real MNIST.
        if small:
            mnist_class = functools.partial(SmallMNIST, seed=seed)
        else:
            mnist_class = torchvision.datasets.MNIST

        self.X = torch.zeros((32, 28, 28), dtype=torch.uint8)
        self.y = torch.zeros(32, dtype=torch.long)
        ## Load MNIST with mnist_class here
        ## YOUR CODE HERE
        # -- start --
        dataset = mnist_class(root='~/', train=train, download=True)
        self.X = dataset.data
        self.y = dataset.targets
        # -- end --


    def __len__(self):
        res = 32
        ## Return number of items that is there in the dataset
        ## YOUR CODE HERE
        # -- start --
        res = len(self.y)
        # -- end --
        return res


    def __getitem__(self, index):
        sample = {
            'image': self.X[index].float(),
            'label': self.y[index].long(),
        }

        ## Return a sample of the dataset that corresponds to the input index
        ## YOUR CODE HERE
        # -- start --
        sample['image'] = self.X[index, :, :].float() / 255.0 * 2 - 1
        sample['label'] = self.y[index].long()
        # -- end --
        
        return sample 
