import os

import torchvision.transforms as t
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10


class DataManager(object):
    def __init__(self, dataset_type, batch_size, cuda, verbose):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.cuda = cuda
        self.verbose = verbose
        self._init_data_sets()
        self._init_data_loaders()
        self._print_data_info()

    def _get_transforms(self, train=True):
        if train:
            return t.Compose([
                t.ToTensor()
            ])
        else:
            return t.Compose([
                t.ToTensor()
            ])

    def _init_data_sets(self):
        download = not os.path.exists(os.path.join("./data_manager/datasets", self.dataset_type))
        if self.dataset_type == 'mnist':
            self._train_dataset = MNIST(root='./mnist',
                                        train=True,
                                        download=download,
                                        transform=self._get_transforms())
            self._valid_dataset = MNIST(root='./mnist',
                                        train=False,
                                        download=download,
                                        transform=self._get_transforms(train=False))
        else:
            self._train_dataset = CIFAR10(root='./cifar10',
                                          train=True,
                                          download=download,
                                          transform=self._get_transforms())
            self._valid_dataset = CIFAR10(root='./cifar10',
                                          train=False,
                                          download=download,
                                          transform=self._get_transforms(train=False))

    def _init_data_loaders(self):
        self._train_dataloader = DataLoader(dataset=self._train_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=self.cuda)
        self._valid_dataloader = DataLoader(dataset=self._valid_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=self.cuda)

    def _print_data_info(self):
        if self.verbose > 0:
            print("="*20, "Data info", "="*20)
            print("\t Dataset:                      {}\n"
                  "\t Number of training images:    {}\n"
                  "\t Number of training classes:   {}\n"
                  "\t Batch size:                   {}".format(
                    self.dataset_type, len(self._train_dataset), len(self._train_dataset.classes), self.batch_size)
            )
            print("="*51)
        else:
            pass

    def get_data_sets(self):
        return self._train_dataset, self._valid_dataset

    def get_data_loaders(self):
        return self._train_dataloader, self._valid_dataloader

    def get_num_labels(self):
        return len(self._train_dataset.classes)

    def get_batch_size(self):
        return self.batch_size
