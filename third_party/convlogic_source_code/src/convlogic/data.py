import math
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T


class ConvLogicDataModule(pl.LightningDataModule):
    """
    Data module for MNIST and CIFAR-10 with threshold-based transforms.
    Splits into train, validation, and test sets, and produces DataLoaders.
    """

    def __init__(
        self,
        dataset_name="cifar10-3",
        data_dir="../../data/",
        batch_size=128,
        num_workers=2,
        threshold_levels=None,
        train_val_split=0.9,
        threshold_type="uniform",
    ):
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.data_dir = os.path.abspath(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.threshold_type = threshold_type.lower()

        if threshold_levels is not None:
            self.threshold_levels = threshold_levels
        else:
            if "-" in dataset_name:
                try:
                    self.threshold_levels = int(dataset_name.split("-")[-1])
                except ValueError:
                    raise ValueError(f"Invalid threshold suffix: {dataset_name}") from None
            else:
                self.threshold_levels = 1

    def build_transforms(self):
        def _apply_thresholds(x):
            return torch.cat([(x > t).float() for t in self.thresholds], dim=0)

        return T.Compose([T.ToTensor(), T.Lambda(_apply_thresholds)])

    def compute_thresholds_from_dataset(self, dataset):
        x_all = torch.stack([x for x, _ in dataset])
        x_flat = x_all.view(-1)

        if self.threshold_type == "uniform":
            thresholds = torch.linspace(1, self.threshold_levels, self.threshold_levels, dtype=torch.float32) / (
                self.threshold_levels + 1
            )
            print(f"Computed uniform thresholds: {thresholds}")

        elif self.threshold_type == "distributive":
            sorted_x = torch.sort(x_flat)[0]
            idx = (sorted_x.shape[0] * torch.arange(1, self.threshold_levels + 1) / (self.threshold_levels + 1)).long()
            thresholds = sorted_x[idx]
            print(f"Computed distributive thresholds: {thresholds}")

        else:
            raise ValueError(f"Unknown threshold_type: {self.threshold_type}")

        return thresholds

    def setup(self, stage: str = None):
        if self.dataset_name.startswith("tinyimagenet"):
            self._setup_tinyimagenet()
            return

        if self.dataset_name.startswith("cifar10"):
            dataset_class = datasets.CIFAR10
        elif self.dataset_name.startswith("mnist"):
            dataset_class = datasets.MNIST
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        raw_train_set = dataset_class(
            root=self.data_dir,
            train=True,
            download=True,
            transform=T.ToTensor(),
        )

        # Compute thresholds
        self.thresholds = self.compute_thresholds_from_dataset(raw_train_set)
        self.transforms = self.build_transforms()

        if self.dataset_name.startswith("cifar10"):
            def _apply_thresholds(x):
                return torch.cat([(x > t).float() for t in self.thresholds], dim=0)

            train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(32, padding=2, padding_mode="edge"),
                T.ToTensor(),
                T.Lambda(_apply_thresholds),
            ])
        else:
            train_transform = self.transforms

        full_train_set = dataset_class(
            root=self.data_dir,
            train=True,
            download=False,
            transform=train_transform,
        )
        self.test_set = dataset_class(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transforms,
        )

        train_size = math.ceil(self.train_val_split * len(full_train_set))
        valid_size = len(full_train_set) - train_size
        self.train_set, self.valid_set = random_split(full_train_set, [train_size, valid_size])

    def _setup_tinyimagenet(self):
        tiny_dir = os.path.join(self.data_dir, "tiny-imagenet-200")

        # Uniform thresholds don't require iterating the dataset; distributive does.
        if self.threshold_type == "distributive":
            raw_train_set = datasets.ImageFolder(os.path.join(tiny_dir, "train"), transform=T.ToTensor())
            self.thresholds = self.compute_thresholds_from_dataset(raw_train_set)
        else:
            self.thresholds = torch.linspace(
                1, self.threshold_levels, self.threshold_levels, dtype=torch.float32
            ) / (self.threshold_levels + 1)
            print(f"Computed uniform thresholds: {self.thresholds}")

        def _apply_thresholds(x):
            return torch.cat([(x > t).float() for t in self.thresholds], dim=0)

        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(64, padding=4, padding_mode="edge"),
            T.ToTensor(),
            T.Lambda(_apply_thresholds),
        ])
        test_transform = T.Compose([T.ToTensor(), T.Lambda(_apply_thresholds)])

        full_train_set = datasets.ImageFolder(os.path.join(tiny_dir, "train"), transform=train_transform)
        self.test_set = datasets.ImageFolder(os.path.join(tiny_dir, "val"), transform=test_transform)

        train_size = math.ceil(self.train_val_split * len(full_train_set))
        valid_size = len(full_train_set) - train_size
        self.train_set, self.valid_set = random_split(full_train_set, [train_size, valid_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            num_workers=self.num_workers,
        )

    @property
    def input_channels(self):
        """
        Returns the number of input channels after thresholding:
        - MNIST: threshold_levels x 1
        - CIFAR-10: threshold_levels x 3
        """
        if self.dataset_name.startswith("mnist"):
            return self.threshold_levels
        elif self.dataset_name.startswith("cifar10") or self.dataset_name.startswith("tinyimagenet"):
            return 3 * self.threshold_levels
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
