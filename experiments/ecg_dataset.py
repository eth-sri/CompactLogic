from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Binarisation helpers
# ---------------------------------------------------------------------------

def binarize_flat(x: torch.Tensor, num_thd: int) -> torch.Tensor:
    """Binarise a 1-D signal into num_thd threshold channels, then flatten.

    Args:
        x: tensor of shape (1, L)
        num_thd: number of threshold levels

    Returns:
        tensor of shape (num_thd * L,)
    """
    bins = [(x > (i + 1) / (num_thd + 1)).float() for i in range(num_thd)]
    return torch.cat(bins, dim=1).squeeze(0)


def binarize_2D(x: torch.Tensor, num_thd: int) -> torch.Tensor:
    """Binarise a 1-D signal into a 2-D threshold map.

    Args:
        x: tensor of shape (1, L) or (L,)
        num_thd: number of threshold levels

    Returns:
        tensor of shape (1, num_thd, L)
    """
    if x.dim() == 2:
        x = x.squeeze(0)  # (L,)

    thresholds = torch.linspace(1, num_thd, steps=num_thd, device=x.device) / (num_thd + 1)
    x_expanded = x.unsqueeze(0)          # (1, L)
    thresholds = thresholds.unsqueeze(1)  # (num_thd, 1)

    return (x_expanded > thresholds).float().unsqueeze(0)  # (1, num_thd, L)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class ECGDataset(Dataset):
    """Flat (1-D) binarised ECG dataset.

    Each sample is binarised with *num_thd* thresholds and returned as a
    flat tensor of shape ``(num_thd * L,)``.
    """

    def __init__(self, X: np.ndarray, y: list | np.ndarray, num_thd: int) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.num_thd = num_thd

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        x = binarize_flat(x, self.num_thd)
        return x, self.y[idx]


class ECGDataset2D(Dataset):
    """2-D binarised ECG dataset.

    Each sample is binarised with *num_thd* thresholds and returned as a
    tensor of shape ``(1, num_thd, L)``.
    """

    def __init__(self, X: np.ndarray, y: list | np.ndarray, num_thd: int) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.num_thd = num_thd

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]                    # (1, L)
        x = binarize_2D(x, self.num_thd)   # (1, num_thd, L)
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# Public loader factory
# ---------------------------------------------------------------------------

def load_ecg_loaders(
    train_path: str = "./data-ECG/mitbih_train.csv",
    test_path: str = "./data-ECG/mitbih_test.csv",
    num_thd: int = 3,
    use_2d: bool = False,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Load the ECG Kaggle (MIT-BIH) dataset and return train / test DataLoaders.

    The function mirrors the preprocessing performed inside
    ``experiment_script.py::load_dataset`` for the ``'ECG'`` branch:

    1. Read train/test CSVs (last column = label, rest = signal).
    2. Add a channel dimension: ``(N, 1, L)``.
    3. Min–max normalise across the combined train+test range.
    4. Remap integer class labels to a contiguous 0-based index.
    5. Wrap in :class:`ECGDataset` (flat) or :class:`ECGDataset2D` (2-D).
    6. Return ``torch.utils.data.DataLoader`` objects.

    Args:
        train_path: Path to ``mitbih_train.csv``.
        test_path:  Path to ``mitbih_test.csv``.
        num_thd:    Number of binarisation thresholds.
        use_2d:     If ``True``, return 2-D binarised samples ``(1, num_thd, L)``
                    via :class:`ECGDataset2D`; otherwise return flat samples
                    ``(num_thd * L,)`` via :class:`ECGDataset`.
        batch_size: Mini-batch size for both loaders.
        num_workers: Number of DataLoader worker processes.

    Returns:
        ``(train_loader, test_loader)`` — a pair of
        :class:`torch.utils.data.DataLoader` instances.
    """
    try:
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\n"
            "ECG data not found. Download the MIT-BIH Arrhythmia dataset from:\n"
            "  https://www.kaggle.com/datasets/shayanfazeli/heartbeat\n"
            "and place mitbih_train.csv and mitbih_test.csv in the data-ECG/ "
            "folder at the root of this repository."
        ) from None

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    # Add channel dimension: (N, L) -> (N, 1, L)
    X_train = X_train[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    # Global min–max normalisation
    X_min = min(X_train.min(), X_test.min())
    X_max = max(X_train.max(), X_test.max())
    X_train = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_test = (X_test - X_min) / (X_max - X_min + 1e-8)

    # Remap class labels to contiguous 0-based indices
    all_classes = sorted(set(y_train) | set(y_test))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    y_train = [class_to_idx[c] for c in y_train]
    y_test = [class_to_idx[c] for c in y_test]

    dataset_cls = ECGDataset2D if use_2d else ECGDataset
    train_set = dataset_cls(X_train, y_train, num_thd)
    test_set = dataset_cls(X_test, y_test, num_thd)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
