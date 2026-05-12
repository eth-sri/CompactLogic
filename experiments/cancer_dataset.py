import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class InfiniteRandomSampler(torch.utils.data.Sampler[int]):
    def __init__(self, data_source, seed: int = 0):
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        n = len(self.data_source)
        while True:
            yield from torch.randperm(n, generator=g).tolist()

    def __len__(self):
        return 2**63 - 1


class SegGroupSum(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c % self.num_classes == 0, (
            f'channel count ({c}) must be divisible by num_classes ({self.num_classes})'
        )
        group_size = c // self.num_classes
        return x.view(b, self.num_classes, group_size, h, w).sum(dim=2)


def _load_npy_lazy(path):
    try:
        return np.load(path, mmap_mode='r')
    except Exception:
        return np.load(path, allow_pickle=True)


def _replace_with_converted_npy(path: Path, target_dtype: np.dtype, chunk_size: int = 32) -> None:
    src = _load_npy_lazy(path)
    if src.dtype == target_dtype:
        return

    tmp_path = path.with_suffix(path.suffix + '.tmp')
    dst = np.lib.format.open_memmap(
        tmp_path,
        mode='w+',
        dtype=target_dtype,
        shape=src.shape,
    )

    n = len(src)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dst[start:end] = np.asarray(src[start:end], dtype=target_dtype)
        print(f'Converting {path.name}: {end}/{n}')

    del dst
    del src
    os.replace(tmp_path, path)


def ensure_cancer_dataset_converted(root_dir: str | Path, chunk_size: int = 32) -> None:
    root = Path(root_dir)
    image_path = root / 'Images' / 'images.npy'
    mask_path = root / 'Masks' / 'masks.npy'

    image_arr = _load_npy_lazy(image_path)
    mask_arr = _load_npy_lazy(mask_path)
    image_dtype = image_arr.dtype
    mask_dtype = mask_arr.dtype
    del image_arr
    del mask_arr

    if image_dtype == np.float32 and mask_dtype == np.uint8:
        return

    print(
        'Cancer dataset storage conversion needed: '
        f'images={image_dtype} -> float32, masks={mask_dtype} -> uint8'
    )
    _replace_with_converted_npy(image_path, np.float32, chunk_size=chunk_size)
    _replace_with_converted_npy(mask_path, np.uint8, chunk_size=chunk_size)
    print('Cancer dataset storage conversion complete.')


class CancerInstSegDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None, auto_convert=True):
        self.root_dir = root_dir

        if auto_convert:
            ensure_cancer_dataset_converted(root_dir)

        self.images = _load_npy_lazy(os.path.join(root_dir, 'Images', 'images.npy'))
        self.masks = _load_npy_lazy(os.path.join(root_dir, 'Masks', 'masks.npy'))
        self.types = _load_npy_lazy(os.path.join(root_dir, 'Images', 'types.npy'))

        assert len(self.images) == len(self.masks) == len(self.types)

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(self.images[idx], copy=True)
        mask = np.array(self.masks[idx], copy=True)
        cell_type = self.types[idx]

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            if isinstance(mask, np.ndarray):
                if mask.ndim == 2:
                    mask = torch.from_numpy(mask).long()
                elif mask.ndim == 3:
                    mask = torch.from_numpy(mask).permute(2, 0, 1).long()
                else:
                    raise ValueError(f"Unsupported mask shape: {mask.shape}")
            else:
                mask = torch.tensor(mask, dtype=torch.long)

        try:
            cell_type = torch.tensor(cell_type)
        except Exception:
            pass

        return image, mask, cell_type


def stable_sample_key(image, mask, cell_type):
    h = hashlib.sha1()

    image_np = np.ascontiguousarray(image)
    mask_np = np.ascontiguousarray(mask)

    h.update(str(image_np.shape).encode())
    h.update(str(image_np.dtype).encode())
    h.update(image_np.tobytes())

    h.update(str(mask_np.shape).encode())
    h.update(str(mask_np.dtype).encode())
    h.update(mask_np.tobytes())

    if isinstance(cell_type, np.ndarray):
        cell_type_np = np.ascontiguousarray(cell_type)
        h.update(str(cell_type_np.shape).encode())
        h.update(str(cell_type_np.dtype).encode())
        h.update(cell_type_np.tobytes())
    else:
        h.update(str(cell_type).encode())

    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert the cancer segmentation dataset in place to float32 images and uint8 masks.',
    )
    parser.add_argument(
        '--root_dir',
        type=Path,
        default=Path('data-cancer-segmentation/Part 1'),
        help='Dataset root containing Images/ and Masks/.',
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=32,
        help='Number of samples to convert per chunk.',
    )
    args = parser.parse_args()

    ensure_cancer_dataset_converted(args.root_dir, chunk_size=args.chunk_size)


if __name__ == '__main__':
    main()
