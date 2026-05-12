from __future__ import annotations
from datetime import datetime
from pathlib import Path

from compactlogic import LogicLayer, GroupSum, ConvLayer, SequentialEntropyFreezer, EntropyRowResampler
from compactlogic import FixedSequentialEntropyFreezer

import argparse
import random
import os
import subprocess

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from results_json import ResultsJSON

import mnist_dataset
import ecg_dataset as ecg_dataset_module
from cancer_dataset import CancerInstSegDataset, InfiniteRandomSampler, SegGroupSum

from torch.utils.data import Dataset

import yaml

from tqdm import tqdm


class PostTransformDataset(Dataset):
    def __init__(self, base_ds, post):
        self.base_ds = base_ds
        self.post = post
    def __len__(self): return len(self.base_ds)
    def __getitem__(self, i):
        img, y = self.base_ds[i]
        return self.post(img), y

torch.set_num_threads(1)

def binarize_tensor(t: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (t >= threshold).float()

def get_data_dir(args, name: str) -> str:
    base = Path(args.data_path) if args.data_path else Path('.')
    return str(base / name)


def ensure_tiny_imagenet_available() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / 'data-tinyimagenet'
    tiny_imagenet_root = data_root / 'tiny-imagenet-200'
    train_dir = tiny_imagenet_root / 'train'
    val_dir = tiny_imagenet_root / 'val'

    if train_dir.exists() and val_dir.exists():
        return tiny_imagenet_root

    script_path = Path(__file__).resolve().parent / 'download_tinyimagenet.sh'
    print(f'TinyImageNet data not found at {data_root}. Downloading with {script_path}...')
    subprocess.run(['bash', str(script_path)], check=True, cwd=repo_root)

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f'TinyImageNet download completed, but expected data folders are still missing under {tiny_imagenet_root}.'
        )

    return tiny_imagenet_root

def load_dataset(args):
    validation_loader = None
    if args.dataset in ['mnist']:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),
            torchvision.transforms.ToTensor(),
        ])
        test_transform = torchvision.transforms.ToTensor()
        
        mnist_root = get_data_dir(args, 'data-mnist')
        train_set = mnist_dataset.MNIST(mnist_root, train=True, transform=train_transform, download=True, remove_border=False)
        test_set = mnist_dataset.MNIST(mnist_root, train=False, transform=test_transform, remove_border=False)
        post = lambda t: binarize_tensor(t)
        train_set = PostTransformDataset(train_set, post)
        test_set = PostTransformDataset(test_set, post)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    elif 'cifar-10' in args.dataset:
        num_thd = int(args.dataset.split('-')[2])
        binarize = lambda x: torch.cat([(x > (i + 1) / (num_thd + 1)).float() for i in range(num_thd)], dim=0)
        
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 2, padding_mode='edge'),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(binarize),
        ])
        transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(binarize),
        ])
        
        cifar_root = get_data_dir(args, 'data-cifar')
        train_set = torchvision.datasets.CIFAR10(cifar_root, train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.CIFAR10(cifar_root, train=False, transform=transforms_test)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    elif 'tiny-imagenet' in args.dataset:
        num_thd = int(args.dataset.split('-')[2])
        binarize = lambda x: torch.cat([(x > (i + 1) / (num_thd + 1)).float() for i in range(num_thd)], dim=0)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(64, 4, padding_mode='edge'),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(binarize),
        ])
        transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(binarize),
        ])

        tiny_imagenet_root = ensure_tiny_imagenet_available()
        train_set = torchvision.datasets.ImageFolder(str(tiny_imagenet_root / 'train'), transform=transforms)
        test_set = torchvision.datasets.ImageFolder(str(tiny_imagenet_root / 'val'), transform=transforms_test)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    elif 'ECG' in args.dataset:
        num_thd = int(args.dataset.split('-')[1])
        ecg_root = Path(get_data_dir(args, 'data-ECG'))
        train_loader, test_loader = ecg_dataset_module.load_ecg_loaders(
            train_path=str(ecg_root / 'mitbih_train.csv'),
            test_path=str(ecg_root / 'mitbih_test.csv'),
            num_thd=num_thd,
            use_2d='2D' in args.dataset,
            batch_size=args.batch_size,
        )
    elif 'cancer-segmentation' in args.dataset:
        num_thd = int(args.dataset.split('-')[2])
    
        # Input x after ToTensor() has shape (C, H, W)
        # Concatenate thresholded copies along channel dimension
        binarize = lambda x: torch.cat(
            [(x > (i + 1) / (num_thd + 1)).float() for i in range(num_thd)],
            dim=0
        )
    
        transforms = T.Compose([
            T.ToTensor(),  # (H, W, C) -> (C, H, W) automatically done for numpy array inputs
            T.Lambda(binarize),  # (C, H, W) -> (C * num_thd, H, W)
        ])
    
        data_root = str(Path(get_data_dir(args, 'data-cancer-segmentation')) / 'Part 1')
        if not Path(data_root).exists():
            raise FileNotFoundError(
                f"Cancer segmentation data not found at {data_root}. "
                "Set --data_path so that data-cancer-segmentation/Part 1 exists under it."
            )

        full_set = CancerInstSegDataset(
            root_dir=data_root,
            image_transform=transforms,
        )

        split_generator = torch.Generator().manual_seed(0)
        shuffled_indices = torch.randperm(len(full_set), generator=split_generator).tolist()
        num_train = int(0.9 * len(shuffled_indices))

        train_indices = shuffled_indices[:num_train]
        test_indices = shuffled_indices[num_train:]

        train_set = torch.utils.data.Subset(full_set, train_indices)
        test_set = torch.utils.data.Subset(full_set, test_indices)
    
        seg_num_workers = 8
        train_sampler = InfiniteRandomSampler(train_set, seed=args.seed)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=seg_num_workers,
            persistent_workers=True,
            prefetch_factor=4,
        )
    
        validation_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=seg_num_workers,
            persistent_workers=True,
            prefetch_factor=4,
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=seg_num_workers,
            persistent_workers=True,
            prefetch_factor=4,
        )
    else:
        raise NotImplementedError(f'The data set {args.dataset} is not supported!')

    return train_loader, validation_loader, test_loader

def load_n(loader, n):
    remaining = n
    while remaining > 0:
        for x in loader:
            yield x
            remaining -= 1
            if remaining == 0:
                return

def is_segmentation_dataset(dataset):
    return 'cancer-segmentation' in dataset


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            return batch[0], batch[1]
        if len(batch) >= 3:
            return batch[0], batch[1]
    raise ValueError(f'Unsupported batch structure: {type(batch)}')


def segmentation_target_to_labels(target: torch.Tensor) -> torch.Tensor:
    if target.ndim != 4:
        raise ValueError(f'Segmentation target must have shape (B, C, H, W), got {tuple(target.shape)}')
    return target.argmax(dim=1).long()


def compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.ndim == 4 and target.ndim == 4 and logits.shape == target.shape:
        pred = logits.argmax(dim=1)
        labels = segmentation_target_to_labels(target)
        return (pred == labels).to(torch.float32).mean().item()

    return (logits.argmax(-1) == target).to(torch.float32).mean().item()


def input_dim_of_dataset(dataset):
    if dataset == 'mnist':
        return 784;
    elif 'cifar-10' in dataset:
        num_thd = int(dataset.split('-')[2])
        return 3 * 32 * 32 * num_thd
    elif 'tiny-imagenet' in dataset:
        num_thd = int(dataset.split('-')[2])
        return 3 * 64 * 64 * num_thd
    elif 'ECG' in dataset:
        num_thd = int(dataset.split('-')[1])
        return num_thd * 187

def num_classes_of_dataset(dataset):
    if dataset == 'mnist':
        return 10
    elif 'cifar-10' in dataset:
        return 10
    elif 'tiny-imagenet' in dataset:
        return 200
    elif 'ECG' in dataset:
        return 5
    elif is_segmentation_dataset(dataset):
        return 6

def train(model, x, y, loss_fn, optimizer, lr_scheduler=None, grad_clip_norm: float = 10.0):
    logits = model(x)
    loss = loss_fn(logits, y)
    train_accuracy = compute_accuracy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss.item(), train_accuracy

def eval(model, loader, mode):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)

        res = np.mean(
            [
                compute_accuracy(model(unpack_batch(batch)[0].to('cuda').round()), unpack_batch(batch)[1].to('cuda'))
                for batch in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()

def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    labels = segmentation_target_to_labels(target)
    return torch.nn.functional.cross_entropy(logits, labels)


def input_shape_of_dataset(dataset):
    if 'mnist' in dataset:
        return (1, 28, 28)
    elif 'cifar-10' in dataset:
        num_thd = int(dataset.split('-')[2])
        return (3 * num_thd, 32, 32)
    elif 'tiny-imagenet' in dataset:
        num_thd = int(dataset.split('-')[2])
        return (3 * num_thd, 64, 64)
    elif 'ECG2D' in dataset:
        num_thd = int(dataset.split('-')[1])
        return (1, num_thd, 187)
    elif is_segmentation_dataset(dataset):
        num_thd = int(dataset.split('-')[2])
        return (3 * num_thd, 256, 256)

def get_model(args):
    llkw = dict(num_gates=args.num_gates,
                residual_init=args.residual_init,
               )

    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    logic_layers = []

    k = args.model_scale
    l = args.num_layers
    tau = args.tau

    logic_layers.append(torch.nn.Flatten())
    logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
    for _ in range(l - 1):
        logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

    model = torch.nn.Sequential(
        *logic_layers,
        GroupSum(class_count, tau)
    )

    model = model.to('cuda')

    print(model)
    if args.experiment_id is not None:
        results.store_results({'model_str': str(model)})

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer

def get_model_conv(args):
    # with convolutional channel mixing
    llkw = dict(num_gates=args.num_gates,
                residual_init=args.residual_init,
               )

    in_shape = input_shape_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    k = args.model_scale
    regular_num_neurons = int(1250 * k) if 'tiny-imagenet' in args.dataset else int(625 * k)
    tau = args.tau
    
    num_chn = args.num_chn
    if args.struct == 'conv_chm':  # with dedicated channel mixing layers
        conv_1 = ConvLayer(in_shape=in_shape,         padding=1, c_out=k, ks=3, stride=2, num_chn=num_chn, **llkw)
        conv_2 = ConvLayer(in_shape=conv_1.out_shape, padding=0, c_out=k, ks=1, stride=1, num_chn=0, **llkw)
        conv_3 = ConvLayer(in_shape=conv_2.out_shape, padding=1, c_out=4*k, ks=3, stride=2, num_chn=num_chn, **llkw)
        conv_4 = ConvLayer(in_shape=conv_3.out_shape, padding=0, c_out=4*k, ks=1, stride=1, num_chn=0, **llkw)
        regular_1 = LogicLayer(in_dim=conv_4.out_dim, out_dim=regular_num_neurons, **llkw)
        regular_2 = LogicLayer(in_dim=regular_1.out_dim, out_dim=regular_num_neurons, **llkw)
    elif args.struct == 'conv_spm':  # only spatial mixing at the convolutional stage
        conv_1 = ConvLayer(in_shape=in_shape,         padding=1, c_out=k, ks=3, stride=2, num_chn=num_chn, **llkw)
        conv_2 = ConvLayer(in_shape=conv_1.out_shape, padding=1, c_out=k, ks=3, stride=1, num_chn=num_chn, **llkw)
        conv_3 = ConvLayer(in_shape=conv_2.out_shape, padding=1, c_out=4*k, ks=3, stride=2, num_chn=num_chn, **llkw)
        conv_4 = ConvLayer(in_shape=conv_3.out_shape, padding=1, c_out=4*k, ks=3, stride=1, num_chn=num_chn, **llkw)
        regular_1 = LogicLayer(in_dim=conv_4.out_dim, out_dim=regular_num_neurons, **llkw)
        regular_2 = LogicLayer(in_dim=regular_1.out_dim, out_dim=regular_num_neurons, **llkw)
    else:
        raise NotImplementedError

    logic_layers = [
        conv_1,
        conv_2,
        conv_3,
        conv_4,
        torch.nn.Flatten(),
        regular_1,
        regular_2,
    ]

    model = torch.nn.Sequential(
        *logic_layers,
        GroupSum(class_count, tau)
    )

    model = model.to('cuda')

    print(model)
    if args.experiment_id is not None:
        results.store_results({'model_str': str(model)})

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer

def get_model_seg(args):
    llkw = dict(num_gates=args.num_gates,
                residual_init=args.residual_init,
               )

    in_shape = input_shape_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    k = args.model_scale
    assert k % class_count == 0, (
        f'model_scale k={k} must be an integer multiple of class_count={class_count}'
    )
    
    num_chn = args.num_chn
    if args.struct == 'seg_chm':  # with dedicated channel mixing layers
        conv_1 = ConvLayer(in_shape=in_shape,         padding=1, c_out=k, ks=3, stride=2, num_chn=num_chn, **llkw)
        conv_2 = ConvLayer(in_shape=conv_1.out_shape, padding=0, c_out=k, ks=1, stride=1, num_chn=0, **llkw)
        conv_3 = ConvLayer(in_shape=conv_2.out_shape, padding=1, c_out=4*k, ks=3, stride=2, num_chn=num_chn, **llkw)
        conv_4 = ConvLayer(in_shape=conv_3.out_shape, padding=0, c_out=4*k, ks=1, stride=1, num_chn=0, **llkw)
    elif args.struct == 'seg_spm':  # only spatial mixing at the convolutional stage
        conv_1 = ConvLayer(in_shape=in_shape,         padding=1, c_out=k, ks=3, stride=2, num_chn=num_chn, **llkw)
        conv_2 = ConvLayer(in_shape=conv_1.out_shape, padding=1, c_out=k, ks=3, stride=1, num_chn=num_chn, **llkw)
        conv_3 = ConvLayer(in_shape=conv_2.out_shape, padding=1, c_out=4*k, ks=3, stride=2, num_chn=num_chn, **llkw)
        conv_4 = ConvLayer(in_shape=conv_3.out_shape, padding=1, c_out=4*k, ks=3, stride=1, num_chn=num_chn, **llkw)
    else:
        raise NotImplementedError

    upsample_1_out_shape = (conv_4.out_shape[0], conv_4.out_shape[1] * 2, conv_4.out_shape[2] * 2)
    conv_5 = ConvLayer(in_shape=upsample_1_out_shape, padding=0, c_out=2*k, ks=1, stride=1, num_chn=0, **llkw)
    upsample_2_out_shape = (conv_5.out_shape[0], conv_5.out_shape[1] * 2, conv_5.out_shape[2] * 2)
    conv_6 = ConvLayer(in_shape=upsample_2_out_shape, padding=0, c_out=k, ks=1, stride=1, num_chn=0, **llkw)

    logic_layers = [
        conv_1,
        conv_2,
        conv_3,
        conv_4,
        torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        conv_5,  # (2*k, 128, 128)
        torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        conv_6,  # (k, 256, 256)
        SegGroupSum(class_count),  # (6, 256, 256)
    ]

    model = torch.nn.Sequential(
        *logic_layers,
    )

    model = model.to('cuda')

    print(model)
    if args.experiment_id is not None:
        results.store_results({'model_str': str(model)})

    loss_fn = segmentation_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer

@torch.no_grad()
def reset_adam_state_for_mask(optimizer, param, mask):
    """
    optimizer: torch.optim.Adam
    param: the Parameter whose entries were reinitialized (e.g. model.weights)
    mask: boolean tensor broadcastable to param.shape indicating entries to reset
    """
    st = optimizer.state.get(param, None)
    if st is None or len(st) == 0:
        return

    if "exp_avg" in st:
        st["exp_avg"].masked_fill_(mask, 0.0)
    if "exp_avg_sq" in st:
        st["exp_avg_sq"].masked_fill_(mask, 0.0)
    if "max_exp_avg_sq" in st:
        st["max_exp_avg_sq"].masked_fill_(mask, 0.0)

def make_run_dir(base: str | Path = './results', date_fmt: str = "%Y-%m-%d") -> Path:
    """
    define a function to create a folder with the realtime date as the folder name
    """
    stamp = datetime.now().strftime(date_fmt)
    base = Path(base)
    run_dir = base / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def get_entropy(model):
    r = {}
    for i, module in enumerate(model.modules()):
        if isinstance(module, LogicLayer):
            r[f'{type(module).__name__}_{i}_entropy'] = module.weights_entropy().mean().item()
    return r


def filtered_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.endswith('.w')
    }

def save_checkpoint(path, step, model, optimizer, freezer_info=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "step": step,
        "model_state": filtered_model_state_dict(model),
        "optim_state": optimizer.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "freezer_info": freezer_info if freezer_info is not None else ""
    }

    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer, map_location="cpu", strict=False):
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model_state"], strict=strict)
    model.to('cuda')
    for layer in model:
        if hasattr(layer, 'w'):
            layer.is_frozen = True

    optimizer.load_state_dict(ckpt["optim_state"])

    device = next(model.parameters()).device
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                    state[k] = v.to(device)

    rng = ckpt.get("rng_state")
    if rng is not None:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    step = ckpt.get("step", None)
    freezer_info = ckpt.get("freezer_info", None)
    print(f"restored from {path}:{step}, freezer_info={freezer_info}")

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file'
    )
    
    parser.add_argument(
        '-eid', '--experiment_id',
        type=str,
        default=datetime.now().strftime("%H%M%S"),
        help='Experiment ID (default: current time hhmmss)'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Base path containing dataset folders like data-mnist and data-cifar'
    )

    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Base path under which results/ will be created for training outputs'
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Optional path to a .pt checkpoint file to restore before training'
    )
    
    parser.add_argument('--dataset', type=str, help='the dataset to use')
    # e.g., mnist, cifar-10-3-thresholds, tiny-imagenet-7-thresholds, ECG-3-thresholds
    
    parser.add_argument('--struct', type=str, choices=['regular', 'conv_chm', 'conv_spm', 'seg_chm', 'seg_spm'])
    parser.add_argument('--num_chn', type=int)
    
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Global grad-norm clipping threshold (default: 10.0)')
    
    parser.add_argument('--num_iterations', '-ni', type=int, default=200_001, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval_freq', '-ef', type=int, default=1_000, help='Evaluation frequency (default: 2_000)')
    
    parser.add_argument('--model_scale', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int, default=0, help='only DLGN needs num_layers')
    parser.add_argument('--tau', '-t', type=float, help='the softmax temperature tau')
    
    parser.add_argument('--num_gates', type=int, default=16)
    parser.add_argument('--residual_init', action='store_true', help='Enable residual initialization')
    
    # resampling arguments
    parser.add_argument('--resample', action='store_true', help='Enable resampler')
    parser.add_argument('--patience', type=int)
    parser.add_argument('--max_iters', type=int)
    
    # freezing arguments
    parser.add_argument('--freeze', action='store_true', help='Enable freezer')
    parser.add_argument('--min_iters', type=int)
    parser.add_argument('--freeze_num_layers', type=int)
    parser.add_argument('--fixed_freeze', action='store_true')
    parser.add_argument('--period', type=int)
    
    args_pre, _ = parser.parse_known_args()

    if args_pre.config:
        cfg = load_config(args_pre.config)
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    
    max_iters = args.max_iters if args.max_iters is not None else np.inf
    min_iters = args.min_iters if args.min_iters is not None else np.inf
    
    results_base_dir = Path('./results') if args.save_path is None else Path(args.save_path) / 'results'
    RUN_DIR = make_run_dir(results_base_dir) / Path(f'{args.experiment_id}')
    print(f"Saving to: {RUN_DIR.resolve()}")
    
    print(vars(args))
    
    assert (args.num_iterations - 1) % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations - 1}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )
    
    results = ResultsJSON(eid=args.experiment_id, path=str(RUN_DIR))
    results.store_args(args)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    train_loader, validation_loader, test_loader = load_dataset(args)
    if args.struct.startswith('seg_'):
        model, loss_fn, optim = get_model_seg(args)
    elif 'conv' in args.struct:
        model, loss_fn, optim = get_model_conv(args)
    elif args.struct == 'regular':
        model, loss_fn, optim = get_model(args)
    else:
        raise NotImplementedError(f'Illegal model structure: {args.struct}')

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optim)
    
    if args.resample:
        kw = dict(
            threshold_high = 0.95,
            threshold_low = 0.4,
            keep_strong_weights = False
        )
        resampler = EntropyRowResampler(
            model=model,
            patience=args.patience,
            max_iters=args.max_iters if args.max_iters is not None else np.inf,
            **kw
        )
    
    if args.freeze:
        if args.fixed_freeze:
            freezer = FixedSequentialEntropyFreezer(
                model,
                min_iters=args.min_iters,
                N=args.period,
                l=args.freeze_num_layers,
            )
        else:
            freezer = SequentialEntropyFreezer(
                model,
                min_iters=args.min_iters,
                l=args.freeze_num_layers,
            )
    
    results.store_results({'resampler': str(resampler) if args.resample else 'None'})
    results.store_results({'freezer': str(freezer) if args.freeze else 'None'})
    
    best_acc_resample = 0
    best_acc_freeze = 0
    
    last_hard_eval_accuracy = None
    loss_ema = None
    train_accuracy_ema = None
    pbar_ema_momentum = 0.9

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc='iteration',
        total=args.num_iterations,
    )
    for i, batch in pbar:

        x, y = unpack_batch(batch)
        x = x.to(torch.float32).to('cuda')
        y = y.to('cuda')
    
        loss, train_accuracy = train(model, x, y, loss_fn, optim, grad_clip_norm=args.grad_clip_norm)
        if loss_ema is None:
            loss_ema = loss
        else:
            loss_ema = (
                pbar_ema_momentum * loss_ema
                + (1 - pbar_ema_momentum) * loss
            )
        if train_accuracy_ema is None:
            train_accuracy_ema = train_accuracy
        else:
            train_accuracy_ema = (
                pbar_ema_momentum * train_accuracy_ema
                + (1 - pbar_ema_momentum) * train_accuracy
            )
    
        if (i) % args.eval_freq == 0 and i != 0:
            eval_train_loader = validation_loader if validation_loader is not None else train_loader
            train_accuracy_train_mode = eval(model, eval_train_loader, mode=True)
            train_accuracy_eval_mode = eval(model, eval_train_loader, mode=False)
            test_accuracy_eval_mode = eval(model, test_loader, mode=False)
            test_accuracy_train_mode = eval(model, test_loader, mode=True)
            last_hard_eval_accuracy = test_accuracy_eval_mode
    
            r = {
                'train_acc_eval_mode': train_accuracy_eval_mode,
                'train_acc_train_mode': train_accuracy_train_mode,
                'test_acc_eval_mode': test_accuracy_eval_mode,
                'test_acc_train_mode': test_accuracy_train_mode,
            } | get_entropy(model)
    
            results.store_results(r)
    
            if test_accuracy_eval_mode > best_acc_resample and i <= max_iters:
                best_acc_resample = test_accuracy_eval_mode
                save_checkpoint(str(RUN_DIR) + f'/ckpt_best_resample.pt', i, model, optim)
    
            if test_accuracy_eval_mode > best_acc_freeze and i >= min_iters:
                best_acc_freeze = test_accuracy_eval_mode
                save_checkpoint(str(RUN_DIR) + f'/ckpt_best_freeze.pt', i, model, optim)
    
            results.save()

        if args.resample:
            resampler.step(i, optim)
        if args.freeze:
            freezer.step(i)

        postfix = {
            'soft_loss_ema': f'{loss_ema:.4f}',
            'soft_train_acc_ema': f'{train_accuracy_ema:.4f}',
        }
        if last_hard_eval_accuracy is not None:
            postfix['last_hard_test_acc'] = f'{last_hard_eval_accuracy:.4f}'
        pbar.set_postfix(postfix)
