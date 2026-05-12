from __future__ import annotations

from typing import Any

import torch

from compactlogic import ConvLayer, GroupSum, LogicLayer


def input_dim_of_dataset(dataset: str) -> int:
    if dataset == 'mnist':
        return 784
    if 'cifar-10' in dataset:
        num_thd = int(dataset.split('-')[2])
        return 3 * 32 * 32 * num_thd
    if 'tiny-imagenet' in dataset:
        num_thd = int(dataset.split('-')[2])
        return 3 * 64 * 64 * num_thd
    if 'ECG' in dataset:
        num_thd = int(dataset.split('-')[1])
        return num_thd * 187
    raise ValueError(f'Unsupported dataset for compactlogic compiler model builder: {dataset!r}')


def input_shape_of_dataset(dataset: str) -> tuple[int, ...]:
    if dataset == 'mnist':
        return (1, 28, 28)
    if 'cifar-10' in dataset:
        num_thd = int(dataset.split('-')[2])
        return (3 * num_thd, 32, 32)
    if 'tiny-imagenet' in dataset:
        num_thd = int(dataset.split('-')[2])
        return (3 * num_thd, 64, 64)
    if 'ECG2D' in dataset:
        num_thd = int(dataset.split('-')[1])
        return (1, num_thd, 187)
    raise ValueError(f'Unsupported dataset for compactlogic compiler model builder: {dataset!r}')


def num_classes_of_dataset(dataset: str) -> int:
    if dataset == 'mnist':
        return 10
    if 'cifar-10' in dataset:
        return 10
    if 'tiny-imagenet' in dataset:
        return 200
    if 'ECG' in dataset:
        return 5
    raise ValueError(f'Unsupported dataset for compactlogic compiler model builder: {dataset!r}')


def build_compactlogic_model(args: dict[str, Any]) -> torch.nn.Sequential:
    """Rebuild the compactlogic topology from saved experiment args.

    This compiler-local copy intentionally lives under ``compiler/`` so the
    compiler/simulation directories remain shippable on their own.
    """
    dataset = str(args.get('dataset'))
    struct = str(args.get('struct'))
    num_gates = int(args['num_gates'])
    residual_init = bool(args.get('residual_init', False))
    tau = float(args['tau'])

    layer_kwargs = {
        'num_gates': num_gates,
        'residual_init': residual_init,
    }

    if struct == 'regular':
        in_dim = input_dim_of_dataset(dataset)
        width = int(args['model_scale'])
        num_layers = int(args['num_layers'])

        modules: list[torch.nn.Module] = [torch.nn.Flatten()]
        modules.append(LogicLayer(in_dim=in_dim, out_dim=width, **layer_kwargs))
        for _ in range(num_layers - 1):
            modules.append(LogicLayer(in_dim=width, out_dim=width, **layer_kwargs))
        modules.append(GroupSum(num_classes_of_dataset(dataset), tau))
        return torch.nn.Sequential(*modules)

    if struct in {'conv_spm', 'conv_chm'}:
        in_shape = input_shape_of_dataset(dataset)
        class_count = num_classes_of_dataset(dataset)
        k = int(args['model_scale'])
        regular_num_neurons = int(1250 * k) if 'tiny-imagenet' in dataset else int(625 * k)
        num_chn = int(args.get('num_chn', 1))

        if struct == 'conv_chm':
            conv_1 = ConvLayer(in_shape=in_shape, padding=1, c_out=k, ks=3, stride=2, num_chn=num_chn, **layer_kwargs)
            conv_2 = ConvLayer(in_shape=conv_1.out_shape, padding=0, c_out=k, ks=1, stride=1, num_chn=0, **layer_kwargs)
            conv_3 = ConvLayer(in_shape=conv_2.out_shape, padding=1, c_out=4 * k, ks=3, stride=2, num_chn=num_chn, **layer_kwargs)
            conv_4 = ConvLayer(in_shape=conv_3.out_shape, padding=0, c_out=4 * k, ks=1, stride=1, num_chn=0, **layer_kwargs)
        else:
            conv_1 = ConvLayer(in_shape=in_shape, padding=1, c_out=k, ks=3, stride=2, num_chn=num_chn, **layer_kwargs)
            conv_2 = ConvLayer(in_shape=conv_1.out_shape, padding=1, c_out=k, ks=3, stride=1, num_chn=num_chn, **layer_kwargs)
            conv_3 = ConvLayer(in_shape=conv_2.out_shape, padding=1, c_out=4 * k, ks=3, stride=2, num_chn=num_chn, **layer_kwargs)
            conv_4 = ConvLayer(in_shape=conv_3.out_shape, padding=1, c_out=4 * k, ks=3, stride=1, num_chn=num_chn, **layer_kwargs)

        regular_1 = LogicLayer(in_dim=conv_4.out_dim, out_dim=regular_num_neurons, **layer_kwargs)
        regular_2 = LogicLayer(in_dim=regular_1.out_dim, out_dim=regular_num_neurons, **layer_kwargs)
        return torch.nn.Sequential(
            conv_1,
            conv_2,
            conv_3,
            conv_4,
            torch.nn.Flatten(),
            regular_1,
            regular_2,
            GroupSum(class_count, tau),
        )

    raise ValueError(f'Unsupported structure for compactlogic compiler model builder: {struct!r}')


__all__ = [
    'build_compactlogic_model',
    'input_dim_of_dataset',
    'input_shape_of_dataset',
    'num_classes_of_dataset',
]
