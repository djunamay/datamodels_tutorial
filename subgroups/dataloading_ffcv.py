"""
This module provides functions to create data loaders for training, validation, and testing datasets using FFCV.

Functions:
    get_val_split_indices: Generate train and validation split indices based on the given seed, dataset length, and alpha.
    get_binary_labels: Generate binary labels for the CIFAR-10 dataset based on the given animate labels.
    get_binary_label_indices: Get the indices of animate and inanimate labels based on the binary labels.
    make_dataloaders: Create data loaders for training, validation, and testing datasets.

Note:
    The function `make_dataloaders` was adapted from https://github.com/libffcv/ffcv/blob/main/examples/cifar/train_cifar.py
"""

from ffcv.loader import Loader, OrderOption
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import (RandomHorizontalFlip, Cutout, RandomTranslate, Convert, 
                             ToDevice, ToTensor, ToTorchImage, ReplaceLabel)
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

import numpy as np
import torch as ch
import torchvision
import time
from typing import List, Optional

def get_val_split_indices(seed: int, length: int, alpha: float, get_val_samples: bool = False) -> tuple:
    """
    Generate train and validation split indices based on the given seed, dataset length, and alpha.

    Parameters
    ----------
    seed : int
        The random seed.
    length : int
        The total number of samples in the dataset.
    alpha : float
        The proportion of the dataset to use for training.

    Returns
    -------
    tuple
        A tuple containing the train and validation indices.
    """

    rng = np.random.default_rng(seed) 
    mask = rng.random(length)>(1-alpha)
    train_samples = np.nonzero(mask)[0]

    if get_val_samples:
        all_samples = np.arange(length)
        val_samples = np.array(list(set(all_samples) - set(train_samples)))
        return train_samples, val_samples
    else:
        return train_samples


def make_dataloaders(train_dataset: str, test_dataset: str, batch_size: int, num_workers: int, seed: int, alpha: float, length: int, test_batch_size: int = None, get_val_samples: bool = False, no_transform: bool = False, return_sequential: bool = False) -> tuple:
    """
    Create data loaders for training, validation, and testing datasets.

    Parameters
    ----------
    train_dataset : str
        Path to the training dataset.
    test_dataset : str
        Path to the testing dataset.
    batch_size : int
        The size of the batches.
    num_workers : int
        The number of worker threads to use for data loading.
    seed : int
        The random seed to split the train dataset into validation and training samples.
    alpha : float
        The proportion of the dataset to use for training.
    length : int
        The total number of samples in the dataset.
    binarize_labels : bool
        Whether to binarize the labels.

    Returns
    -------
    tuple
        A tuple containing the data loaders, training indices, validation indices, and the start time.
    """
    # adapted from https://github.com/libffcv/ffcv/blob/main/examples/cifar/train_cifar.py

    if get_val_samples:
        paths = {
            'train': train_dataset,
            'val': train_dataset,
            'test': test_dataset
        }
    else:
        paths = {
            'train': train_dataset,
            'test': test_dataset
        }

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    # Get indices for training and validation splits
    train_indices = get_val_split_indices(seed, length, alpha, get_val_samples)

    if get_val_samples:
        indices = {
            'train': train_indices[0],
            'val': train_indices[1],
            'test': None  # or don't make this None but have all indices
        }
        names = ['train', 'val', 'test']

    else:
        indices = {
            'train': train_indices,
            'test': None
        }
        names = ['train', 'test']


    if test_batch_size is not None:
        batch_sizes = {
            'train': batch_size,
            'val': batch_size, 
            'test': test_batch_size
        }
    else:
        batch_sizes = {
            'train': batch_size,
            'val': batch_size, 
            'test': batch_size
            }
        
    # Create data loaders for train, val, and test sets
    for name in names:
    
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice(ch.device('cuda:0')),
            Squeeze()
        ]

        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        if no_transform:
            image_pipeline.extend([
                ToTensor(),
                ToDevice(ch.device('cuda:0'), non_blocking=True),
                ToTorchImage(),
                Convert(ch.float16)
            ])
        else:
            if name == 'train':
                image_pipeline.extend([
                    RandomHorizontalFlip(),
                    RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                    Cutout(4, tuple(map(int, CIFAR_MEAN))),
                ])

            image_pipeline.extend([
                ToTensor(),
                ToDevice(ch.device('cuda:0'), non_blocking=True),
                ToTorchImage(),
                Convert(ch.float16),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])

        if return_sequential:
            ordering = OrderOption.SEQUENTIAL
        else:
            ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(
            paths[name],
            batch_size=batch_sizes[name],
            os_cache=True,
            num_workers=num_workers,
            order=ordering,
            drop_last=(name == 'train'),
            indices=indices[name],
            pipelines={'image': image_pipeline, 'label': label_pipeline}
        )

    return loaders, train_indices
