# from https://github.com/libffcv/ffcv/blob/main/examples/cifar/write_datasets.py

from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.utils.data import Subset
import torchvision

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

Section('data', 'arguments to give the writer').params(
    train_dataset=Param(str, 'Where to write the new dataset', required=True),
    val_dataset=Param(str, 'Where to write the new dataset', required=True),
    subset_indices=Param(int, 'Number of indices to use for the train subset', required=False),
    binary_labels=Param(bool, 'Whether to binarize the labels', required=False),
    subset_val=Param(bool, 'Whether to subset the validation set', required=False),
)

def get_binary_labels(dataset, animate_labels: set = {2, 3, 4, 5, 6, 7}) -> List[bool]:
    binary_targets = [label in animate_labels for label in dataset.targets]
    return binary_targets

@param('data.train_dataset')
@param('data.val_dataset')
@param('data.subset_indices')
@param('data.binary_labels')
@param('data.subset_val')
def main(train_dataset, val_dataset, binary_labels=False, subset_indices=None, subset_val=False):
    datasets = {
        'train': torchvision.datasets.CIFAR10('./CIFAR10', train=True, download=True),
        'test': torchvision.datasets.CIFAR10('./CIFAR10', train=False, download=True)
        }

    for (name, ds) in datasets.items():
        if binary_labels:
            ds.targets = get_binary_labels(ds)
        if (subset_indices is not None) & (name == 'train'):
            ds = Subset(ds, range(subset_indices))
        if (subset_indices is not None) & (subset_val) & (name == 'test'):
            ds = Subset(ds, range(subset_indices))
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()