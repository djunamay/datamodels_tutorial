# from https://github.com/MadryLab/datamodels/blob/main/datamodels/regression/write_dataset.py

import numpy as np
import os
from typing import Optional, Sequence
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.decorators import param
from fastargs import get_current_config

Section('cfg', 'arguments to give the writer').params(
    data_dir=Param(str, 'Where to find the mmap datasets', required=False),
    out_path=Param(str, 'Where to write the FFCV dataset', required=False),
    x_name=Param(str, 'What portion of the data to write', required=False),
    y_name=Param(str, 'What portion of the data to write', required=False),
    y_slice=Param(int, 'If given, take a target slice', required=False),
    x_slice=Param(int, 'If given, take a target slice', required=False),
    completed_name=Param(str, 'Where to find completed datamodels', required=False),
    ignore_completed=Param(bool, 'Whether to ignore the completed array', is_flag=True, required=False)
)


class RegressionDataset(Dataset):
    def __init__(self, *, masks_path: str, y_path: str,
                 completed_path: str,
                 subset: Optional[Sequence[int]] = None,
                 y_slice: Optional[int] = None,
                 x_slice: Optional[int] = None,
                 ignore_completed: bool = False):
        super().__init__()
        self.comp_inds = None
        if not ignore_completed:
            comp_fp = np.lib.format.open_memmap(completed_path, mode='r')
            self.comp_inds = np.nonzero(comp_fp)[0]
        self.masks_fp = np.lib.format.open_memmap(masks_path, mode='r')
        self.x_dtype = self.masks_fp.dtype
        self.y_vals_fp = np.lib.format.open_memmap(y_path, mode='r')
        if y_slice is not None and y_slice > -1:
            self.y_vals_fp = self.y_vals_fp[:, y_slice]
        if x_slice is not None and x_slice > -1:
            self.masks_fp = self.masks_fp[:x_slice]
            self.y_vals_fp = self.y_vals_fp[:x_slice]
        self.y_dtype = np.dtype('float32')

        total_len = len(self.masks_fp if self.comp_inds is None else self.comp_inds)
        self.subset = subset or range(total_len)

    def __getitem__(self, idx):
        ind = self.subset[idx]
        if self.comp_inds is not None:
            ind = self.comp_inds[ind]
        x_val = self.masks_fp[ind]
        y_val = self.y_vals_fp[ind].astype('float32')
        return x_val, y_val, ind

    def shape(self):
        return self.masks_fp.shape[1], self.y_vals_fp.shape[1]

    def __len__(self):
        return len(self.subset)


@param('cfg.data_dir')
@param('cfg.out_path')
@param('cfg.x_name')
@param('cfg.y_name')
@param('cfg.completed_name')
@param('cfg.y_slice')
@param('cfg.x_slice')
@param('cfg.ignore_completed')
def write_dataset(data_dir: str, out_path: str,
                  x_name: str, y_name: str,
                  completed_name: str, y_slice: int, x_slice: int, ignore_completed: bool):
    ds = RegressionDataset(
            completed_path=os.path.join(data_dir, f'{completed_name}.npy'),
            masks_path=os.path.join(data_dir, f'{x_name}.npy'),
            y_path=os.path.join(data_dir, f'{y_name}.npy'),
            y_slice=y_slice, x_slice=x_slice, ignore_completed=ignore_completed)

    x_dim, y_dim = ds.shape()
    writer = DatasetWriter(out_path, {
        'mask': NDArrayField(dtype=ds.x_dtype, shape=(x_dim,)),
        'targets': NDArrayField(dtype=ds.y_dtype, shape=(y_dim,)),
        'idx': IntField()
    })

    writer.from_indexed_dataset(ds)


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    write_dataset()