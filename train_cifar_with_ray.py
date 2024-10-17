"""
This script trains a specified number of models on the CIFAR-10 dataset using FFCV and RESNET9 on a user-specified subset of the training data, parameterized by alpha.
"""
print('\033[4m0. importing packages\033[0m')
import os
import numpy as np
import time
from tqdm.auto import tqdm
from os import environ
from argparse import ArgumentParser

from subgroups.train import full_iteration_ffcv
from subgroups.utils import save_model_outputs 

from fastargs import get_current_config
from fastargs import Param, Section
from fastargs.decorators import param

from ray import tune
import ray

# Define the training hyperparameters section
Section('resources', 'Resources').params(
    cpus_per_trial=Param(float, 'Number of CPUs per trial', default=2),
    gpus_per_trial=Param(float, 'Number of GPUs per trial', default=0.3),
    num_samples=Param(int, 'Number of samples', default=100),
    max_concurrent_trials=Param(int, 'Maximum number of concurrent trials', default=2),
)

Section('output', 'output related stuff').params(
    directory=Param(str, 'Directory to save output arrays', default='/tmp/cifar_train.beton'),
    nmodels_per_trial=Param(int, 'Number of models to be trained', default=10),
    Ntrials=Param(int, 'Number of trials', default=80),
    lowest_trial=Param(int, 'Lowest trial', default=0),
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', default='/home/gridsan/djuna/TsaiMadry_shared/datamodels_clustering/CIFAR10/cifar10_train_subset_binaryLabels.beton'),
    test_dataset=Param(str, '.dat file to use for validation', default='/home/gridsan/djuna/TsaiMadry_shared/datamodels_clustering/CIFAR10/cifar10_val_subset_binaryLabels.beton'),
    alpha=Param(float, 'Proportion of the dataset to use for training', default=0.1),
    length=Param(int, 'Length of the dataset', default=25000),
    get_val_samples=Param(bool, 'Whether to get validation samples', default=False),
    no_transform=Param(bool, 'Whether to not transform the data', default=False),
    return_sequential=Param(bool, 'Whether to return the data in sequential order', default=False),
)

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=0.4),
    epochs=Param(int, 'Number of epochs to run for', default=24),
    batch_size=Param(int, 'Batch size', default=1024),
    test_batch_size=Param(int, 'Test batch size', default=1024),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.0),
    num_workers=Param(int, 'The number of workers', default=1),
    num_classes=Param(int, 'The number of output classes', default=2),
    optimizer=Param(str, 'Optimizer to use', default='SGD'),
    gamma=Param(float, 'Gamma for ExponentialLR', default=0.2),
    step_size=Param(float, 'Step size for StepLR', default=7),
    lr_scheduler=Param(str, 'Learning rate scheduler', default='exponential'),
    lr_tta=Param(bool, 'Whether to use lr tta', default=False),
    get_val_samples=Param(bool, 'Whether to get validation samples', default=False),
    no_transform=Param(bool, 'Whether to not transform the data', default=False),
    return_sequential=Param(bool, 'Whether to return the data in sequential order', default=False),
)

def generate_random_seed(rng: np.random.Generator) -> int:
    """
    Generate a random seed using a provided NumPy Generator.
    
    Parameters
    ----------
    rng : np.random.Generator
        A NumPy Generator instance.
    
    Returns
    -------
    int
        A new random seed generated from the generator.
    """
    return rng.integers(0, 2**32 - 1)

def full_iteration_ffcv_with_ray(config: dict, 
                                 directory,
                                 nmodels,
                                 train_dataset,
                                 test_dataset,
                                 alpha,
                                 length,
                                 val_length,
                                 get_val_samples,
                                 no_transform,
                                 return_sequential,
                                 batch_size,
                                 test_batch_size,
                                 num_workers,
                                 num_classes,
                                 lr,
                                 epochs,
                                 momentum,
                                 weight_decay,
                                 label_smoothing,
                                 optimizer,
                                 gamma,
                                 step_size,
                                 lr_scheduler,
                                 lr_tta):
    # for each trial create output arrays named according to seed for that trial
    time.sleep(config["start_delay"])
    output_arrays = create_output_arrays(directory, config["seed"], nmodels, length, val_length)
    rng = np.random.default_rng(config["seed"]) # initialize the random number generator with a starting seed
    for model_id in range(nmodels):
        model_outputs = full_iteration_ffcv(seed=generate_random_seed(rng),
                                            # Data-related parameters
                                            train_dataset=train_dataset,
                                            test_dataset=test_dataset,
                                            alpha=alpha,
                                            length=length,
                                            get_val_samples=get_val_samples,
                                            no_transform=no_transform,
                                            return_sequential=return_sequential,
                                            batch_size=batch_size,
                                            test_batch_size=test_batch_size,
                                            num_workers=num_workers,
                                            # Training hyperparameters
                                            num_classes=num_classes,
                                            lr=lr,
                                            epochs=epochs,
                                            momentum=momentum,
                                            weight_decay=weight_decay,
                                            label_smoothing=label_smoothing,
                                            optimizer=optimizer,
                                            gamma=gamma,
                                            step_size=step_size,
                                            lr_scheduler=lr_scheduler,
                                            lr_tta=lr_tta)
        save_model_outputs(model_id, model_outputs, output_arrays)


def create_output_arrays(directory: str, seed: int, nmodels_per_trial: int, length: int, val_length: int) -> list:
    directory = directory + 'trial_' + str(seed)
    directories = [directory + '/masks_train.npy', directory + '/margins_test.npy', directory + '/acc_out_test.npy', directory + '/model_done.npy']
    # Check if directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    mode = 'w+'
    masks_train = np.lib.format.open_memmap(directories[0], dtype=bool, mode=mode, shape=(nmodels_per_trial, length))
    margins_test = np.lib.format.open_memmap(directories[1], dtype='float32', mode=mode, shape=(nmodels_per_trial, val_length))
    acc_out_test = np.lib.format.open_memmap(directories[2], dtype='float32', mode=mode, shape=(nmodels_per_trial,))
    model_done = np.lib.format.open_memmap(directories[3], dtype=bool, mode=mode, shape=(nmodels_per_trial,))
    output_arrays = [masks_train, margins_test, acc_out_test, model_done]
    return output_arrays


@param('resources.cpus_per_trial')
@param('resources.gpus_per_trial')
@param('resources.num_samples')
@param('resources.max_concurrent_trials')
@param('output.directory')
@param('output.nmodels_per_trial')
@param('output.Ntrials')
@param('output.lowest_trial')
@param('data.train_dataset')
@param('data.test_dataset')
@param('training.batch_size')
@param('training.num_workers')
@param('data.alpha')
@param('training.num_classes')
@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('data.length')
@param('data.val_length')
@param('training.get_val_samples')
@param('training.no_transform')
@param('training.return_sequential')
@param('training.test_batch_size')
@param('training.optimizer')
@param('training.gamma')
@param('training.step_size')
@param('training.lr_scheduler')
@param('training.lr_tta')
def train_models_with_ray(cpus_per_trial: float = None, 
                          gpus_per_trial: float = None, 
                          num_samples: int = None, 
                          max_concurrent_trials: int = None, 
                          train_dataset: str = None,
                          test_dataset: str = None,
                          length: int = None,
                          val_length: int = None,
                          alpha: float = None,
                          get_val_samples: bool = None,
                          no_transform: bool = None,
                          return_sequential: bool = None,
                          batch_size: int = None,
                          test_batch_size: int = None,
                          num_workers: int = None,
                          num_classes: int = None,
                          lr: float = None,
                          epochs: int = None,
                          momentum: float = None,
                          weight_decay: float = None,
                          label_smoothing: float = None,
                          optimizer: str = None,
                          gamma: float = None,
                          step_size: float = None,
                          lr_scheduler: str = None,
                          lr_tta: bool = None,
                          directory: str = None,
                          nmodels_per_trial: int = None,
                          Ntrials: int = None,
                          lowest_trial: int = None):
    seed_range = list(range(lowest_trial, Ntrials)) 
    config =     {
        "seed": tune.grid_search(seed_range),
        "start_delay": tune.loguniform(1, 25)  # Log-uniform distribution for start delay
    }


    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(full_iteration_ffcv_with_ray, 
                                 directory=directory,
                                 nmodels=nmodels_per_trial,
                                 train_dataset=train_dataset,
                                 test_dataset=test_dataset,
                                 length=length,
                                 val_length=val_length,
                                 alpha=alpha,
                                 get_val_samples=get_val_samples,
                                 no_transform=no_transform,
                                 return_sequential=return_sequential,
                                 batch_size=batch_size,
                                 test_batch_size=test_batch_size,
                                 num_workers=num_workers,
                                 num_classes=num_classes,
                                 lr=lr,
                                 epochs=epochs,
                                 momentum=momentum,
                                 weight_decay=weight_decay,
                                 label_smoothing=label_smoothing, 
                                 optimizer=optimizer,
                                 gamma=gamma,
                                 step_size=step_size,
                                 lr_scheduler=lr_scheduler,
                                 lr_tta=lr_tta),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        ), 
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials
        ),
        param_space=config,
    )
    print('tuner')
    print(tuner)
    print('fitting')
    results = tuner.fit()
    print('done fitting')


if __name__ == "__main__":
    # get start time

    # get parameter inputs
    print('\033[4m1. getting config\033[0m')
    param_config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    param_config.augment_argparse(parser)
    param_config.collect_argparse_args(parser)
    param_config.validate(mode='stderr')
    param_config.summary()
    
    ray.init(address='auto')

    # train models
    print('training models')
    start_train_time = time.time()
    train_models_with_ray()
    end_train_time = time.time()
    print(f'done, training took {end_train_time - start_train_time} seconds')

    ray.shutdown()