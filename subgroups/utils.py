"""
This module provides utility functions for saving and loading model outputs.

Functions:
    create_output_arrays: Create output arrays for training and evaluation results.
    save_model_outputs: Save the model outputs to the respective output arrays.
"""

import os
import numpy as np
from os import environ
import yaml
from typing import Optional, Dict, Any
import torch as ch

from fastargs import Param, Section
from fastargs.decorators import param

# Define the output-related parameters section
Section('output', 'output related stuff').params(
    directory=Param(str, 'Directory to save output arrays', default='/tmp/cifar_train.beton'),
    nmodels=Param(int, 'Number of models to be trained', default=100),
    models_per_job=Param(int, 'Number of models to be trained per job', default=10),
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', default='/CIFAR10/cifar_train.beton'),
    test_dataset=Param(str, '.dat file to use for validation', default='/CIFAR10/cifar_test.beton'),
    binarize_labels=Param(bool, 'Whether to binarize labels', default=True),
    alpha=Param(float, 'Proportion of the dataset to use for training', default=0.5),
    length=Param(int, 'Length of the dataset', default=50000),
    val_length=Param(int, 'Length of the validation dataset', default=10000),
    random_alpha=Param(bool, 'Whether to randomize alpha', default=False),
)



def save_model_outputs(model_ID, model_outputs, output_arrays):
    """
    Save the model outputs to the respective output arrays.

    Parameters
    ----------
    model_ID : int
        The ID of the model.
    train_samples : array-like
        Indices of the training samples.
    val_samples : array-like
        Indices of the validation samples.
    val_probs : array-like
        Validation probabilities.
    test_probs : array-like
        Test probabilities.
    val_true_labs : array-like
        True labels for validation set.
    test_true_labs : array-like
        True labels for test set.
    val_acc : float
        Validation accuracy.
    test_acc : float
        Test accuracy.
    output_arrays : list
        List of output arrays created by create_output_arrays function.
    """
    test_acc, test_margins, train_samples = model_outputs
    masks_train, margins_test, acc_out_test, model_done = output_arrays

    masks_train[model_ID][train_samples] = 1
    margins_test[model_ID] = test_margins
    acc_out_test[model_ID] = test_acc
    model_done[model_ID] = True

@param('output.models_per_job')
def get_model_range(models_per_job: int) -> tuple:
    """
    Get the start and end indices for model training based on the SLURM_ARRAY_TASK_ID environment variable.

    Parameters
    ----------
    models_per_job : int
        The number of models to train per job.

    Returns
    -------
    tuple
        A tuple containing the start and end indices for model training.
    """
    if 'SLURM_ARRAY_TASK_ID' in environ:
        start = models_per_job * int(environ['SLURM_ARRAY_TASK_ID'])
    else:
        start = 0
    end = start + models_per_job
    return start, end


def read_yaml_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Read and parse a YAML file.

    This function opens a YAML file at the specified path, attempts to parse its contents,
    and returns the parsed data as a Python dictionary. If an error occurs during parsing,
    it prints an error message and returns None.

    Parameters
    ----------
    file_path : str
        The path to the YAML file to be read.

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary containing the parsed YAML data if successful, or None if an error occurred.

    Raises
    ------
    FileNotFoundError
        If the specified file_path does not exist.
    PermissionError
        If the user does not have permission to read the file.
    """
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error opening YAML file: {e}")
        return None



def load_datamodels(file_path: str):
    """
    Load the datamodels from the specified file.

    Parameters
    ----------
    file_path : str
        Path to the file to load the datamodels from.

    Returnsju
    -------
    dict
        A dictionary containing the loaded datamodels.
    """
    # Load the saved file
    data = ch.load(file_path)

    # Access the contents
    weight = data['weight']
    bias = data['bias']
    lam = data['lam']

    # Import the ResNet9 model from the resnet9 module
    from subgroups.resnet9 import construct_rn9

    # Construct the ResNet9 model
    model = construct_rn9()

    return weight, bias, lam, model