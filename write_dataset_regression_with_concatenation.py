import numba as nb
import numpy as np
import os
from fastargs import get_current_config
from fastargs import Param, Section
from fastargs.decorators import param
from argparse import ArgumentParser
from tqdm import tqdm
from subgroups.write_dataset_regression import write_dataset


Section('config', 'Config').params(
    input_directory=Param(str, 'Location of the trial outputs'),
    output_directory=Param(str, 'Directory to save concatenated output arrays', required=True),
    nmodels=Param(int, 'Number of models trained'),
    length=Param(int, 'Length of the training dataset (Number of features in the mask matrix)', default=25000),
    val_length=Param(int, 'Length of the validation dataset (Number of columns in the margins matrix)', default=10000),
    prefix=Param(str, 'Prefix for the trial outpus', default='trial_'),
    x_name=Param(str, 'Name of the x matrix', default='masks_train', required=True),
    y_name=Param(str, 'Name of the y matrix', default='margins_test', required=True),
    ntrained_models=Param(int, 'Number of models trained', default=None)
)

def create_output_arrays(directory, nmodels, length, val_length):
    if not os.path.exists(directory):
        os.makedirs(directory)
    mode = 'w+'
    masks = np.lib.format.open_memmap(directory + '/masks.npy', dtype=bool, mode=mode, shape=(nmodels, length))
    margins = np.lib.format.open_memmap(directory + '/margins.npy', dtype=float, mode=mode, shape=(nmodels, val_length))
    #labels = np.lib.format.open_memmap(directory + '/labels.npy', dtype=bool, mode=mode, shape=(nmodels, val_length))
    accuracies = np.lib.format.open_memmap(directory + '/accuracies.npy', dtype=float, mode=mode, shape=(nmodels, ))
    return [masks, margins, accuracies]

def open_memmap(input_directory):
    mode = 'r'
    masks = np.lib.format.open_memmap(input_directory + '/masks_train.npy', mode=mode)
    margins = np.lib.format.open_memmap(input_directory + '/margins_test.npy', mode=mode)
    #labels = np.lib.format.open_memmap(input_directory + '/labs_test.npy', mode=mode)
    accuracies = np.lib.format.open_memmap(input_directory + '/acc_out_test.npy', mode=mode)
    model_done = np.lib.format.open_memmap(input_directory + '/model_done.npy', mode=mode)
    return [masks, margins, accuracies, model_done]

def get_folders_with_prefix(directory: str, prefix: str) -> list:
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder)) and folder.startswith(prefix)]

@nb.njit()
def write_trial_rows(models_done, temp, start, matrix_out, number_done):
    if number_done==len(models_done):
        write_rows(temp, matrix_out, start) # 0.53 s
    else:
        write_rows(temp, matrix_out, start, models_done) # 0.53 s

@nb.njit()
def write_rows(matrix, output_array, start, models_done=None):
    if models_done is None:
        for i in nb.prange(matrix.shape[0]):
            output_array[start+i] = matrix[i]
    else:
        output_index = start
        for i in nb.prange(matrix.shape[0]):
            if models_done[i]:
                output_array[output_index] = matrix[i]
                output_index += 1
                #print(output_index)
            else:
                continue
        

def write_matrix(input_directory, matrix_index, trials, matrix_out, matrix_name):
    start = 0
    for trial in tqdm(trials, desc='Concatenating training outputs for ' + matrix_name):
        dir = input_directory + '/' + trial 
        input_matrices = open_memmap(dir)
        models_done = input_matrices[3]
        temp = input_matrices[matrix_index]
        number_done = np.sum(models_done)
        write_trial_rows(models_done, temp, start, matrix_out, number_done)
        start = number_done + start
    if matrix_name == 'margins_test.npy':
        print('In total, ' + str(start) + ' models were trained')
    return start


def concatenate_dataset(input_directory, output_directory, nmodels, length, val_length, prefix):
    input_name = ['masks_train.npy', 'margins_test.npy', 'accuracies_test.npy']
    trials = get_folders_with_prefix(input_directory, prefix)
    output_arrays = create_output_arrays(output_directory, nmodels, length, val_length)
    for i, matrix in enumerate(output_arrays):
        start = write_matrix(input_directory, i, trials, matrix, input_name[i])
    return start

def create_mock_dataset(output_directory, ntrials=3, ntrain=10, ntest=5, nmodels=3):
    np.random.seed(42)  # Set the seed for reproducibility
    if not os.path.exists(output_directory):
        os.makedirs(output_directory) 
    for trial in range(ntrials):
        trial_dir = os.path.join(output_directory, f'trial_{trial}')
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)
        # Create mock training data
        masks_train = np.random.rand(nmodels, ntrain)
        margins_test = np.random.rand(nmodels, ntest)
        accuracies_test = np.random.rand(nmodels,)
        models_done = np.random.randint(0, 2, size=(nmodels,))
        masks_train_memmap = np.lib.format.open_memmap(os.path.join(trial_dir, 'masks_train.npy'), dtype=bool, mode='w+', shape=masks_train.shape)
        masks_train_memmap[:] = masks_train[:]
        del masks_train_memmap
        margins_test_memmap = np.lib.format.open_memmap(os.path.join(trial_dir, 'margins_test.npy'), dtype=float, mode='w+', shape=margins_test.shape)
        margins_test_memmap[:] = margins_test[:]
        del margins_test_memmap
        #labels_test_memmap = np.lib.format.open_memmap(os.path.join(trial_dir, 'labs_test.npy'), dtype=bool, mode='w+', shape=labels_test.shape)
        #labels_test_memmap[:] = labels_test[:]
        #del labels_test_memmap
        accuracies_test_memmap = np.lib.format.open_memmap(os.path.join(trial_dir, 'acc_out_test.npy'), dtype=float, mode='w+', shape=accuracies_test.shape)
        accuracies_test_memmap[:] = accuracies_test[:]
        del accuracies_test_memmap
        models_done_memmap = np.lib.format.open_memmap(os.path.join(trial_dir, 'model_done.npy'), dtype=bool, mode='w+', shape=models_done.shape)
        models_done_memmap[:] = models_done[:]
        del models_done_memmap
    print(f'Mock dataset created at {output_directory}')


@param('config.input_directory')
@param('config.output_directory')
@param('config.nmodels')
@param('config.length')
@param('config.val_length')
@param('config.prefix')
@param('config.x_name')
@param('config.y_name')
@param('config.ntrained_models')
def main(input_directory, output_directory, nmodels, length, val_length, prefix, x_name, y_name, ntrained_models=None):

    if not os.path.exists(output_directory + '/masks.npy'):
        print("Concatenating training outputs")
        ntrained_models = concatenate_dataset(input_directory, output_directory, nmodels, length, val_length, prefix)
        print(ntrained_models, 'models were trained')
    print("Making .beton file")
    write_dataset(data_dir=output_directory, out_path=output_directory + '/datamodels_input.beton', x_name=x_name, y_name=y_name, x_slice=ntrained_models, ignore_completed=True, completed_name=None, y_slice=-1)
    # add an x slice option to the write dataset function



if __name__ == '__main__':
    print('\033[4m1. getting config\033[0m')
    param_config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    param_config.augment_argparse(parser)
    param_config.collect_argparse_args(parser)
    param_config.validate(mode='stderr')
    param_config.summary()
    main()