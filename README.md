# <code>datamodels</code> Tutorial: 
## Identifying Subgroups in Biomedical Datasets using Data Attribution 

Understanding how training data influences model predictions ("data attribution") is an active area of machine learning research. In this tutorial, we will introduce a data attribution method (datamodels: https://gradientscience.org/datamodels-1/) and explore how it can be applied in the life sciences  to identify meaningful subgroups in biomedical datasets, such as disease subtypes. We will begin with a simple example from image classification (CIFAR10), offering a step-by-step guide to demonstrate how the data attribution method works in practice. Since the approach involves training thousands of lightweight classifiers, we will focus on strategies for fast and efficient model training. Next, we will explore its applications in biomedical science, with a focus on single-cell and genetic datasets, highlighting the biological insights gained from applying this computational approach. The tutorial will conclude with an interactive, hands-on session using Google Colab, where participants can apply the techniques themselves and explore the approach further. This session is designed to be accessible to participants of all coding and machine learning experience levels—whether you're new to machine learning or curious about its intersection with biomedical applications.

## Tutorial Materials
- [recording](https://cbmm.mit.edu/computational-tutorials/recordings)
- [slides](https://drive.google.com/file/d/1qGahNYBUnThba07D2D9gZTviiU_kOedF/view?usp=sharing)
- [google collab; code no outputs](https://colab.research.google.com/drive/1lwl7-Xsc7lg9bTg97hEEqPt54x-J1qeU?usp=sharing)
- [google collab; code with outputs](https://colab.research.google.com/drive/1u2jZzWs7SVT6kj-O8rMsUphHfvyeqnHh?usp=sharing)

## This Repository
This repository contains code to reproduce the example experiment given in the tutorial (i.e. `datamodels.pt`).

```{bash}
conda env create -f environment.yml --name ffcv
conda activate ffcv

pip install tqdm ffcv pyyaml fastargs ray torchvision fast_l1 notebook matplotlib
pip install "ray[tune]"

# install fastl1 from https://github.com/MadryLab/fast_l1

# optionally install ipykernel to use the notebook interface
conda install ipykernel
python -m ipykernel install --user --name=ffcv
```
### 1. Subset the CIFAR10 dataset
```{bash}
conda activate ffcv
python write_datasets.py --data.train_dataset ./CIFAR10/cifar10_train_subset_binaryLabels.beton \
                         --data.val_dataset ./CIFAR10/cifar10_val_subset_binaryLabels.beton \
                         --data.binary_labels True \
                         --data.subset_indices 25000 # subset the training set to 25k samples
```

> ### 2. Steps for Model Training and Tuning
> 
> **Optional** ***if you plan on using the same dataset and alpha as in the example, move on to step 3***
> 1. **Inspect the Dataloader (Optional)**  
>    Before starting training, you can inspect the dataloader by running the following notebook:
>    
>    `inspect_dataloader.ipynb`
> 
> 2. **Verify Training**  
>    Ensure that the model training is functioning correctly by running the training notebook:
>    
>    `train_a_good_model.ipynb`
> 
> 3. **Parameter Tuning for Alpha**  
>    To fine-tune your model parameters for a specific alpha value, use the notebook:
>    
>    `train_a_better_model.ipynb`
>
>    Make sure you have the wandb library installed (`pip install wandb`)
>

### 3. Train many models on different subsets of the dataset
```{bash}
conda activate ffcv
sbatch launch_headnode.sh
sbatch train_cifar_with_ray.sh # once the headnode is running, update the address in this file before submitting the training jobs (found in the .out file for the launch_headnode job) 
```

### 4. Fit datamodels
```{bash}
conda activate ffcv
sbatch train_datamodels.sh
```

### 5. Explore the datamodels embeddings

- [google collab; code no outputs](https://colab.research.google.com/drive/1lwl7-Xsc7lg9bTg97hEEqPt54x-J1qeU?usp=sharing)

## System Requirements
### Python libraries
- **tqdm**: 4.66.5
- **ffcv**: 1.0.2
- **pyyaml**: 6.0.2
- **fastargs**: 1.2.0
- **ray**: 2.37.0
- **torchvision**: 0.19.0+cu118
### Hardware
- **GPU**: NVIDIA Tesla V100-PCIE-32GB
  - **Memory**: 32 GB
  - **CUDA Capability**: Required for running GPU-accelerated tasks.
### Software
- **NVIDIA Driver Version**: 535.183.01
- **CUDA Version**: 12.2
### Notes
- Update the sbatch files according to your resource availability
- You can adjust the number of simultaneous ray trials by modifying `cpus_per_trial` and `gpus_per_trial` parameters in the `config` file
- Profile your GPU and CPU usage by running `nvidia-smi -l` or `htop`, respectively on your compute node
