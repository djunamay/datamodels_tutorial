# <code>datamodels</code> Tutorial: 
## Identifying Subgroups in Biomedical Datasets using Data Attribution 

Understanding how training data influences model predictions ("data attribution") is an active area of machine learning research. In this tutorial, we will introduce a data attribution method (datamodels: https://gradientscience.org/datamodels-1/) and explore how it can be applied in the life sciences  to identify meaningful subgroups in biomedical datasets, such as disease subtypes. We will begin with a simple example from image classification (CIFAR10), offering a step-by-step guide to demonstrate how the data attribution method works in practice. Since the approach involves training thousands of lightweight classifiers, we will focus on strategies for fast and efficient model training. Next, we will explore its applications in biomedical science, with a focus on single-cell and genetic datasets, highlighting the biological insights gained from applying this computational approach. The tutorial will conclude with an interactive, hands-on session using Google Colab, where participants can apply the techniques themselves and explore the approach further. This session is designed to be accessible to participants of all coding and machine learning experience levels—whether you're new to machine learning or curious about its intersection with biomedical applications.

## Tutorial Materials
- [recording](https://cbmm.mit.edu/computational-tutorials/recordings)
- [slides](https://drive.google.com/file/d/1qGahNYBUnThba07D2D9gZTviiU_kOedF/view?usp=sharing)
- [google collab; code no outputs](https://colab.research.google.com/drive/1lwl7-Xsc7lg9bTg97hEEqPt54x-J1qeU?usp=sharing)
- [google collab; code with outputs](https://colab.research.google.com/drive/1u2jZzWs7SVT6kj-O8rMsUphHfvyeqnHh?usp=sharing)
- [notes]()

## This Repository
This repository contains code to reproduce the exammple experiment in the tutorial; i.e. the code necessary to get to the 'datamodels.pt' output used in the google collab example.

```{bash}
conda env create -f environment.yml --name ffcv
conda activate ffcv
pip instal ... # version xx

```
1. Subset the CIFAR10 dataset
```{bash}
conda activate ffcv
python write_datasets.py --data.train_dataset ../tutorial/toy_dataset/cifar10_toy_train.beton \
                         --data.val_dataset ../tutorial/toy_dataset/cifar10_toy_val.beton \
                         --data.binary_labels True \
                         --data.subset_indices 1000 \
                         --data.subset_val True
```
[Optional]
2. Inspect the dataloader
`inspect_dataloader.ipynb`

3. Check that the training works as expected
`train_a_good_model.ipynb`

4. Tune your parameters for a given alpha
`train_a_better_model.ipynb`

## Requirements
