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

Experiment

In this experiment, we trained circa 60,000 classifiers on random subsets of the CIFAR10 training set (alpha = 0.1) on the task to predict animate vs inanimate objects. This is a comparatively simple task compared to learning to predict all 10 classes (dogs, airplanes, e.t.c ...) and allows us to investigate model class behavior with some information on known sub-classes in the dataset for which models will likely have learnt different patterns (when trained on random subsets of the data) to solve this task.
