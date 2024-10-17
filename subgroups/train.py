"""
This module provides functions to perform a full training and evaluation iteration using FFCV, and to train a PyTorch model using a given data loader and training parameters.

Functions:
    full_iteration_ffcv: Perform a full training and evaluation iteration using FFCV.
    train: Train a given model using the provided data loader and training parameters.

Note:
    The functions `train` were adapted from https://github.com/MadryLab/trak
"""
from subgroups.dataloading_ffcv import make_dataloaders
from subgroups.resnet9 import construct_rn9

import torch

from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from torch.optim import Adam

from tqdm import tqdm

from torch.optim.lr_scheduler import ExponentialLR, StepLR
from fastargs import Param, Section
from fastargs.decorators import param


# Define the training hyperparameters section
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
)

# Define the data-related parameters section
Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', default='/home/gridsan/djuna/TsaiMadry_shared/datamodels_clustering/CIFAR10/cifar10_train_subset_binaryLabels.beton'),
    test_dataset=Param(str, '.dat file to use for validation', default='/home/gridsan/djuna/TsaiMadry_shared/datamodels_clustering/CIFAR10/cifar10_val_subset_binaryLabels.beton'),
    binarize_labels=Param(bool, 'Whether to binarize labels', default=True),
    alpha=Param(float, 'Proportion of the dataset to use for training', default=0.1),
    length=Param(int, 'Length of the dataset', default=25000),
    random_alpha=Param(bool, 'Whether to randomize alpha', default=False),
    get_val_samples=Param(bool, 'Whether to get validation samples', default=False),
    no_transform=Param(bool, 'Whether to not transform the data', default=False),
    return_sequential=Param(bool, 'Whether to return the data in sequential order', default=False),
)

# Decorate the function with parameters from the defined sections
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
@param('data.get_val_samples')
@param('data.no_transform')
@param('data.return_sequential')
@param('training.test_batch_size')
@param('training.optimizer')
@param('training.gamma')
@param('training.step_size')
@param('training.lr_scheduler')
@param('training.lr_tta')
def full_iteration_ffcv(train_dataset: str = None, test_dataset: str = None, batch_size: int = None, num_workers: int = None, 
                        seed: int = None, alpha: float = None, num_classes: int = None, lr: float = None, epochs: int = None, 
                        momentum: float = None, weight_decay: float = None, label_smoothing: float = None, 
                        length: int = None, get_val_samples: bool = None, no_transform: bool = None,  
                        return_sequential: bool = None, test_batch_size: int = None, optimizer: str = None, gamma: float = None, 
                        step_size: float = None, lr_scheduler: str = None, lr_tta: bool = None) -> tuple:
    
    loaders, _ = make_dataloaders(
        train_dataset=train_dataset, test_dataset=test_dataset, batch_size=batch_size,
        num_workers=num_workers, seed=seed, alpha=alpha, length=length, test_batch_size=test_batch_size, get_val_samples=get_val_samples, no_transform=no_transform, return_sequential=return_sequential,
    )  
        
    model = construct_rn9(num_classes=num_classes).to(memory_format=torch.channels_last).cuda()
    
    average_acc, all_margins = train(
        model, loaders, lr=lr, epochs=epochs, momentum=momentum,
        weight_decay=weight_decay, label_smoothing=label_smoothing,
        optimizer=optimizer, gamma=gamma, step_size=step_size, 
        lr_scheduler=lr_scheduler, lr_tta=lr_tta
    )

    return [average_acc, all_margins, loaders['train'].indices]

def eval_test(model, loader, lr_tta: bool = False):
    model.eval()
    with torch.no_grad():
        temp_acc, N = 0.0, 0.0
        all_margins = []
        for it, (ims, labs) in enumerate(loader['test']):
            with autocast('cuda'):
                out = model(ims)
                if lr_tta:
                    out += model(torch.fliplr(ims))
                    out /= 2

                pred = out.argmax(1).eq(labs)
                total_correct = pred.sum().cpu().item()
                temp_acc+=total_correct/ims.shape[0]
                N+=1

                class_logits = out[torch.arange(out.shape[0]), labs].clone()
                out[torch.arange(out.shape[0]), labs] = -1000
                next_classes = out.argmax(1)
                class_logits -= out[torch.arange(out.shape[0]), next_classes]
                all_margins.append(class_logits.cpu())

    all_margins = torch.cat(all_margins)                
    average_acc = temp_acc/N
    return average_acc, all_margins


def train(model: torch.nn.Module, loader: dict = None, lr: float = None, epochs: int = None, momentum: float = None,
          weight_decay: float = None, label_smoothing: float = None, optimizer: str = None, gamma: float = None, 
          step_size: float = None, lr_scheduler: str = None, lr_tta: bool = None) -> list:
    
    if optimizer=='SGD':
        opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer=='adam':
        opt = Adam(model.parameters(), lr=lr)

    if lr_scheduler=='exponential':
        scheduler = ExponentialLR(opt, gamma=gamma)
    elif lr_scheduler=='step':
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

    scaler = GradScaler('cuda')
    
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    model.train()

    for ep in tqdm(range(epochs)):
        
        for it, (ims, labs) in enumerate(loader['train']):
            opt.zero_grad(set_to_none=True)

            with autocast('cuda'):
                out = model(ims)
                loss = loss_fn(out, labs)
            
            #unique_labels = torch.unique(labs)
            #print(f"Unique labels in batch: {unique_labels}")
            #assert torch.all((labs >= 0) & (labs < 2))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        scheduler.step()
    
    average_acc, all_margins = eval_test(model, loader, lr_tta=lr_tta)
    return average_acc, all_margins