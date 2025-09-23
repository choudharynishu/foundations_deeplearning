import os
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split


import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from config.config import settings
from src.model import BaseNetwork


def plot_distributions(input_dict:dict, xlabel:str|None=None,
                       stat='count', use_kde:bool=True, print_variance:bool=True):
    #ncols = 3
    #nrows = len(input_dict)//ncols+1
    ncols = len(input_dict)
    nrows = 2 if use_kde else 1
    curr_col = 1
    kde_titles = [f"KDE_{key}" for key in input_dict.keys()]
    subplot_title = list(input_dict.keys())+kde_titles if use_kde\
        else list(input_dict.keys())

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_title)

    for key, values in input_dict.items():

        fig.add_trace(go.Histogram(x=values, name=f"{key}"), row=1, col=curr_col)

        if use_kde:
            kde_kernels = gaussian_kde(values)
            x_grid = np.linspace(min(values), max(values), 300)
            kde_values = kde_kernels(x_grid)
            fig.add_trace(go.Scatter(x=x_grid, y=kde_values, name=f"KDE - {key}"), row=2, col=curr_col)

        curr_col+=1

    if print_variance:
        for key in sorted(input_dict.keys()):
            print(f"{key} - Variance: {np.var(input_dict[key])}")
    #fig.update_layout(yaxis=dict(range=[0,8]))
    return fig

def visualize_gradients(model:BaseNetwork, train_dataset:Dataset,
                        init_technique:str):

    model.eval()
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    images, labels = next(iter(train_loader))
    pred = model(images)

    loss = nn.CrossEntropyLoss()
    model.zero_grad()

    loss_value = loss(pred, labels)
    loss_value.backward()

    gradients = {name:parameter.grad.view(-1).numpy() for name, parameter in
                 model.named_parameters() if not 'bias' in name}
    fig = plot_distributions(gradients)
    fig.update_layout(title=f"Histogram and Kernel Density Estimation"
                            f" of Gradients using {init_technique} Initialization")
    fig.write_html(os.path.join(settings.data_dir, f"Gradients_{init_technique}.html"))


def visualize_preactivations(model:BaseNetwork, train_dataset:Dataset,
                             init_technique:str):

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    images, labels = next(iter(train_loader))
    features = images.view(images.size(0), -1)

    pre_activations = {}
    for index, layer in enumerate(model.layers):
        features = layer(features)
        if isinstance(layer, nn.Linear):
            pre_activations[f"layer_{index}"] = features.detach().view(-1).numpy()

    fig = plot_distributions(pre_activations)
    fig.update_layout(title=f"Histogram and Kernel Density Estimation Plots"
                            f" for Pre-Activation Values using {init_technique} Initialization")
    fig.write_html(os.path.join(settings.data_dir, f"PreActivations_{init_technique}.html"))


def visualize_activations(model:BaseNetwork, train_dataset:Dataset,
                          init_technique:str):
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    images, labels = next(iter(train_loader))
    features = images.view(images.size(0), -1)

    activations = {}
    for index, layer in enumerate(model.layers):
        features = layer(features)
        if not isinstance(layer, nn.Linear):
            activations[f"layer_{index}"] = features.detach().view(-1).numpy()

    fig = plot_distributions(activations)
    fig.update_layout(title=f"Histogram and Kernel Density Estimation Plots"
                            f" for Activation Outputs using {init_technique} Initialization")
    fig.write_html(os.path.join(settings.data_dir, f"Activations_{init_technique}.html"))


def preprocess_data():
    train_data, val_data, test_data = download_data()
    train_loader = create_dataloader(train_data, train=True)
    val_loader = create_dataloader(val_data, train=False)
    test_loader = create_dataloader(test_data, train=False)

    return train_loader, val_loader, test_loader


def download_data(train_val_split: float = 0.8):
    transform_list = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])

    train_data = datasets.FashionMNIST(root=settings.data_dir,
                                       train=True,
                                       transform=transform_list,
                                       download=True)

    test_data = datasets.FashionMNIST(root=settings.data_dir,
                                      train=False,
                                      transform=transform_list,
                                      download=True)
    train_data_len = int(len(train_data) * train_val_split)
    val_data_len = len(train_data) - train_data_len
    split_sequence = [train_data_len, val_data_len]

    train_dataset, validation_dataset = random_split(train_data, split_sequence)

    return (train_dataset, validation_dataset, test_data)


def create_dataloader(data: Dataset, train: bool = True,
                      batchsize: int | None = None):
    """
        Create a DataLoader for the given dataset.

        This function wraps a PyTorch dataset into a DataLoader, which
        allows batching, shuffling, and parallel data loading.

        Parameters
        ----------
        data : torch.utils.data.Dataset
            The dataset to wrap into a DataLoader.
        train : bool, optional
            If True, shuffle the data. Default is True.
        batchsize : int or None, optional
            The number of samples per batch. If None, defaults to
            ``settings.batchsize``.

        Returns
        -------
        torch.utils.data.DataLoader
            The DataLoader object for the dataset.
    """
    if batchsize is None:
        batchsize = settings.batchsize

    dataloader = DataLoader(
        data,
        batch_size=batchsize,
        shuffle=train,
        drop_last=False
    )
    return dataloader
