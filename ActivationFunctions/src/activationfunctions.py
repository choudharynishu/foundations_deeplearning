import os
import json
import torch
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import seaborn as sns

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
from src.model import BaseNetwork
from config.config import settings
from src.preprocessing import preprocess_data
from src.utils import get_gradients, get_model_file, get_config_file, set_seed


class Activation(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {'name': self.name}


class ELU(Activation):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.config['alpha'] = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.config['alpha']*(torch.exp(x)-1))


class ReLU(Activation):
    def forward(self, x):
        return x * (x > 0).float()


class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1+torch.exp(-x))


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.config['alpha'] = alpha

    def forward(self, x) -> float:
        return torch.where(x > 0, x, self.config['alpha']*x)


class Tanh(Activation):
    def forward(self, x):
        numerator = torch.exp(x)-torch.exp(-x)
        denominator = torch.exp(x)+torch.exp(-x)
        return numerator/denominator


class Swish(Activation):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.config['beta'] = beta

    def forward(self, x):
        return x / (1+torch.exp(-self.config['beta']*x))


activation_fn_name= {'elu': ELU,
                     'relu': ReLU,
                     'sigmoid': Sigmoid,
                     'leakyrelu': LeakyReLU,
                     'tanh': Tanh,
                     'swish': Swish
                     }

ncols = 3
nrows = len(activation_fn_name)//ncols
col, row = 1, 1
showlegend_val=True
fig = make_subplots(rows=nrows, cols=ncols,
                    subplot_titles=list(activation_fn_name.keys()),
                    vertical_spacing = 0.1)

for key, values in activation_fn_name.items():
    activation_fn = values()

    x = torch.linspace(-5, 5, 1000)
    y_values = activation_fn(x)
    x_grads = get_gradients(activation_fn, x)

    fig.add_trace(go.Scatter(x=x.numpy(), y=y_values.numpy(),
                             line=dict(color="red",width=2),
                             name='Activation Function',
                             legendgroup='Activation Function',
                             showlegend=showlegend_val),
                  row=row, col=col)

    fig.add_trace(go.Scatter(x=x.numpy(), y=x_grads.numpy(),
                             line=dict(color="green",width=2),
                             name='Gradients',
                             legendgroup='Gradient',
                             showlegend=showlegend_val),
                  row=row, col=col)

    if col==ncols:
        row+=1
    col = col + 1 if col < ncols else 1
    showlegend_val=False

fig.update_yaxes(range=[-3,3])
fig.update_layout(title="Activation Functions and their Gradients",
                  legend=dict(orientation="h",
                              yanchor="top",
                              y=1.10,
                              xanchor="right",
                              x=1),
                  )
fig.write_html(f"activationFunctions_gradients.html")

def load_file(NeuralNet, model_name):
    """
        Load a neural network and its configuration from saved files.

        This function reads a JSON configuration file and a corresponding
        model checkpoint file. It initializes the neural network with the
        specified architecture and activation function, and then loads
        the trained weights.

        Parameters
        ----------
        NeuralNet : torch.nn.Module or None
            A neural network instance. If None, a new network is initialized
            using the configuration file.
        model_name : str
            The name of the model to load. Used to locate configuration and
            checkpoint files.

        Returns
        -------
        torch.nn.Module
            The neural network with loaded weights.
    """
    config_file, model_file = get_config_file(model_name), get_model_file(model_name)

    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    if NeuralNet:
        activation_fn_title = config_dict['act_fn'].pop('name').lower()
        activation_fn = activation_fn_name[activation_fn_title](**config_dict.pop("act_fn"))
        NeuralNet = BaseNetwork(input_dim = settings.input_dimension,
                                hidden_layers = settings.hidden_layers,
                                n_classes = settings.num_classes,
                                activation_fn = activation_fn)


    NeuralNet.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    return NeuralNet

def save_model(NeuralNet: BaseNetwork, model_name: str):
    """
    Given a model, this function save the state_dict and hyperparameters.

    Inputs:
        NeuralNet - Network object to save parameters from
        model_name - Name of the model (str)
    """
    model_path = os.path.join(settings.model_dir, model_name+".tar")
    torch.save(NeuralNet.state_dict(), model_path)

def train(NeuralNet: BaseNetwork|None=None,
          model_name: str|None=None,
          max_epochs: int|None=None,
          patience: int|None=None,
          batch_size: int|None=None,
          overwrite:bool=False,
          diagnose:bool=False):

    # If trained model file (state dictionary) exists skip training unless specified to overwrite
    modelfile_exists = os.path.isfile(os.path.join(settings.model_dir, model_name+".tar"))
    train_loader, val_loader, test_loader = preprocess_data()

    if modelfile_exists and not overwrite:
        print("Model file already exists. Skipping training...")

    else:
        if modelfile_exists:
            print("Model file exists, but will be overwritten...")

        optimizer = optim.SGD(NeuralNet.parameters(),
                                  lr=settings.learning_rate,
                                  momentum=settings.momentum)

        loss = nn.CrossEntropyLoss()
        NeuralNet.train()
        if max_epochs is None:
            max_epochs = settings.max_epochs
        if patience is None:
            patience = settings.patience
        val_score = []
        best_val_epoch = -1


        for epoch in range(max_epochs):
            train_score = []

            num_iterations = 50 # *** Diagnosis vars
            frames_to_capture = 10 # *** Diagnosis vars
            capture_interval = num_iterations // frames_to_capture # * Diagnosis vars
            frame_count=0
            gif_image_list = []
            for batch_num, (images, labels), in enumerate(train_loader):
                pred = NeuralNet(images)
                loss_val = loss(pred, labels)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                true_preds = (pred.argmax(dim=-1)==labels).sum()
                count =labels.shape[0]
                train_score.append(true_preds/count)

                # For Diagnose - we only record gradients for first epoch
                if diagnose and epoch==9 and batch_num%capture_interval==0:
                    frame_count+=1
                    gradients_snapshot = NeuralNet.get_gradient_snapshot()
                    filename = os.path.join(settings.data_dir, f"frame_{frame_count:03d}.png")
                    # Plot and save the frame
                    plot_and_save_gradients(gradients_snapshot,
                                            f"Gradients (Iteration {batch_num})",
                                            filename)

                    print(f"Captured frame {frame_count}/{frames_to_capture} at iteration {batch_num}")
                    gif_image_list.append(filename)

            if diagnose and epoch==9:
                frames = [imageio.imread(filename) for filename in gif_image_list]
                gif_path = os.path.join(settings.data_dir,"gradient_evolution.gif")
                imageio.mimsave(gif_path, frames, fps=10)  # fps is frames per second

                print(f"GIF saved to {gif_path}")

            train_accuracy = np.mean(train_score)
            val_accuracy = test_model(NeuralNet, val_loader)
            val_score.append(val_accuracy)

            print(f"[Epoch {epoch + 1:2d}] Training accuracy: {train_accuracy * 100.0:05.2f}%,"
                  f" Validation accuracy: {val_accuracy * 100.0:05.2f}%")

            if len(val_score) == 1 or val_accuracy > val_score[best_val_epoch]:
                print("\t   (New best performance, saving model...)")
                save_model(NeuralNet, model_name)
                best_val_epoch = epoch
            elif best_val_epoch <= epoch - patience:
                print(f"Early stopping due to no improvement over the last {patience} epochs")
                break


    NeuralNet = load_file(NeuralNet, model_name)
    test_accuracy = test_model(NeuralNet, test_loader)
    print(f" Test accuracy: {test_accuracy * 100.0:4.2f}% ")
    return test_accuracy

def test_model(NeuralNet:BaseNetwork, val_loader: DataLoader):
    NeuralNet.eval()
    with torch.no_grad():
        val_accuracy = []
        for images, labels in val_loader:
            pred = NeuralNet(images)
            true_preds = (pred.argmax(dim=-1) == labels).sum()
            count = labels.shape[0]
            val_accuracy.append(true_preds / count)

    return np.mean(val_accuracy)

def run_activation_fns():
    for act_fn_name in activation_fn_name:
        print(f"Training BaseNetwork with {act_fn_name} activation...")
        set_seed(42)
        act_fn = activation_fn_name[act_fn_name]()
        network_actfn = BaseNetwork(act_fn)
        accuracy = train(network_actfn, f"FashionMNIST_{act_fn_name}", overwrite=False)

def diagnose_run(act_fn_name):
    act_fn = activation_fn_name[act_fn_name]()
    network_actfn = BaseNetwork(act_fn)
    test_accuracy = train(network_actfn,
                                         f"FashionMNIST_{act_fn_name}",
                                         overwrite=True,
                                         diagnose=True)

def plot_and_save_gradients(gradients:dict, title:str, filename:str):
    layer_names = list(gradients.keys())
    gradients = [g.flatten() for g in gradients.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(gradients, vert=False, patch_artist=True, boxprops=dict(facecolor='royalblue'))
    ax.set_yticklabels(layer_names)
    ax.set_title(title)
    ax.set_xlabel("Gradient Value")
    ax.set_ylabel("Layer Name")
    ax.set_xlim(-0.05, 0.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

        # Save the figure
    plt.savefig(filename)
    plt.close(fig)
