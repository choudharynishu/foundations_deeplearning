import torch
import torch.nn as nn
from config.config import settings

class BaseNetwork(nn.Module):
    def __init__(self, activation_fn,input_dim: int = settings.input_dimension,
                 hidden_layers: list= settings.hidden_layers,
                 n_classes: int = settings.num_classes,
                 ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features=input_dim, out_features=hidden_layers[0]))
        layers.append(activation_fn)

        for in_dim, out_dim in zip(hidden_layers, hidden_layers[1:]):
            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim)),
            layers.append(activation_fn)

        layers.append(nn.Linear(in_features=hidden_layers[-1], out_features=n_classes))

        self.layers = nn.Sequential(*layers)

        self.config = {'input_dim': input_dim, 'activation_fn': activation_fn,
                       'hidden_layers': hidden_layers, 'n_classes': n_classes}

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out
