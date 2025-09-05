import torch
import torch.nn as nn
from config.config import settings
from sympy.abc import lamda


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
        self.gradients = {} #Store gradients for diagnosis

        self.config = {'input_dim': input_dim, 'activation_fn': activation_fn,
                       'hidden_layers': hidden_layers, 'n_classes': n_classes}
        self._register_hooks()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        for name, param in self.named_parameters():
            param.register_hook(lambda grad, name=name: self._save_grad(grad, name))

    def _save_grad(self, grad, name):
        if grad is not None:
            self.gradients[name] = grad.detach().cpu().numpy()

    def get_gradient_snapshot(self):
        """Returns current gradient values in form of a dictonary, with layer names as keys."""
        return {name: grad.copy() for name, grad in self.gradients.items()}
