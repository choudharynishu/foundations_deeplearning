import torch.nn as nn
from config.config import settings

class BaseNetwork(nn.Module):
    def __init__(self, activation_function,
                 input_dim: int=settings.input_dim,
                 hidden_layers:list=settings.hidden_layers,
                 num_classes: int = settings.num_classes):
        super().__init__()
        layers=[]
        layers.append(nn.Linear(in_features=input_dim, out_features=hidden_layers[0]))
        layers.append(activation_function)

        for in_dim, out_dim in zip(hidden_layers, hidden_layers[1:]):
            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim)),
            layers.append(activation_function)

        layers.append(nn.Linear(in_features=hidden_layers[-1], out_features=num_classes))

        self.layers = nn.Sequential(*layers)
        self.gradients = {}  # Store gradients for diagnosis

        self.config = {'input_dim': input_dim, 'activation_function': activation_function,
                       'hidden_layers': hidden_layers, 'num_classes': num_classes}
        self._register_hooks()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.layers(x)
        return output

    def _register_hooks(self):
        for name, param in self.named_parameters():
            param.register_hook(lambda grad, name=name: self._save_gradient(grad,name))

    def _save_gradient(self, grad, name):
        if grad is not None:
            self.gradients[name] = grad.detach().cpu().numpy()
