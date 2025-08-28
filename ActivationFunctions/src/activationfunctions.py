import torch
import torch.nn as nn

from torch import Tensor
from utils import get_gradients
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
                     'leaky_relu': LeakyReLU,
                     'tanh': Tanh,
                     'swish': Swish
                     }


def visualize(activation_fn: Activation, x: Tensor):
    y_values = activation_fn(x)
    x_grads = get_gradients(activation_fn, x)

ncols = 3
nrows = len(activation_fn_name)//ncols
col, row = 1, 1
fig = make_subplots(rows=nrows, cols=ncols)

for key, values in activation_fn_name.items():
    activation_fn = values()

    x = torch.linspace(-5, 5, 1000)
    y_values = activation_fn(x)
    x_grads = get_gradients(activation_fn, x)

    fig.add_trace(go.Scatter(x=x.numpy(), y=y_values.numpy(),
                             line=dict(color="red",width=2)),
                  row=row, col=col)

    fig.add_trace(go.Scatter(x=x.numpy(), y=x_grads.numpy(),
                             line=dict(color="green",width=2)),
                  row=row, col=col)

    if col==ncols:
        row+=1
    col = col + 1 if col < ncols else 1

fig.update_layout(title="Activation Functions and their Gradients",
                  showlegend=False)
fig.write_html(f"activationFunctions_gradients.html")