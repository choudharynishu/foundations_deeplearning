import torch
from torch import Tensor


def get_gradients(activation_fn, x):
    x = x.clone().requires_grad_()
    z = activation_fn(x)
    z.backward(torch.ones_like(x))
    return x.grad
