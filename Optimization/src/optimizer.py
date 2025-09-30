import os
import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
from src.model import BaseNetwork
from config.config import settings

class Optimizer():
    def __init__(self, params:Module.parameters(), lr: float):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is not None:
                self.update_params(p)

    def update_params(self, p):
        return NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr):
        super.__init__(params, lr)

    def update_params(self, p):
        p_update = -self.lr * p.grad
        p.add_(p_update)

class SGDMomentum(Optimizer):
    def __init__(self, params, lr):
        super.__init__(params, lr)

    def update_params(self, p):
        p_update = -self.lr * p.grad
        p.add_(p_update)