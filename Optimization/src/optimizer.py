import os
import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
from src.model import BaseNetwork
from config.config import settings

class Optimizer():
    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            self.update_params(p)

    def update_params(self, p):
        return NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)

    def update_params(self, p):
        p_update = -self.lr * p.grad
        p.add_(p_update)

class SGDMomentum(Optimizer):
    def __init__(self, params, lr, momentum):
        super().__init__(params, lr)
        self.momentum = momentum
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}

    def update_params(self, p):
        self.param_momentum[p] = (1 - self.momentum) * p.grad + self.momentum * self.param_momentum[p]
        p_update = -self.lr * self.param_momentum[p]
        p.add_(p_update)

class Adam(Optimizer):
    def __init__(self, params, lr, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta_1= beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.t = {p: 0 for p in self.params}
        self.param_momentum1 = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_momentum2 = {p: torch.zeros_like(p.data) for p in self.params}

    def update_params(self, p):
        self.t[p] += 1

        self.param_momentum1[p] = ((1 - self.beta_1) * p.grad +
                                   self.beta_1 * self.param_momentum1[p])
        self.param_momentum2[p] = ((1 - self.beta_2) * (p.grad**2)
                                   + self.beta_2 * self.param_momentum2[p])

        param_momentum1_correct = self.param_momentum1[p]/(1-self.beta_1**self.t[p])
        param_momentum2_correct = self.param_momentum2[p]/(1-self.beta_2**self.t[p])

        p_update = -(self.lr/(torch.sqrt(param_momentum2_correct)+self.eps)) * param_momentum1_correct
        p.add_(p_update)
