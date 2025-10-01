import os
import json
import torch
import numpy as np
import torch.nn as nn
from src.model import BaseNetwork
from config.config import settings
from src.trainer import train
import src.optimizer as optimizer
from src.pathological_curve import plot_weights


# Function for setting the seed
def set_seed(seed=settings.seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed()

model = BaseNetwork(nn.Tanh())
SGD_results = train(model, "FashionMNIST_SGD",
                          lambda params: optimizer.SGD(params, lr=1e-1),
                          max_epochs=40, batch_size=256)
model = BaseNetwork(nn.Tanh())
SGDMom_results = train(model, "FashionMNIST_SGDMom",
                             lambda params: optimizer.SGDMomentum(params, lr=1e-1, momentum=0.9),
                             max_epochs=40, batch_size=256)
model = BaseNetwork(nn.Tanh())
Adam_results = train(model, "FashionMNIST_Adam",
                           lambda params: optimizer.Adam(params, lr=1e-3),
                           max_epochs=40, batch_size=256)
plot_weights()
