import math
import torch
import torch.nn as nn

from src.model import BaseNetwork

def constant_initialize(model: BaseNetwork, constant):
    for param in model.parameters():
        param.data.fill_(constant)


def constant_variance(model: BaseNetwork, variance):
    for param in model.parameters():
        param.data.normal_(std=variance)


def xavier_initialize(model: BaseNetwork):
    for name, param in model.named_parameters():
        if "bias" in name:
            param.data.fill_(0)
        else:
            d_y, d_x = param.shape[0], param.shape[1]
            bound = math.sqrt(6)/(math.sqrt((d_x+d_y)))
            param.data.uniform_(-bound, bound)


def kaiming_initialize(model: BaseNetwork):
    for name, param in model.named_parameters():
        if "bias" in name:
            param.data.fill_(0)
        elif name.startswith("layers.0"):
            param.data.normal_(0, 1/math.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, math.sqrt(2)/math.sqrt(param.shape[1]))
