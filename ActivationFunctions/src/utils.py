import os
import torch
import numpy as np

from config.config import settings


def get_gradients(activation_fn, x):
    x = x.clone().requires_grad_()
    z = activation_fn(x)
    z.backward(torch.ones_like(x))
    return x.grad

def get_config_file(model_name):
    return os.path.join(settings.model_dir, model_name+".config")

def get_model_file(model_name):
    return os.path.join(settings.model_dir, model_name+".tar")

def set_seed(seed=settings.seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

