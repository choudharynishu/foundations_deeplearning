import os
import json
import torch

from src.model import BaseNetwork
from config.config import settings
from src.trainer import load_model

load_model(settings.model_dir, model_name='FashionMNIST_tanh', NeuralNet=None)
