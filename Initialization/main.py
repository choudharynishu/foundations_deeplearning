import torch.nn as nn
from src.preprocess import (visualize_gradients,
                            visualize_activations,
                            visualize_preactivations,
                            download_data)
from src.model import BaseNetwork
from src.initialize import initialize
from config.config import settings

def visualize(train_dataset, init_technique:str='kaiming', **kwargs):
    print(init_technique)
    model = initialize(init_technique, **kwargs)
    visualize_gradients(model, train_dataset, init_technique)
    visualize_preactivations(model, train_dataset, init_technique)
    visualize_activations(model, train_dataset, init_technique)

if __name__=='__main__':
    train_dataset, _, _ = download_data()
    # initialization_list = [("constant", {"constant": settings.constant_init}),
    #                        ("variance", {"variance": 0.01}),
    #                        ("xavier", {}),
    #                        ("kaiming", {})]
    initialization_list = [("variance", {"variance": 0.01}),
                           ("xavier", {}),
                           ("kaiming", {})]

    for init_tech, params in initialization_list:
        visualize(train_dataset,init_tech,
                  **params)