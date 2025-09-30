import os
import json
import torch
import torchvision

import torch.nn as nn
from plotly.graph_objs import go
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

from src.model import BaseNetwork
from config.config import settings


def _get_config_file(model_path: str, model_name: str):
    if model_path is None:
        model_path = settings.model_dir
    return os.path.join(model_path, f"{model_name}.config")

def _get_model_file(model_path: str, model_name: str):
    return os.path.join(model_path, f"{model_name}.tar")

def _get_result_file(model_path: str, model_name: str):
    return os.path.join(model_path, f"{model_name}_results.json")

def load_model(model_path: str, model_name: str, NeuralNet: BaseNetwork):
    config_file = _get_config_file(model_path, model_name)
    model_file = _get_model_file(model_path, model_name)

    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    if NeuralNet is None:
        activation_func = config_dict["act_fn"]["name"]
        activation_function = getattr(nn, activation_func)
        NeuralNet = BaseNetwork(activation_function)

    NeuralNet.load_state_dict(torch.load(model_file))
    return NeuralNet

def save_model(NeuralNet: BaseNetwork, model_path:str, model_name:str):
    if NeuralNet is None:
       NeuralNet = load_model(model_path, model_name, NeuralNet=None)
    config_dict = NeuralNet.config

    config_file = _get_config_file(model_path, model_name)
    model_file = _get_model_file(model_path, model_name)

    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(NeuralNet.state_dict(), model_file)

def download_data(train_val_split: float = 0.8):
    transform_list = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])

    train_data = datasets.FashionMNIST(root=settings.data_dir,
                                       train=True,
                                       transform=transform_list,
                                       download=True)

    train_data_len = int(len(train_data) * train_val_split)
    val_data_len = len(train_data) - train_data_len
    split_sequence = [train_data_len, val_data_len]

    train_dataset, validation_dataset = random_split(train_data, split_sequence)

    test_data = datasets.FashionMNIST(root=settings.data_dir,
                                      train=False,
                                      transform=transform_list,
                                      download=True)

    return train_dataset, validation_dataset, test_data

def train(NeuralNet:BaseNetwork, model_name: str, optimizer_func,
          max_epochs: int=settings.max_epochs, batch_size: int=settings.batch_size):

    train_data, validation_data, test_data = download_data(train_val_split=0.8)
    file_exists = os.path.isfile(_get_model_file(settings.model_dir, model_name))

    if file_exists:
        print(f"Model file of \"{model_name}\" already exists. Skipping training...")
        with open(_get_result_file(settings.model_dir, model_name), "r") as f:
            results = json.load(f)
    else:
        if NeuralNet is None:
            NeuralNet = load_model(model_path=settings.model_dir,
                               model_name= model_name)

        optimizer = optimizer_func(NeuralNet.parameter,
                                   learning_rate = settings.learning_rate)

        loss = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_data, batch_size=settings.batch_size,
                                  shuffle=True, drop_last=False)
        NeuralNet.train()

        results = None
        val_scores = []
        train_losses, train_scores = [], []
        best_val_epoch = -1

        for epoch in max_epochs:
            true_pred, count = 0, 0
            for images, labels in train_loader:
                preds = NeuralNet(images)
                loss_val = loss(preds, labels)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                true_pred += (preds.argmax(dim=-1) == labels).sum().item()
                count += labels.shape[0]
                train_losses.append(loss_val.item())

            train_acc = true_pred / count
            train_scores.append(train_acc)
            train_losses.append()

            validation_accuracy = test(NeuralNet, validation_data)
            val_scores.append(validation_accuracy)
            print(
                f"[Epoch {epoch + 1:2d}] Training accuracy: {train_acc * 100.0:05.2f}%,"
                f" Validation accuracy: {validation_accuracy  * 100.0:05.2f}%")

            if len(val_scores) == 1 or validation_accuracy > val_scores[best_val_epoch]:
                print("\t   (New best performance, saving model...)")
                save_model(NeuralNet, settings.model_dir, model_name)
                best_val_epoch = epoch
    if results is None:
        test_acc = test(NeuralNet, test_data)
        results = {"test_acc": test_acc, "val_scores": val_scores, "train_losses": train_losses,
                       "train_scores": train_scores}
        with open(_get_result_file(settings.model_dir, model_name), "w") as f:
            json.dump(results, f)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=results["train_scores"], name="Train Accuracy"))
    fig.add_trace(go.Scatter(y=results["val_scores"], name="Validation Accuracy"))
    fig.update_trace(
        title=f"Validation performance of {model_name}",
        xaxis = dict(title=f"Epochs"),
        yaxis = dict(title=f"Accuracy")
    )
    fig.write_html(settings.artifacts, f"validation_accuracy_{model_name}")
    return results

def test(NeuralNet, test_data: datasets):
    NeuralNet.eval()
    test_loader = DataLoader(test_data, batch_size=settings.batch_size,
                             shuffle=False, drop_last=False)
    true_pred, count = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = NeuralNet(images)
            true_pred += (preds.argmax(dim=-1) == labels).sum().item()
            count += labels.shape[0]

        test_acc = true_pred / count
    return test_acc
