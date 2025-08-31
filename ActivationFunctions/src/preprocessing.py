import os
import torch
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split

from config.config import settings


def preprocess_data():
    train_data, val_data, test_data = download_data()
    train_loader = create_dataloader(train_data, train=True)
    val_loader = create_dataloader(val_data, train=False)
    test_loader = create_dataloader(test_data, train=False)

    return train_loader, val_loader, test_loader


def download_data(train_val_split: float = 0.8):
    """
        Download and preprocess the FashionMNIST dataset.

        This function applies a series of transformations to the dataset:
        - Converts images to PyTorch tensors.
        - Normalizes pixel values to the range [-1, 1].

        It downloads both the training and testing portions of the
        FashionMNIST dataset and applies the same transformations.

        Parameters
        ----------
        train_val_split : float, optional
        The proportion of the original training set to use for training.
        The remainder is used for validation. Default is 0.8 (80% train,
        20% validation).

        Returns
        -------
        tuple
            A tuple containing:
            - train_data (torchvision.datasets.FashionMNIST): The training dataset.
            - val_data (torchvision.datasets.FashionMNIST): The testing dataset.
            - test_data (torchvision.datasets.FashionMNIST): The testing dataset.
    """
    transform_list = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, ), (0.5,))])

    train_data = datasets.FashionMNIST(root = settings.data_dir,
                                       train=True,
                                       transform = transform_list,
                                       download=True)

    test_data = datasets.FashionMNIST(root = settings.data_dir,
                                       train=False,
                                       transform = transform_list,
                                       download=True)
    train_data_len = int(len(train_data)*train_val_split)
    val_data_len = len(train_data) - train_data_len
    split_sequence = [train_data_len, val_data_len]

    train_dataset, validation_dataset = random_split(train_data, split_sequence)

    return (train_dataset,validation_dataset,test_data)

def create_dataloader(data: Dataset, train: bool = True, batchsize: int | None = None):
    """
        Create a DataLoader for the given dataset.

        This function wraps a PyTorch dataset into a DataLoader, which
        allows batching, shuffling, and parallel data loading.

        Parameters
        ----------
        data : torch.utils.data.Dataset
            The dataset to wrap into a DataLoader.
        train : bool, optional
            If True, shuffle the data. Default is True.
        batchsize : int or None, optional
            The number of samples per batch. If None, defaults to
            ``settings.batchsize``.

        Returns
        -------
        torch.utils.data.DataLoader
            The DataLoader object for the dataset.
    """
    if batchsize is None:
        batchsize = settings.batchsize

    dataloader = DataLoader(
        data,
        batch_size=batchsize,
        shuffle=train,
        drop_last=False
    )
    return dataloader

def visualized_sample_data(data: Dataset, index:int = 42):
    """
        Save a sample image from the dataset to the data directory.

        Parameters
        ----------
        data : torch.utils.data.Dataset
            The dataset from which to extract the sample image.
        index : int, optional
            The index of the sample to visualize and save.
            Default is 42.

        Returns
        -------
        None
            The function saves the image as a PNG file in the data directory.
    """
    image, label = data[index]

    # Convert tensor back to PIL image if needed
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    save_path = os.path.join(
        settings.data_dir, f"sample_image_label{label}.png"
    )

    image.save(save_path)
