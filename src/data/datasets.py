"""
Dataset loading utilities.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from ..config import Config, get_config


def load_full_dataset(config: Config = None) -> Tuple[Dataset, Dataset, int]:
    """
    Load the full train and test datasets based on configuration.
    
    Args:
        config: Configuration object. If None, uses global config.
    
    Returns:
        Tuple of (train_dataset, test_dataset, num_classes)
    """
    if config is None:
        config = get_config()
    
    dataset_name = config.dataset.name.lower()
    data_root = Path(config.paths.data_root)
    
    # Basic transform for loading
    transform = transforms.ToTensor()
    
    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=str(data_root),
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(data_root),
            train=False,
            download=True,
            transform=transform,
        )
        num_classes = 10
        
    elif dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=str(data_root),
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=str(data_root),
            train=False,
            download=True,
            transform=transform,
        )
        num_classes = 100
        
    elif dataset_name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root=str(data_root),
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.MNIST(
            root=str(data_root),
            train=False,
            download=True,
            transform=transform,
        )
        num_classes = 10
        
    elif dataset_name == "fashionmnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=str(data_root),
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=str(data_root),
            train=False,
            download=True,
            transform=transform,
        )
        num_classes = 10
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes


def get_dataset_labels(dataset: Dataset) -> np.ndarray:
    """
    Extract labels from a dataset.
    
    Args:
        dataset: PyTorch dataset with targets attribute
    
    Returns:
        NumPy array of labels
    """
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets, dtype=int)
    elif hasattr(dataset, 'labels'):
        return np.array(dataset.labels, dtype=int)
    else:
        # Fallback: iterate through dataset
        labels = []
        for _, label in dataset:
            labels.append(label)
        return np.array(labels, dtype=int)
