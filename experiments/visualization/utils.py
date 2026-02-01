"""
Common utilities for visualization experiments.
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from src.config import Config, get_config


class SimpleFeatureExtractor:
    """
    Simplified feature extractor for visualization purposes.
    Uses ResNet-18 pretrained features.
    """
    
    def __init__(self, device: torch.device = None, config: Config = None):
        if config is None:
            config = get_config()
        
        if device is None:
            device = torch.device(config.model.device)
        
        self.device = device
        
        weights_tag = config.model.weights
        weights_enum = torchvision.models.ResNet18_Weights
        weights = getattr(weights_enum, weights_tag, weights_enum.DEFAULT)
        
        self.preprocessor = weights.transforms()
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        self.feature_dim = 512
        
        self.model = model.to(self.device)
        self.model.eval()
    
    def extract(
        self,
        images: np.ndarray,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> np.ndarray:
        """
        Extract features from batch of images.
        
        Args:
            images: Array of shape (N, H, W, C) uint8
            batch_size: Batch size for extraction
            num_workers: Number of data loader workers
        
        Returns:
            Features of shape (N, D)
        """
        class _ImgDataset(Dataset):
            def __init__(self, imgs, transform):
                self.imgs = imgs
                self.transform = transform
            
            def __len__(self):
                return len(self.imgs)
            
            def __getitem__(self, idx):
                img = self.imgs[idx]
                pil = transforms.functional.to_pil_image(img)
                return self.transform(pil)
        
        ds = _ImgDataset(images, self.preprocessor)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        feats = []
        with torch.no_grad():
            for x in dl:
                x = x.to(self.device)
                z = self.model(x)
                feats.append(z.cpu().numpy())
        
        return np.concatenate(feats, axis=0)


def ensure_cifar10c_downloaded(root: str) -> str:
    """
    Ensure CIFAR-10-C is downloaded and extracted.
    
    Args:
        root: Root directory for data
    
    Returns:
        Path to CIFAR-10-C directory
    """
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)
    target_dir = os.path.join(root, 'CIFAR-10-C')
    
    # Check if already exists
    if os.path.isdir(target_dir) and os.path.isfile(os.path.join(target_dir, 'labels.npy')):
        return target_dir
    
    tar_path = os.path.join(root, 'CIFAR-10-C.tar')
    
    if not os.path.exists(tar_path):
        url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1'
        print(f"Downloading CIFAR-10-C from {url}...")
        urllib.request.urlretrieve(url, tar_path)
    
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=root)
    
    print("CIFAR-10-C extracted.")
    return target_dir


def sample_paired_cifar10_and_c(
    classes: List[int],
    n_per_class: int = 100,
    corruption: str = 'brightness',
    severity: int = 5,
    seed: int = 42,
    config: Config = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample paired images from CIFAR-10 test and CIFAR-10-C.
    
    Args:
        classes: List of class indices to sample
        n_per_class: Number of samples per class per domain
        corruption: Corruption type
        severity: Severity level (1-5)
        seed: Random seed
        config: Configuration object
    
    Returns:
        Tuple of (images, labels, domains)
        - images: (N, 32, 32, 3) uint8
        - labels: (N,) int class IDs
        - domains: (N,) int 0=source, 1=target
    """
    if config is None:
        config = get_config()
    
    assert 1 <= severity <= 5
    
    # Load CIFAR-10 test
    transform = transforms.ToTensor()
    cifar10_test = torchvision.datasets.CIFAR10(
        root=config.paths.data_root,
        train=False,
        download=True,
        transform=transform,
    )
    test_images = cifar10_test.data  # (10000, 32, 32, 3) uint8
    test_labels = np.array(cifar10_test.targets, dtype=int)
    
    # Load CIFAR-10-C
    c_root = ensure_cifar10c_downloaded(config.paths.data_root)
    c_path = os.path.join(c_root, f'{corruption}.npy')
    labels_path = os.path.join(c_root, 'labels.npy')
    
    c_data = np.load(c_path)  # (50000, 32, 32, 3)
    c_labels = np.load(labels_path)  # (50000,)
    
    # Select severity slice
    start = (severity - 1) * 10000
    end = severity * 10000
    c_images_slice = c_data[start:end]
    
    rng = np.random.default_rng(seed)
    
    src_imgs = []
    tgt_imgs = []
    
    for cls in classes:
        idx_all = np.where(test_labels == cls)[0]
        if len(idx_all) < n_per_class:
            raise ValueError(f"Class {cls} has only {len(idx_all)} samples")
        
        chosen = rng.choice(idx_all, size=n_per_class, replace=False)
        src_imgs.append(test_images[chosen])
        tgt_imgs.append(c_images_slice[chosen])
    
    src_imgs = np.concatenate(src_imgs, axis=0)
    tgt_imgs = np.concatenate(tgt_imgs, axis=0)
    
    # Combine
    num_per_domain = len(classes) * n_per_class
    labels_src = [cls for cls in classes for _ in range(n_per_class)]
    labels_tgt = list(labels_src)
    domains_src = [0] * num_per_domain
    domains_tgt = [1] * num_per_domain
    
    images = np.concatenate([src_imgs, tgt_imgs], axis=0)
    labels = np.array(labels_src + labels_tgt, dtype=int)
    domains = np.array(domains_src + domains_tgt, dtype=int)
    
    return images, labels, domains
