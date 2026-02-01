"""
CIFAR-10-C corrupted dataset support.
"""

import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ..config import Config, get_config


class CIFAR10CSubset(Dataset):
    """
    Dataset class for CIFAR-10-C corrupted images.
    
    CIFAR-10-C contains 19 corruption types, each with 5 severity levels.
    Each corruption file contains 50000 images (10000 per severity level).
    """
    
    CORRUPTIONS = [
        'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
        'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
        'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
        'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
        'zoom_blur'
    ]
    
    def __init__(
        self,
        root: str,
        corruption: str = 'brightness',
        severity: int = 5,
        transform: Optional[Callable] = None,
        indices: Optional[np.ndarray] = None,
    ):
        """
        Initialize CIFAR-10-C dataset.
        
        Args:
            root: Root directory containing CIFAR-10-C folder
            corruption: Corruption type (e.g., 'brightness', 'gaussian_noise')
            severity: Severity level (1-5)
            transform: Optional transform to apply to images
            indices: Optional subset indices to use
        """
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
        
        # Load corruption data
        c10c_dir = self.root / 'CIFAR-10-C'
        corruption_path = c10c_dir / f'{corruption}.npy'
        labels_path = c10c_dir / 'labels.npy'
        
        if not corruption_path.exists():
            raise FileNotFoundError(f"Corruption file not found: {corruption_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        # Load full data
        all_data = np.load(corruption_path)  # (50000, 32, 32, 3)
        all_labels = np.load(labels_path)    # (50000,)
        
        # Select severity level (each level has 10000 images)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        self.data = all_data[start_idx:end_idx]
        self.targets = all_labels[start_idx:end_idx].astype(int)
        
        # Apply subset indices if provided
        if indices is not None:
            self.data = self.data[indices]
            self.targets = self.targets[indices]
        
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """
        Get item by index.
        
        Args:
            idx: Index
        
        Returns:
            Tuple of (image, label)
        """
        img = self.data[idx]  # HWC uint8
        label = int(self.targets[idx])
        
        if self.transform is not None:
            # Convert to PIL Image for transforms
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            # Default: convert to tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, label


def load_cifar10c_subset(
    corruption: str = 'brightness',
    severity: int = 5,
    indices: Optional[np.ndarray] = None,
    config: Config = None,
) -> CIFAR10CSubset:
    """
    Load CIFAR-10-C subset with specified corruption and severity.
    
    Args:
        corruption: Corruption type
        severity: Severity level (1-5)
        indices: Optional subset indices
        config: Configuration object
    
    Returns:
        CIFAR10CSubset dataset
    """
    if config is None:
        config = get_config()
    
    return CIFAR10CSubset(
        root=config.paths.data_root,
        corruption=corruption,
        severity=severity,
        indices=indices,
    )
