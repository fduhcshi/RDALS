"""
Feature extraction using pre-trained models.
Supports loading from pre-computed embedding cache.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset

from ..config import Config, get_config


class TransformedDataset(Dataset):
    """Wrapper dataset that applies transforms to images."""
    
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Convert tensor to PIL Image if needed
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        # Convert to RGB if needed
        if hasattr(img, 'mode') and img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


class FeatureExtractor:
    """
    Feature extractor using pre-trained models.
    Supports caching of extracted features.
    """
    
    def __init__(self, model_name: str = None, device: str = None, config: Config = None):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Model name (resnet18, resnet50, vit_b_16)
            device: Device to use for computation
            config: Configuration object
        """
        if config is None:
            config = get_config()
        
        self.config = config
        self.model_name = model_name or config.model.name
        self.device = torch.device(device or config.model.device)
        
        # Load model and preprocessor
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model and set up for feature extraction."""
        weights_tag = self.config.model.weights
        
        if self.model_name == "resnet18":
            weights_enum = torchvision.models.ResNet18_Weights
            weights = getattr(weights_enum, weights_tag, weights_enum.DEFAULT)
            self.preprocessor = weights.transforms()
            model = torchvision.models.resnet18(weights=weights)
            model.fc = nn.Identity()
            self.feature_dim = 512
            
        elif self.model_name == "resnet50":
            weights_enum = torchvision.models.ResNet50_Weights
            weights = getattr(weights_enum, weights_tag, weights_enum.DEFAULT)
            self.preprocessor = weights.transforms()
            model = torchvision.models.resnet50(weights=weights)
            model.fc = nn.Identity()
            self.feature_dim = 2048
            
        elif self.model_name == "vit_b_16":
            weights_enum = torchvision.models.ViT_B_16_Weights
            weights = getattr(weights_enum, weights_tag, weights_enum.DEFAULT)
            self.preprocessor = weights.transforms()
            model = torchvision.models.vit_b_16(weights=weights)
            model.heads = nn.Identity()
            self.feature_dim = 768
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        self.model = model.to(self.device)
        self.model.eval()
    
    def _apply_transforms(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing transforms to dataset."""
        return TransformedDataset(dataset, self.preprocessor)
    
    def _infer_cache_spec(self, dataset: Dataset):
        """
        Infer cache specification from dataset.
        
        Returns:
            Tuple of (split, indices, cache_tag)
            - split: 'train', 'test', 'custom', or None
            - indices: numpy array of indices
            - cache_tag: string tag for cache file
        """
        default_cache_tag = self.config.dataset.name.lower()
        
        # Custom dataset with explicit cache_tag + indices (e.g. CIFAR-10-C)
        if hasattr(dataset, 'cache_tag') and hasattr(dataset, 'indices'):
            try:
                tag = str(getattr(dataset, 'cache_tag'))
                idx = np.array(getattr(dataset, 'indices'))
                return 'custom', idx, tag
            except Exception:
                pass
        
        if isinstance(dataset, Subset):
            base = dataset.dataset
            if hasattr(base, 'train'):
                split = 'train' if getattr(base, 'train') else 'test'
            else:
                split = None
            return split, np.array(dataset.indices), default_cache_tag
        else:
            base = dataset
            if hasattr(base, 'train'):
                split = 'train' if getattr(base, 'train') else 'test'
                return split, np.arange(len(base)), default_cache_tag
            return None, None, None
    
    @torch.no_grad()
    def extract_features(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataset, using cache if available.
        
        Args:
            dataset: PyTorch dataset
        
        Returns:
            Tuple of (features, labels) as NumPy arrays
        """
        cache_dir = Path(self.config.paths.embedding_dir)
        weights_tag = self.config.model.weights.lower()
        
        split, indices, cache_tag = self._infer_cache_spec(dataset)
        
        if cache_tag is None:
            cache_tag = self.config.dataset.name.lower()
        
        cache_name = f"{cache_tag}_{self.model_name}_{weights_tag}.npz"
        cache_path = cache_dir / cache_name
        
        if split is not None and indices is not None:
            if cache_path.exists():
                data = np.load(str(cache_path))
                if split == 'custom':
                    Z_full = data['Z']
                    Y_full = data['Y']
                else:
                    Z_full = data['Z_train'] if split == 'train' else data['Z_test']
                    Y_full = data['Y_train'] if split == 'train' else data['Y_test']
                Z = Z_full[indices]
                Y = Y_full[indices]
                return Z, Y
        
        # Fallback: extract online
        return self._extract_online(dataset)
    
    def _extract_online(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features online (without cache)."""
        transformed = self._apply_transforms(dataset)
        num_workers = self.config.model.num_workers
        loader = DataLoader(
            transformed,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                feats = self.model(images)
                features_list.append(feats.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return features, labels
    
    def _forward_to_layer3(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass up to layer3 for ResNet."""
        m = self.model
        required = all(hasattr(m, a) for a in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3'])
        if not required:
            raise NotImplementedError("extract_layer3_features only supports ResNet architecture")
        x = m.conv1(images)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        return x
    
    @torch.no_grad()
    def extract_layer3_features(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract layer3 features (for ResNet models only).
        
        Args:
            dataset: PyTorch dataset
        
        Returns:
            Tuple of (features, labels) as NumPy arrays
        """
        cache_dir = Path(self.config.paths.embedding_layer3_dir)
        weights_tag = self.config.model.weights.lower()
        
        split, indices, cache_tag = self._infer_cache_spec(dataset)
        
        if cache_tag is None:
            cache_tag = self.config.dataset.name.lower()
        
        cache_name = f"{cache_tag}_{self.model_name}_{weights_tag}_layer3.npz"
        cache_path = cache_dir / cache_name
        
        if split is not None and indices is not None:
            if cache_path.exists():
                data = np.load(str(cache_path))
                if split == 'custom':
                    Z_full = data['Z3']
                    Y_full = data['Y']
                else:
                    Z_full = data['Z3_train'] if split == 'train' else data['Z3_test']
                    Y_full = data['Y_train'] if split == 'train' else data['Y_test']
                Z = Z_full[indices]
                Y = Y_full[indices]
                return Z, Y
        
        # Fallback: extract online
        return self._extract_layer3_online(dataset)
    
    def _extract_layer3_online(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract layer3 features online."""
        transformed = self._apply_transforms(dataset)
        num_workers = self.config.model.num_workers
        loader = DataLoader(
            transformed,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                feats = self._forward_to_layer3(images)
                features_list.append(feats.cpu().numpy().astype(np.float16))
                labels_list.append(labels.cpu().numpy().astype(np.int64))
        
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return features, labels
