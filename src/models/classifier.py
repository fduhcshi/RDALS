"""
Classifier training and evaluation utilities.
"""

from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score

from ..config import Config, get_config
from .extractor import FeatureExtractor


def _train_head(
    Z: np.ndarray,
    Y: np.ndarray,
    num_classes: int,
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 256,
    device: str = "cuda:0",
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    class_weight: Optional[np.ndarray] = None,
) -> nn.Linear:
    """
    Train a linear classification head on extracted features.
    
    Args:
        Z: Feature array of shape (N, D)
        Y: Label array of shape (N,)
        num_classes: Number of classes
        epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device for training
        momentum: SGD momentum
        weight_decay: Weight decay
        class_weight: Optional per-class weights for loss
    
    Returns:
        Trained linear head
    """
    device = torch.device(device)
    feature_dim = Z.shape[1]
    
    # Create head
    head = nn.Linear(feature_dim, num_classes).to(device)
    
    # Setup loss with optional class weights
    if class_weight is not None:
        weight_tensor = torch.tensor(class_weight, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = optim.SGD(
        head.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    
    # Create data loader
    Z_tensor = torch.tensor(Z, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    dataset = TensorDataset(Z_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    head.train()
    for epoch in range(epochs):
        for batch_Z, batch_Y in loader:
            batch_Z = batch_Z.to(device)
            batch_Y = batch_Y.to(device)
            
            optimizer.zero_grad()
            logits = head(batch_Z)
            loss = criterion(logits, batch_Y)
            loss.backward()
            optimizer.step()
    
    head.eval()
    return head


def train_weighted_head(
    source_dset: Dataset,
    target_dset: Dataset,
    w_hat: np.ndarray,
    config: Config = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train a classifier on source domain with importance weighting and evaluate on target.
    
    Args:
        source_dset: Source domain dataset
        target_dset: Target domain dataset
        w_hat: Estimated importance weights
        config: Configuration object
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with evaluation metrics (acc, macro_f1)
    """
    if config is None:
        config = get_config()
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    device = torch.device(config.model.device)
    num_classes = config.dataset.num_classes
    
    # Extract features
    extractor = FeatureExtractor(config=config)
    Z_S, Y_S = extractor.extract_features(source_dset)
    Z_T, Y_T = extractor.extract_features(target_dset)
    
    # Compute empirical source distribution
    p_emp = np.bincount(Y_S.astype(int), minlength=num_classes).astype(float)
    p_emp = p_emp / max(p_emp.sum(), 1.0)
    
    # Compute class weights from w_hat
    # Weight for class c = w_hat[c] * (1 / p_emp[c]) normalized
    eps = 1e-12
    class_weight = w_hat.copy()
    # Normalize to have mean 1
    class_weight = class_weight / (class_weight.mean() + eps)
    
    finetune_mode = config.downstream.finetune_mode
    
    if finetune_mode == 1:
        # Head-only training
        head = _train_head(
            Z_S, Y_S, num_classes,
            epochs=config.downstream.epochs,
            lr=config.downstream.head_lr,
            batch_size=config.model.batch_size,
            device=str(device),
            momentum=config.downstream.momentum,
            weight_decay=config.downstream.weight_decay,
            class_weight=class_weight,
        )
        
        # Evaluate on target
        Z_T_tensor = torch.tensor(Z_T, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = head(Z_T_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
        
    elif finetune_mode == 2:
        # Layer4 + head finetuning (ResNet only)
        Z_S_l3, Y_S_l3 = extractor.extract_layer3_features(source_dset)
        Z_T_l3, _ = extractor.extract_layer3_features(target_dset)
        
        # Build layer4 + head model
        model = extractor.model
        layer4_head = nn.Sequential(
            model.layer4,
            model.avgpool,
            nn.Flatten(),
            nn.Linear(extractor.feature_dim, num_classes),
        ).to(device)
        
        # Setup weighted loss
        weight_tensor = torch.tensor(class_weight, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        
        optimizer = optim.SGD(
            layer4_head.parameters(),
            lr=config.downstream.backbone_lr,
            momentum=config.downstream.momentum,
            weight_decay=config.downstream.weight_decay,
        )
        
        # Training
        Z_tensor = torch.tensor(Z_S_l3, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_S_l3, dtype=torch.long)
        dataset = TensorDataset(Z_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=config.model.batch_size, shuffle=True)
        
        layer4_head.train()
        for epoch in range(config.downstream.epochs):
            for batch_Z, batch_Y in loader:
                batch_Z = batch_Z.to(device)
                batch_Y = batch_Y.to(device)
                
                optimizer.zero_grad()
                logits = layer4_head(batch_Z)
                loss = criterion(logits, batch_Y)
                loss.backward()
                optimizer.step()
        
        layer4_head.eval()
        
        # Evaluate
        Z_T_tensor = torch.tensor(Z_T_l3, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = layer4_head(Z_T_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
    
    else:
        # Full finetuning (mode 3)
        raise NotImplementedError("Full finetuning mode not yet implemented")
    
    # Compute metrics
    acc = float(np.mean(preds == Y_T))
    macro_f1 = float(f1_score(Y_T, preds, average='macro'))
    
    return {
        'acc': acc,
        'macro_f1': macro_f1,
    }


def train_source_classifier_for_bias(
    source_dset: Dataset,
    target_dset: Dataset,
    config: Config = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train a classifier on source domain without reweighting.
    Returns target logits for logit-bias evaluation.
    
    Args:
        source_dset: Source domain dataset
        target_dset: Target domain dataset
        config: Configuration object
        seed: Random seed
    
    Returns:
        Dictionary with logits_T and Y_T
    """
    if config is None:
        config = get_config()
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    device = torch.device(config.model.device)
    num_classes = config.dataset.num_classes
    
    # Extract features
    extractor = FeatureExtractor(config=config)
    Z_S, Y_S = extractor.extract_features(source_dset)
    Z_T, Y_T = extractor.extract_features(target_dset)
    
    # Train head without class weighting
    head = _train_head(
        Z_S, Y_S, num_classes,
        epochs=config.downstream.epochs,
        lr=config.downstream.head_lr,
        batch_size=config.model.batch_size,
        device=str(device),
        momentum=config.downstream.momentum,
        weight_decay=config.downstream.weight_decay,
        class_weight=None,
    )
    
    # Get target logits
    Z_T_tensor = torch.tensor(Z_T, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_T = head(Z_T_tensor).cpu().numpy()
    
    return {
        'logits_T': logits_T,
        'Y_T': Y_T,
    }


def evaluate_with_logit_bias(
    logits_T: np.ndarray,
    Y_T: np.ndarray,
    w_hat: np.ndarray,
    gamma: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Evaluate classifier with logit bias adjustment.
    
    Args:
        logits_T: Target logits of shape (N, K)
        Y_T: Target labels of shape (N,)
        w_hat: Importance weights of shape (K,)
        gamma: Scaling factor for bias
        eps: Small constant for numerical stability
    
    Returns:
        Dictionary with acc and macro_f1
    """
    # Compute bias
    w_safe = np.maximum(w_hat, eps)
    bias = gamma * np.log(w_safe)
    
    # Apply bias
    biased_logits = logits_T + bias[np.newaxis, :]
    
    # Predict
    preds = biased_logits.argmax(axis=1)
    
    # Compute metrics
    acc = float(np.mean(preds == Y_T))
    macro_f1 = float(f1_score(Y_T, preds, average='macro'))
    
    return {
        'acc': acc,
        'macro_f1': macro_f1,
    }
