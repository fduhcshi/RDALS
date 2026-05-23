"""
Label shift generation and dataset creation.
"""

from typing import Tuple, Optional, Dict, Any

import numpy as np
from torch.utils.data import Dataset, Subset

from ..config import Config, get_config
from .datasets import load_full_dataset, get_dataset_labels
from .cifar10c import CIFAR10CSubset


def generate_dirichlet_distribution(
    num_classes: int,
    alpha: float,
    min_prob: float = 0.01,
) -> np.ndarray:
    """
    Generate a probability distribution using Dirichlet distribution.
    
    Args:
        num_classes: Number of classes
        alpha: Dirichlet concentration parameter
        min_prob: Minimum probability for each class
    
    Returns:
        Probability distribution array of shape (num_classes,)
    """
    # Sample from Dirichlet
    p_raw = np.random.dirichlet([alpha] * num_classes)
    
    # Smooth the sampled distribution to enforce a minimum class probability.
    k = num_classes
    dist = (1.0 - k * min_prob) * p_raw + min_prob
    
    return dist


def generate_tweak_one_distribution(
    num_classes: int,
    target_label: int,
    rho: float,
    min_prob: float = 0.01,
) -> np.ndarray:
    """
    Generate a tweak-one label shift distribution.
    
    The target class gets probability rho, others share (1-rho) uniformly.
    
    Args:
        num_classes: Number of classes
        target_label: Target class index
        rho: Probability for target class
        min_prob: Minimum probability for each class
    
    Returns:
        Probability distribution array of shape (num_classes,)
    """
    # Assign rho to the selected class and distribute the remaining mass evenly.
    dist = np.full(num_classes, (1.0 - rho) / (num_classes - 1))
    dist[target_label] = rho
    # The construction already sums to one; only validate the minimum mass.
    if np.min(dist) < min_prob:
        raise ValueError(f"tweak_one distribution has class probability below {min_prob*100:.0f}%: min={np.min(dist):.4f}")
    return dist


def _allocate_counts(probs: np.ndarray, total_samples: int) -> np.ndarray:
    """
    Allocate sample counts per class based on probability distribution.
    Uses floor allocation and assigns the remaining samples by largest
    fractional remainder.
    
    Args:
        probs: Probability distribution
        total_samples: Total number of samples to allocate
    
    Returns:
        Array of sample counts per class
    """
    # Use floor counts, then distribute remaining samples by fractional part.
    desired = total_samples * probs
    n = np.floor(desired).astype(int)
    remaining = int(total_samples - np.sum(n))
    if remaining > 0:
        frac = desired - n
        order = np.argsort(-frac)  # descending order of fractional parts
        for idx in order:
            if remaining <= 0:
                break
            n[idx] += 1
            remaining -= 1
    return n


def _sample_indices_by_counts(
    labels: np.ndarray,
    counts: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Sample indices from labels according to per-class counts.
    Samples per-class indices according to the requested count vector.
    
    Args:
        labels: Array of labels
        counts: Number of samples to draw per class
        num_classes: Number of classes
    
    Returns:
        Array of sampled indices (shuffled)
    """
    idx_all = np.arange(len(labels))
    out = []
    for j in range(num_classes):
        need = int(counts[j])
        if need <= 0:
            continue
        pool = idx_all[labels == j]
        if pool.size == 0:
            raise ValueError(f"Class {j} has no samples in dataset, but need {need}")
        replace = need > pool.size  # allow replacement if needed
        chosen = np.random.choice(pool, need, replace=replace)
        out.extend(chosen)
    np.random.shuffle(out)
    return np.array(out, dtype=int)


def generate_shift_and_indices(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    config: Config = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate label shift distributions and sample indices.
    
    Args:
        train_labels: Labels of training set
        test_labels: Labels of test set
        config: Configuration object
    
    Returns:
        Tuple of (source_indices, target_indices, p_true, q_true)
    """
    if config is None:
        config = get_config()
    
    num_classes = config.dataset.num_classes
    shift_name = config.shift.name.lower()
    shift_domain = config.shift.domain.lower()
    min_prob = config.estimation.min_prob
    
    # Generate distributions based on shift type
    if shift_name == "dirichlet":
        shifted_dist = generate_dirichlet_distribution(
            num_classes=num_classes,
            alpha=config.shift.alpha,
            min_prob=min_prob,
        )
    elif shift_name == "tweak_one":
        shifted_dist = generate_tweak_one_distribution(
            num_classes=num_classes,
            target_label=config.shift.target_label,
            rho=config.shift.rho,
            min_prob=min_prob,
        )
    else:
        raise ValueError(f"Unknown shift type: {shift_name}")
    
    # Uniform distribution
    uniform_dist = np.full(num_classes, 1.0 / num_classes)
    
    # Assign source and target distributions
    if shift_domain == "source":
        p_true = shifted_dist
        q_true = uniform_dist
    else:  # target
        p_true = uniform_dist
        q_true = shifted_dist
    
    # Allocate counts
    source_counts = _allocate_counts(p_true, config.dataset.source_samples)
    target_counts = _allocate_counts(q_true, config.dataset.target_samples)
    
    # Sample indices
    source_indices = _sample_indices_by_counts(train_labels, source_counts, num_classes)
    target_indices = _sample_indices_by_counts(test_labels, target_counts, num_classes)
    
    return source_indices, target_indices, p_true, q_true


def create_label_shift_datasets(
    config: Config = None,
) -> Tuple[Dataset, Dataset, np.ndarray, np.ndarray]:
    """
    Create source and target datasets with label shift.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (source_dataset, target_dataset, p_true, q_true)
    """
    if config is None:
        config = get_config()
    
    # Note: Do NOT set random seed here - it should be managed by the caller
    # (e.g., metrics.py sets seed per trial)
    
    # Load full datasets
    train_dataset, test_dataset, num_classes = load_full_dataset(config)
    
    # Get labels
    train_labels = get_dataset_labels(train_dataset)
    test_labels = get_dataset_labels(test_dataset)
    
    # Generate shift and indices
    source_indices, target_indices, p_true, q_true = generate_shift_and_indices(
        train_labels, test_labels, config
    )
    
    # Create subset datasets
    source_dataset = Subset(train_dataset, source_indices)
    
    # Check if using CIFAR-10-C for target
    dataset_name = config.dataset.name.lower()
    if dataset_name == "cifar10c" or (
        hasattr(config, 'use_cifar10c_target') and config.use_cifar10c_target
    ):
        # Use CIFAR-10-C for target domain
        target_dataset = CIFAR10CSubset(
            root=config.paths.data_root,
            corruption=getattr(config, 'cifar10c_corruption', 'brightness'),
            severity=getattr(config, 'cifar10c_severity', 5),
            indices=target_indices,
        )
    else:
        target_dataset = Subset(test_dataset, target_indices)
    
    return source_dataset, target_dataset, p_true, q_true
