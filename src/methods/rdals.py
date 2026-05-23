"""
RDALS: Label shift estimation via LDA projection and regularized least squares.
"""

import time
from typing import Dict, Any, Optional

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from ..config import Config, get_config
from ..models.extractor import FeatureExtractor
from ..models.classifier import train_weighted_head


class LabelShiftEstimator:
    """
    Label shift estimator using LDA projection and regularized least squares.
    
    The method consists of:
    1. Extract features using pre-trained model
    2. Fit LDA on source domain for dimensionality reduction
    3. Build A matrix (class-conditional means in LDA space)
    4. Build b vector (global mean of target in LDA space)
    5. Solve regularized optimization: min ||Aq - b||^2 + lambda * ||q||^2
    """
    
    def __init__(self, lda_components: int, regularizer_lambda: float = 0.01):
        """
        Initialize estimator.
        
        Args:
            lda_components: Number of LDA components (typically num_classes - 1)
            regularizer_lambda: Regularization strength
        """
        self.k = lda_components + 1  # Number of classes
        self.lambda_reg = regularizer_lambda
        
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis(
            n_components=lda_components,
            solver='eigen',
            shrinkage='auto',
        )
        
        self.p_true = None  # Can be set for initialization
    
    def fit_lda(self, Z_S: np.ndarray, Y_S: np.ndarray) -> None:
        """
        Fit LDA on source domain features.
        
        Args:
            Z_S: Source features of shape (N, D)
            Y_S: Source labels of shape (N,)
        """
        Z_scaled = self.scaler.fit_transform(Z_S)
        self.lda.fit(Z_scaled, Y_S)
    
    def _build_A(self, Z_S: np.ndarray, Y_S: np.ndarray) -> np.ndarray:
        """
        Build A matrix: class-conditional means in LDA space.
        
        Args:
            Z_S: Source features
            Y_S: Source labels
        
        Returns:
            A matrix of shape (lda_components, k)
        """
        Z_scaled = self.scaler.transform(Z_S)
        H_S = self.lda.transform(Z_scaled)
        
        k = self.k
        d = H_S.shape[1]
        A = np.zeros((d, k), dtype=float)
        
        for c in range(k):
            mask = (Y_S == c)
            if mask.sum() > 0:
                A[:, c] = H_S[mask].mean(axis=0)
        
        return A
    
    def _build_b(self, Z_T: np.ndarray) -> np.ndarray:
        """
        Build b vector: global mean of target in LDA space.
        
        Args:
            Z_T: Target features
        
        Returns:
            b vector of shape (lda_components,)
        """
        Z_scaled = self.scaler.transform(Z_T)
        H_T = self.lda.transform(Z_scaled)
        return H_T.mean(axis=0)
    
    def _solve_regularized(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the regularized optimization problem.
        
        Args:
            A: Matrix of shape (d, k)
            b: Vector of shape (d,)
        
        Returns:
            Estimated distribution q_hat of shape (k,)
        """
        k = self.k
        
        # Use the unsquared L2 objective used by the RDALS estimator.
        def objective(Q: np.ndarray) -> float:
            residual = A @ Q - b
            eps = 1e-12
            l2_loss = np.sqrt(np.sum(residual ** 2) + eps)
            reg = self.lambda_reg * np.sqrt(np.sum(Q ** 2) + eps)
            return l2_loss + reg
        
        # Constraints: sum(Q) = 1, Q_j >= 0
        constraints = [{'type': 'eq', 'fun': lambda Q: np.sum(Q) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(k)]
        
        # Initial guess
        if self.p_true is not None:
            Q_init = self.p_true.copy()
        else:
            Q_init = np.full(k, 1.0 / k, dtype=float)
        
        result = minimize(
            objective,
            Q_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False},
        )
        
        Q_hat = result.x
        Q_hat = np.maximum(Q_hat, 0.0)
        s = Q_hat.sum()
        if s > 0:
            Q_hat = Q_hat / s
        
        return Q_hat
    
    def estimate_proportions(
        self,
        Z_S: np.ndarray,
        Y_S: np.ndarray,
        Z_T: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate target class proportions.
        
        Args:
            Z_S: Source features
            Y_S: Source labels
            Z_T: Target features
        
        Returns:
            Estimated target distribution q_hat
        """
        A = self._build_A(Z_S, Y_S)
        b = self._build_b(Z_T)
        return self._solve_regularized(A, b)


def rdals_method(
    source_dset: Dataset,
    target_dset: Dataset,
    p_true: np.ndarray,
    q_true: np.ndarray,
    train_downstream: bool = False,
    config: Config = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    RDALS method for label shift estimation.
    
    Args:
        source_dset: Source domain dataset
        target_dset: Target domain dataset
        p_true: True source label distribution
        q_true: True target label distribution
        train_downstream: Whether to train downstream classifier
        config: Configuration object
        seed: Random seed
    
    Returns:
        Dictionary with estimation results
    """
    if config is None:
        config = get_config()
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    t0 = time.time()
    
    device = torch.device(config.model.device)
    num_classes = config.dataset.num_classes
    
    # Extract features
    extractor = FeatureExtractor(config=config)
    Z_S, Y_S = extractor.extract_features(source_dset)
    Z_T, Y_T = extractor.extract_features(target_dset)
    
    # Initialize estimator
    estimator = LabelShiftEstimator(
        lda_components=num_classes - 1,
        regularizer_lambda=config.estimation.regularizer_lambda,
    )
    
    # Fit LDA on source
    estimator.fit_lda(Z_S, Y_S)
    
    # Set p_true for initialization if specified
    if config.estimation.q_init_type == 2:
        estimator.p_true = p_true
    
    # Estimate proportions
    q_hat = estimator.estimate_proportions(Z_S, Y_S, Z_T)
    
    # Compute importance weights
    p_empirical = np.bincount(Y_S.astype(int), minlength=num_classes).astype(float)
    p_empirical = p_empirical / max(p_empirical.sum(), 1.0)
    
    eps = 1e-12
    w_hat = q_hat / np.maximum(p_empirical, eps)
    w_true = q_true / p_true
    
    time_sec = time.time() - t0
    
    result = {
        'q_hat': q_hat,
        'w_hat': w_hat,
        'w_true': w_true,
        'time_sec': time_sec,
    }
    
    # Optional downstream training
    if train_downstream:
        downstream_result = train_weighted_head(
            source_dset, target_dset, w_hat,
            config=config, seed=seed,
        )
        result['acc'] = downstream_result['acc']
        result['macro_f1'] = downstream_result['macro_f1']
    
    return result
