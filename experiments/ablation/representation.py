"""
Ablation study on representation mapping techniques.
Compares LDA, PCA, and Random Projection.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

from src.config import Config, get_config, reload_config
from src.data.shift import create_label_shift_datasets
from src.models.extractor import FeatureExtractor
from src.models.classifier import train_weighted_head
from src.methods.rdals import LabelShiftEstimator
from src.methods.baselines import clear_stat_cache


class PCAProjectionEstimator(LabelShiftEstimator):
    """Label shift estimator using PCA projection instead of LDA."""
    
    def __init__(self, n_components: int, regularizer_lambda: float = 0.01):
        self.k = n_components + 1
        self.n_components = n_components
        self.lambda_reg = regularizer_lambda
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.p_true = None
    
    def fit_projection(self, Z_S: np.ndarray, Y_S: np.ndarray = None) -> None:
        """Fit PCA on source features (labels not used)."""
        Z_scaled = self.scaler.fit_transform(Z_S)
        self.pca.fit(Z_scaled)
    
    def _project(self, Z: np.ndarray) -> np.ndarray:
        """Project features using PCA."""
        Z_scaled = self.scaler.transform(Z)
        return self.pca.transform(Z_scaled)
    
    def _build_A(self, Z_S: np.ndarray, Y_S: np.ndarray) -> np.ndarray:
        """Build A matrix using PCA projection."""
        H_S = self._project(Z_S)
        k = self.k
        d = H_S.shape[1]
        A = np.zeros((d, k), dtype=float)
        for c in range(k):
            mask = (Y_S == c)
            if mask.sum() > 0:
                A[:, c] = H_S[mask].mean(axis=0)
        return A
    
    def _build_b(self, Z_T: np.ndarray) -> np.ndarray:
        """Build b vector using PCA projection."""
        H_T = self._project(Z_T)
        return H_T.mean(axis=0)


class RandomProjectionEstimator(LabelShiftEstimator):
    """Label shift estimator using random orthogonal projection."""
    
    def __init__(self, n_components: int, regularizer_lambda: float = 0.01):
        self.k = n_components + 1
        self.n_components = n_components
        self.lambda_reg = regularizer_lambda
        self.scaler = StandardScaler()
        self.W = None
        self.p_true = None
    
    def fit_projection(self, Z_S: np.ndarray, Y_S: np.ndarray = None, random_state: int = None) -> None:
        """Fit random projection on source features."""
        if random_state is not None:
            np.random.seed(random_state)
        
        Z_scaled = self.scaler.fit_transform(Z_S)
        d_in = Z_scaled.shape[1]
        
        # Generate random Gaussian matrix and orthogonalize
        G = np.random.randn(d_in, self.n_components)
        Q, _ = np.linalg.qr(G)
        self.W = Q
    
    def _project(self, Z: np.ndarray) -> np.ndarray:
        """Project features using random projection."""
        Z_scaled = self.scaler.transform(Z)
        return Z_scaled @ self.W
    
    def _build_A(self, Z_S: np.ndarray, Y_S: np.ndarray) -> np.ndarray:
        """Build A matrix using random projection."""
        H_S = self._project(Z_S)
        k = self.k
        d = H_S.shape[1]
        A = np.zeros((d, k), dtype=float)
        for c in range(k):
            mask = (Y_S == c)
            if mask.sum() > 0:
                A[:, c] = H_S[mask].mean(axis=0)
        return A
    
    def _build_b(self, Z_T: np.ndarray) -> np.ndarray:
        """Build b vector using random projection."""
        H_T = self._project(Z_T)
        return H_T.mean(axis=0)


def pca_projection_method(
    source_dset,
    target_dset,
    p_true: np.ndarray,
    q_true: np.ndarray,
    train_downstream: bool = False,
    config: Config = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """PCA-based label shift estimation."""
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
    
    # Initialize PCA estimator
    estimator = PCAProjectionEstimator(
        n_components=num_classes - 1,
        regularizer_lambda=config.estimation.regularizer_lambda,
    )
    
    # Fit and estimate
    estimator.fit_projection(Z_S)
    
    if config.estimation.q_init_type == 2:
        estimator.p_true = p_true
    
    A = estimator._build_A(Z_S, Y_S)
    b = estimator._build_b(Z_T)
    q_hat = estimator._solve_regularized(A, b)
    
    # Compute weights
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
    
    if train_downstream:
        downstream_result = train_weighted_head(
            source_dset, target_dset, w_hat, config=config, seed=seed,
        )
        result['acc'] = downstream_result['acc']
        result['macro_f1'] = downstream_result['macro_f1']
    
    return result


def random_projection_method(
    source_dset,
    target_dset,
    p_true: np.ndarray,
    q_true: np.ndarray,
    train_downstream: bool = False,
    config: Config = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Random projection-based label shift estimation."""
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
    
    # Initialize random projection estimator
    estimator = RandomProjectionEstimator(
        n_components=num_classes - 1,
        regularizer_lambda=config.estimation.regularizer_lambda,
    )
    
    # Fit and estimate
    estimator.fit_projection(Z_S, random_state=seed)
    
    if config.estimation.q_init_type == 2:
        estimator.p_true = p_true
    
    A = estimator._build_A(Z_S, Y_S)
    b = estimator._build_b(Z_T)
    q_hat = estimator._solve_regularized(A, b)
    
    # Compute weights
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
    
    if train_downstream:
        downstream_result = train_weighted_head(
            source_dset, target_dset, w_hat, config=config, seed=seed,
        )
        result['acc'] = downstream_result['acc']
        result['macro_f1'] = downstream_result['macro_f1']
    
    return result


def run_representation_ablation(
    trials: int = None,
    train_downstream: bool = False,
    config: Config = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run ablation study comparing LDA, PCA, and Random Projection.
    """
    if config is None:
        config = get_config()
    
    if trials is None:
        trials = config.experiment.num_trials
    
    from src.methods.rdals import rdals_method
    
    methods = {
        'LDA (Ours)': rdals_method,
        'PCA': pca_projection_method,
        'Random': random_projection_method,
    }
    
    # Collect results
    per_trial: Dict[str, List[float]] = {name: [] for name in methods}
    
    for i in range(trials):
        print(f"Trial {i + 1}/{trials}...")
        
        source_dset, target_dset, p_true, q_true = create_label_shift_datasets(config)
        
        for name, method_fn in methods.items():
            try:
                result = method_fn(
                    source_dset, target_dset, p_true, q_true,
                    train_downstream=train_downstream,
                    config=config,
                )
                
                w_hat = np.array(result['w_hat'], dtype=float)
                w_true = np.array(result['w_true'], dtype=float)
                mse = float(np.mean((w_hat - w_true) ** 2))
                per_trial[name].append(mse)
                
            except Exception as e:
                print(f"  [WARN] {name} failed: {e}")
                per_trial[name].append(np.nan)
        
        print("=" * 50)
    
    # Aggregate
    exclude_ratio = config.experiment.exclude_extreme_ratio
    summary_rows = []
    
    for name in methods:
        values = np.array([v for v in per_trial[name] if np.isfinite(v)], dtype=float)
        n_total = len(values)
        n_exclude = int(np.floor(exclude_ratio * n_total))
        
        if n_exclude > 0 and n_total > n_exclude:
            sorted_idx = np.argsort(values)
            values = values[sorted_idx[:-n_exclude]]
        
        summary_rows.append({
            'method': name,
            'mse_mean': float(np.mean(values)) if len(values) > 0 else np.nan,
            'mse_std': float(np.std(values)) if len(values) > 0 else np.nan,
            'n_kept': len(values),
        })
    
    df = pd.DataFrame(summary_rows)
    
    print("\n" + "=" * 60)
    print("Representation Ablation Summary:")
    print("-" * 60)
    print(df.to_string(index=False))
    
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'representation_ablation',
                'methods': list(methods.keys()),
                'trials': trials,
            }
        )
        
        csv_path = results_dir / f"{config.dataset.name}_representation_ablation.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Representation mapping ablation study')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--trials', type=int, default=None, help='Number of trials')
    parser.add_argument('--downstream', action='store_true', help='Train downstream classifier')
    
    args = parser.parse_args()
    
    if args.config:
        config = reload_config(args.config)
    else:
        config = get_config()
    
    run_representation_ablation(
        trials=args.trials,
        train_downstream=args.downstream,
        config=config,
    )


if __name__ == '__main__':
    main()
