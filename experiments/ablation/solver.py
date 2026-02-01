"""
Ablation study on solver and projection decoupling.
Compares different combinations of projection (LDA, PCA) and solver (inverse, regularized).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

from src.config import Config, get_config, reload_config
from src.data.shift import create_label_shift_datasets
from src.models.extractor import FeatureExtractor


def _compute_p_empirical(y_s: np.ndarray, k: int) -> np.ndarray:
    """Compute empirical class proportions."""
    counts = np.bincount(y_s.astype(int), minlength=k).astype(float)
    return counts / max(counts.sum(), 1.0)


def _clip_and_renorm(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip negative values and renormalize."""
    q = np.maximum(q, 0.0)
    s = q.sum()
    if s > eps:
        q = q / s
    else:
        q = np.full_like(q, 1.0 / len(q))
    return q


def _solve_inverse(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve using pseudoinverse: Q = pinv(A) @ b."""
    # A: (k-1, k), b: (k-1,)
    q = np.linalg.pinv(A) @ b
    return _clip_and_renorm(q)


def _solve_regularized_socp(
    A: np.ndarray,
    b: np.ndarray,
    lambda_reg: float,
    q_init: np.ndarray,
) -> np.ndarray:
    """Solve using regularized SLSQP optimization."""
    k = int(A.shape[1])
    eps = 1e-12
    
    def objective(Q: np.ndarray) -> float:
        residual = A @ Q - b  # A: (k-1, k), Q: (k,) -> residual: (k-1,)
        l2_loss = float(np.sqrt(np.sum(residual ** 2) + eps))
        reg = float(lambda_reg) * float(np.sqrt(np.sum(Q ** 2) + eps))
        return l2_loss + reg
    
    constraints = [{'type': 'eq', 'fun': lambda Q: np.sum(Q) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(k)]
    
    result = minimize(
        objective,
        q_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': False},
    )
    
    return _clip_and_renorm(result.x)


class ProjectionBase:
    """Base class for projection methods."""
    
    def fit(self, z_s: np.ndarray, y_s: np.ndarray) -> None:
        raise NotImplementedError
    
    def transform(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def build_A_b(
        self,
        z_s: np.ndarray,
        y_s: np.ndarray,
        z_t: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build A matrix and b vector."""
        h_s = self.transform(z_s)
        h_t = self.transform(z_t)
        
        d = h_s.shape[1]
        A = np.zeros((d, k), dtype=float)
        for c in range(k):
            mask = (y_s == c)
            if mask.sum() > 0:
                A[:, c] = h_s[mask].mean(axis=0)
        
        b = h_t.mean(axis=0)
        return A, b


class PCAProjection(ProjectionBase):
    """PCA-based projection."""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
    
    def fit(self, z_s: np.ndarray, y_s: np.ndarray) -> None:
        z_scaled = self.scaler.fit_transform(z_s)
        self.pca.fit(z_scaled)
    
    def transform(self, z: np.ndarray) -> np.ndarray:
        z_scaled = self.scaler.transform(z)
        return self.pca.transform(z_scaled)


class LDAProjection(ProjectionBase):
    """LDA-based projection."""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis(
            n_components=n_components,
            solver='eigen',
            shrinkage='auto',
        )
    
    def fit(self, z_s: np.ndarray, y_s: np.ndarray) -> None:
        z_scaled = self.scaler.fit_transform(z_s)
        self.lda.fit(z_scaled, y_s)
    
    def transform(self, z: np.ndarray) -> np.ndarray:
        z_scaled = self.scaler.transform(z)
        return self.lda.transform(z_scaled)


def run_solver_projection_ablation(
    trials: int = None,
    config: Config = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run ablation study on solver and projection combinations.
    """
    if config is None:
        config = get_config()
    
    if trials is None:
        trials = config.experiment.num_trials
    
    num_classes = config.dataset.num_classes
    lambda_reg = config.estimation.regularizer_lambda
    
    projection_options = ['pca', 'lda']
    solver_options = ['inverse', 'regularized']
    
    all_results = []
    
    for i in range(trials):
        print(f"Trial {i + 1}/{trials}...")
        
        source_dset, target_dset, p_true, q_true = create_label_shift_datasets(config)
        
        # Extract features
        extractor = FeatureExtractor(config=config)
        Z_S, Y_S = extractor.extract_features(source_dset)
        Z_T, Y_T = extractor.extract_features(target_dset)
        
        p_emp = _compute_p_empirical(Y_S, num_classes)
        w_true = q_true / p_true
        
        for proj_name in projection_options:
            # Create projection
            if proj_name == 'pca':
                proj = PCAProjection(n_components=num_classes - 1)
            else:
                proj = LDAProjection(n_components=num_classes - 1)
            
            t0 = time.time()
            proj.fit(Z_S, Y_S)
            A, b = proj.build_A_b(Z_S, Y_S, Z_T, num_classes)
            proj_time = time.time() - t0
            
            for solver_name in solver_options:
                t0 = time.time()
                
                if solver_name == 'inverse':
                    q_hat = _solve_inverse(A, b)
                else:
                    q_init = np.full(num_classes, 1.0 / num_classes)
                    q_hat = _solve_regularized_socp(A, b, lambda_reg, q_init)
                
                solve_time = time.time() - t0
                
                # Compute metrics
                eps = 1e-12
                w_hat = q_hat / np.maximum(p_emp, eps)
                mse_w = float(np.mean((w_hat - w_true) ** 2))
                l1_q = float(np.sum(np.abs(q_hat - q_true)))
                
                all_results.append({
                    'trial': i + 1,
                    'projection': proj_name,
                    'solver': solver_name,
                    'mse_w': mse_w,
                    'l1_q': l1_q,
                    'proj_time': proj_time,
                    'solve_time': solve_time,
                })
        
        print("=" * 50)
    
    df_all = pd.DataFrame(all_results)
    
    # Aggregate by (projection, solver)
    summary_rows = []
    exclude_ratio = config.experiment.exclude_extreme_ratio
    
    for proj_name in projection_options:
        for solver_name in solver_options:
            mask = (df_all['projection'] == proj_name) & (df_all['solver'] == solver_name)
            values = df_all.loc[mask, 'mse_w'].values
            
            n_total = len(values)
            n_exclude = int(np.floor(exclude_ratio * n_total))
            
            if n_exclude > 0 and n_total > n_exclude:
                sorted_idx = np.argsort(values)
                values = values[sorted_idx[:-n_exclude]]
            
            summary_rows.append({
                'projection': proj_name.upper(),
                'solver': solver_name,
                'mse_mean': float(np.mean(values)) if len(values) > 0 else np.nan,
                'mse_std': float(np.std(values)) if len(values) > 0 else np.nan,
                'n_kept': len(values),
            })
    
    df_summary = pd.DataFrame(summary_rows)
    
    print("\n" + "=" * 60)
    print("Solver-Projection Ablation Summary:")
    print("-" * 60)
    print(df_summary.to_string(index=False))
    
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'solver_projection_ablation',
                'projections': projection_options,
                'solvers': solver_options,
                'trials': trials,
            }
        )
        
        # Save detailed results
        detail_path = results_dir / f"{config.dataset.name}_solver_ablation_detail.csv"
        df_all.to_csv(detail_path, index=False)
        
        # Save summary
        summary_path = results_dir / f"{config.dataset.name}_solver_ablation_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        
        print(f"\nResults saved to: {results_dir}")
    
    return df_summary


def main():
    parser = argparse.ArgumentParser(description='Solver-projection ablation study')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--trials', type=int, default=None, help='Number of trials')
    
    args = parser.parse_args()
    
    if args.config:
        config = reload_config(args.config)
    else:
        config = get_config()
    
    run_solver_projection_ablation(
        trials=args.trials,
        config=config,
    )


if __name__ == '__main__':
    main()
