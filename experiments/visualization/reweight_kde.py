"""
KDE visualization of source reweighting in LDA projection space.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src.config import Config, get_config, reload_config
from src.data.shift import create_label_shift_datasets
from src.models.extractor import FeatureExtractor
from src.methods.rdals import rdals_method, LabelShiftEstimator


def plot_1d_kde_with_fill(
    h: np.ndarray,
    ax: plt.Axes,
    color: str,
    label: str = None,
    weights: np.ndarray = None,
    num_points: int = 256,
    x_grid: np.ndarray = None,
):
    """
    Plot KDE with filled area.
    
    Args:
        h: 1D data array
        ax: Matplotlib axes
        color: Line/fill color
        label: Legend label
        weights: Optional sample weights
        num_points: Number of grid points
        x_grid: Optional pre-defined x grid
    
    Returns:
        Tuple of (x_grid, y_kde)
    """
    h = np.asarray(h, dtype=float).ravel()
    
    if weights is not None:
        kde = gaussian_kde(h, weights=weights)
    else:
        kde = gaussian_kde(h)
    
    if x_grid is None:
        x_min, x_max = float(np.min(h)), float(np.max(h))
        pad = 0.05 * (x_max - x_min + 1e-12)
        x_grid = np.linspace(x_min - pad, x_max + pad, num_points)
    
    y_kde = kde(x_grid)
    
    ax.plot(x_grid, y_kde, color=color, label=label)
    ax.fill_between(x_grid, y_kde, color=color, alpha=0.1)
    
    return x_grid, y_kde


def run_reweight_visualization(
    seed: int = None,
    config: Config = None,
    save_results: bool = True,
) -> dict:
    """
    Visualize source reweighting effect in LDA projection space.
    
    Shows three distributions:
    - Source (unweighted)
    - Source (reweighted by w_hat)
    - Target
    """
    if config is None:
        config = get_config()
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create datasets
    source_dset, target_dset, p_true, q_true = create_label_shift_datasets(config)
    
    # Run method to get w_hat
    result = rdals_method(source_dset, target_dset, p_true, q_true, config=config)
    w_hat = result['w_hat']
    
    # Extract features
    extractor = FeatureExtractor(config=config)
    Z_S, Y_S = extractor.extract_features(source_dset)
    Z_T, Y_T = extractor.extract_features(target_dset)
    
    # Fit LDA and project
    estimator = LabelShiftEstimator(
        lda_components=config.dataset.num_classes - 1,
        regularizer_lambda=config.estimation.regularizer_lambda,
    )
    estimator.fit_lda(Z_S, Y_S)
    
    Z_S_std = estimator.scaler.transform(Z_S)
    Z_T_std = estimator.scaler.transform(Z_T)
    H_S = estimator.lda.transform(Z_S_std)
    H_T = estimator.lda.transform(Z_T_std)
    
    # First LDA component
    h_S = H_S[:, 0]
    h_T = H_T[:, 0]
    
    # Sample weights
    Y_S = np.asarray(Y_S, dtype=int)
    weights_S = w_hat[Y_S]
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Common x grid
    all_h = np.concatenate([h_S, h_T])
    x_min, x_max = float(np.min(all_h)), float(np.max(all_h))
    pad = 0.05 * (x_max - x_min + 1e-12)
    x_grid = np.linspace(x_min - pad, x_max + pad, 256)
    
    # Source unweighted
    _, y_source_unweighted = plot_1d_kde_with_fill(
        h_S, ax, color="#888888", label="Source (unweighted)",
        weights=None, x_grid=x_grid,
    )
    
    # Source reweighted
    _, y_source_reweighted = plot_1d_kde_with_fill(
        h_S, ax, color="#AA2B46", label="Source reweighted by $\\hat{w}$",
        weights=weights_S, x_grid=x_grid,
    )
    
    # Target
    _, y_target = plot_1d_kde_with_fill(
        h_T, ax, color="#3B5DA3", label="Target",
        weights=None, x_grid=x_grid,
    )
    
    ax.set_xlabel("$h_1(z)$", fontsize=16)
    ax.set_ylabel("pdf", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'reweight_visualization',
                'seed': seed,
            }
        )
        
        # Save figure
        fig_path = results_dir / f"{config.dataset.name}_projection_reweight.pdf"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved figure to: {fig_path}")
        
        # Save KDE data
        kde_df = pd.DataFrame({
            'x': x_grid,
            'source_unweighted': y_source_unweighted,
            'source_reweighted': y_source_reweighted,
            'target': y_target,
        })
        csv_path = results_dir / f"{config.dataset.name}_projection_reweight_kde.csv"
        kde_df.to_csv(csv_path, index=False)
        print(f"Saved KDE data to: {csv_path}")
    
    plt.close(fig)
    
    return {
        'h_S': h_S,
        'h_T': h_T,
        'w_hat': w_hat,
        'weights_S': weights_S,
    }


def main():
    parser = argparse.ArgumentParser(description='Reweighting visualization')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.config:
        config = reload_config(args.config)
    else:
        config = get_config()
    
    run_reweight_visualization(
        seed=args.seed,
        config=config,
    )


if __name__ == '__main__':
    main()
