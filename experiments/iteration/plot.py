"""
Plotting utilities for iteration analysis.
"""

from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def plot_iteration_curves(
    df: pd.DataFrame,
    max_iters: int = 10,
    figsize: tuple = (6, 4),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot iteration vs MSE curves.
    
    Args:
        df: DataFrame with columns: iter, rdals, mlls, cpmcn
        max_iters: Maximum number of iterations to show
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_full = df['iter'].values
    max_points = min(max_iters, len(x_full))
    x = x_full[:max_points]
    
    method_config = {
        'rdals': {'label': 'RDALS', 'color': '#d62728', 'linestyle': '-', 'marker': 's'},
        'mlls': {'label': 'MLLS', 'color': '#1f77b4', 'linestyle': '--', 'marker': 'o'},
        'cpmcn': {'label': 'CPMCN', 'color': '#ff7f0e', 'linestyle': ':', 'marker': '^'},
    }
    
    for name, cfg in method_config.items():
        if name not in df.columns:
            continue
        y_full = df[name].values
        y = y_full[:max_points]
        ax.plot(
            x, y,
            label=cfg['label'],
            color=cfg['color'],
            linestyle=cfg['linestyle'],
            marker=cfg['marker'],
            markersize=4,
        )
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved figure to: {save_path}")
    
    return ax
