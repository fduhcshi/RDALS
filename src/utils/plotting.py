"""
Plotting utilities for label shift experiments.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set matplotlib defaults
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 6


# Method display names
LEGEND_ALIAS: Dict[str, str] = {
    'rdals': 'RDALS (Ours)',
    'rlls': 'RLLS',
    'bbsl': 'BBSL',
    'mlls': 'MLLS',
    'cpmcn': 'CPMCN',
    'naive': 'Naive',
    'oracle': 'Oracle',
}

# Markers for each method
MARKER_MAP: Dict[str, str] = {
    'rdals': 's',
    'rlls': 'o',
    'bbsl': '^',
    'mlls': 'D',
    'cpmcn': 'v',
    'naive': 'x',
    'oracle': '*',
}

# Line styles
LINESTYLE_MAP: Dict[str, str] = {
    'rdals': '-',
    'rlls': '--',
    'bbsl': '-.',
    'mlls': ':',
    'cpmcn': '--',
    'naive': '-.',
    'oracle': ':',
}

# Colors
COLOR_MAP: Dict[str, str] = {
    'rdals': '#d62728',  # Red
    'rlls': '#1f77b4',   # Blue
    'bbsl': '#2ca02c',   # Green
    'mlls': '#ff7f0e',   # Orange
    'cpmcn': '#9467bd',  # Purple
    'naive': '#8c564b',  # Brown
    'oracle': '#7f7f7f', # Gray
}


def plot_w_mse(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str] = None,
    show_std: bool = True,
    yscale: str = "linear",
    symlog_linthresh: float = 1e-4,
    ax: plt.Axes = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Axes:
    """
    Plot MSE curves for multiple methods.
    
    Args:
        df: DataFrame with x values and method columns
        x_col: Column name for x-axis
        y_cols: List of column names for y-axis (methods)
        show_std: Whether to show standard deviation as shaded area
        yscale: Y-axis scale ('linear', 'log', 'symlog')
        symlog_linthresh: Linear threshold for symlog scale
        ax: Matplotlib axes (created if None)
        figsize: Figure size
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if y_cols is None:
        # Auto-detect method columns
        y_cols = [c for c in df.columns if c != x_col and not c.endswith('_std')]
    
    x = df[x_col].values
    
    for method in y_cols:
        if method not in df.columns:
            continue
        
        y = df[method].values
        
        label = LEGEND_ALIAS.get(method, method)
        marker = MARKER_MAP.get(method, 'o')
        linestyle = LINESTYLE_MAP.get(method, '-')
        color = COLOR_MAP.get(method, None)
        
        ax.plot(
            x, y,
            label=label,
            marker=marker,
            linestyle=linestyle,
            color=color,
        )
        
        # Show standard deviation
        if show_std:
            std_col = f"{method}_std"
            if std_col in df.columns:
                std = df[std_col].values
                ax.fill_between(
                    x,
                    y - std,
                    y + std,
                    alpha=0.2,
                    color=color,
                )
    
    ax.set_xlabel(x_col)
    ax.set_ylabel('MSE')
    
    if yscale == 'symlog':
        ax.set_yscale('symlog', linthresh=symlog_linthresh)
    elif yscale != 'linear':
        ax.set_yscale(yscale)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax


def plot_backbone_histogram(
    df: pd.DataFrame,
    ax: plt.Axes = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Axes:
    """
    Plot MSE histogram grouped by backbone and method.
    
    Args:
        df: DataFrame with 'backbone', 'method', 'MSE' columns
        ax: Matplotlib axes
        figsize: Figure size
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique backbones and methods
    backbones = df['backbone'].unique()
    methods = df['method'].unique()
    
    n_backbones = len(backbones)
    n_methods = len(methods)
    
    bar_width = 0.8 / n_methods
    x = np.arange(n_backbones)
    
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        values = []
        for backbone in backbones:
            row = method_data[method_data['backbone'] == backbone]
            if len(row) > 0:
                values.append(float(row['MSE'].iloc[0]))
            else:
                values.append(0)
        
        color = COLOR_MAP.get(method, None)
        label = LEGEND_ALIAS.get(method, method)
        
        ax.bar(
            x + i * bar_width - 0.4 + bar_width / 2,
            values,
            bar_width,
            label=label,
            color=color,
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(backbones)
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    return ax


def plot_boxplot(
    data: Dict[str, List[float]],
    ax: plt.Axes = None,
    figsize: Tuple[float, float] = (8, 5),
    show_fliers: bool = False,
) -> plt.Axes:
    """
    Plot boxplot for multiple methods.
    
    Args:
        data: Dictionary mapping method names to lists of values
        ax: Matplotlib axes
        figsize: Figure size
        show_fliers: Whether to show outliers
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    methods = list(data.keys())
    values = [data[m] for m in methods]
    
    bp = ax.boxplot(values, patch_artist=True, showfliers=show_fliers)
    
    # Color boxes
    for i, (patch, method) in enumerate(zip(bp['boxes'], methods)):
        color = COLOR_MAP.get(method, '#1f77b4')
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Set labels
    labels = [LEGEND_ALIAS.get(m, m) for m in methods]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('MSE')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    return ax


def save_figure(
    fig: plt.Figure,
    path: str,
    dpi: int = 150,
    bbox_inches: str = 'tight',
    pad_inches: float = 0.02,
) -> None:
    """Save figure to file."""
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    print(f"Saved figure to: {path}")
