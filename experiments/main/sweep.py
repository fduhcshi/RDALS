"""
Parameter sweep experiments for label shift estimation.
Generates MSE curves across varying alpha, rho (tweak_one), or sample size.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config, get_config, reload_config
from src.data.shift import create_label_shift_datasets
from src.evaluation.metrics import compute_w_emse, METHODS
from src.methods.baselines import clear_stat_cache
from src.utils.plotting import plot_w_mse, save_figure


def sweep_alpha(
    alpha_values: List[float],
    methods: List[str] = None,
    trials: int = None,
    config: Config = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Sweep over Dirichlet alpha values.
    
    Args:
        alpha_values: List of alpha values to test
        methods: Methods to evaluate
        trials: Number of trials per alpha
        config: Configuration object
        save_results: Whether to save results
    
    Returns:
        DataFrame with results
    """
    if config is None:
        config = get_config()
    
    if methods is None:
        # Public sweep defaults track RDALS only. Pass --methods to include
        # optional baselines in comparison sweeps.
        methods = ['rdals']
    
    if trials is None:
        trials = config.experiment.num_trials
    
    all_results = []
    
    for alpha in alpha_values:
        print(f"\n{'=' * 60}")
        print(f"Alpha = {alpha}")
        print('=' * 60)
        
        # Update config
        config.shift.alpha = alpha
        
        # Create dataset sampler with current alpha
        def dataset_sampler():
            return create_label_shift_datasets(config)
        
        # Run evaluation
        results = compute_w_emse(
            methods=methods,
            trials=trials,
            config=config,
            dataset_sampler=dataset_sampler,
        )
        
        # Extract summary
        row = {'alpha': alpha}
        for method_name, stats in results['summary'].items():
            row[method_name] = stats['mse_mean']
            row[f'{method_name}_std'] = stats['mse_std']
        
        all_results.append(row)
        clear_stat_cache()
    
    df = pd.DataFrame(all_results)
    
    # Save and plot
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'alpha_sweep',
                'alpha_values': alpha_values,
                'methods': methods,
                'trials': trials,
            }
        )
        
        csv_path = results_dir / f"{config.dataset.name}_alpha_sweep.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_w_mse(
            df, 'alpha',
            y_cols=methods,
            show_std=config.plotting.show_std,
            yscale=config.plotting.y_scale,
            symlog_linthresh=config.plotting.y_symlog_linthresh,
            ax=ax,
        )
        ax.set_xlabel('Dirichlet Alpha')
        
        fig_path = results_dir / f"{config.dataset.name}_alpha_sweep.pdf"
        save_figure(fig, str(fig_path))
        plt.close(fig)
    
    return df


def sweep_rho(
    rho_values: List[float],
    methods: List[str] = None,
    trials: int = None,
    config: Config = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Sweep over tweak_one rho values.
    
    Args:
        rho_values: List of rho values to test (0-1, shift intensity)
        methods: Methods to evaluate
        trials: Number of trials per rho
        config: Configuration object
        save_results: Whether to save results
    
    Returns:
        DataFrame with results
    """
    if config is None:
        config = get_config()
    
    if methods is None:
        # Public sweep defaults track RDALS only. Pass --methods to include
        # optional baselines in comparison sweeps.
        methods = ['rdals']
    
    if trials is None:
        trials = config.experiment.num_trials
    
    # Set shift to tweak_one
    config.shift.name = 'tweak_one'
    
    all_results = []
    
    for rho in rho_values:
        print(f"\n{'=' * 60}")
        print(f"Rho = {rho}")
        print('=' * 60)
        
        # Update config
        config.shift.rho = rho
        
        # Create dataset sampler with current rho
        def dataset_sampler():
            return create_label_shift_datasets(config)
        
        # Run evaluation
        results = compute_w_emse(
            methods=methods,
            trials=trials,
            config=config,
            dataset_sampler=dataset_sampler,
        )
        
        # Extract summary
        row = {'rho': rho}
        for method_name, stats in results['summary'].items():
            row[method_name] = stats['mse_mean']
            row[f'{method_name}_std'] = stats['mse_std']
        
        all_results.append(row)
        clear_stat_cache()
    
    df = pd.DataFrame(all_results)
    
    # Save and plot
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'rho_sweep',
                'rho_values': rho_values,
                'methods': methods,
                'trials': trials,
            }
        )
        
        csv_path = results_dir / f"{config.dataset.name}_rho_sweep.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_w_mse(
            df, 'rho',
            y_cols=methods,
            show_std=config.plotting.show_std,
            yscale=config.plotting.y_scale,
            symlog_linthresh=config.plotting.y_symlog_linthresh,
            ax=ax,
        )
        ax.set_xlabel('Tweak-One Rho')
        
        fig_path = results_dir / f"{config.dataset.name}_rho_sweep.pdf"
        save_figure(fig, str(fig_path))
        plt.close(fig)
    
    return df


def sweep_sample_size(
    sample_sizes: List[int],
    methods: List[str] = None,
    trials: int = None,
    config: Config = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Sweep over target sample sizes.
    
    Args:
        sample_sizes: List of sample sizes to test
        methods: Methods to evaluate
        trials: Number of trials per size
        config: Configuration object
        save_results: Whether to save results
    
    Returns:
        DataFrame with results
    """
    if config is None:
        config = get_config()
    
    if methods is None:
        # Public sweep defaults track RDALS only. Pass --methods to include
        # optional baselines in comparison sweeps.
        methods = ['rdals']
    
    if trials is None:
        trials = config.experiment.num_trials
    
    all_results = []
    
    for n_samples in sample_sizes:
        print(f"\n{'=' * 60}")
        print(f"Target samples = {n_samples}")
        print('=' * 60)
        
        # Update config
        config.dataset.target_samples = n_samples
        
        # Create dataset sampler
        def dataset_sampler():
            return create_label_shift_datasets(config)
        
        # Run evaluation
        results = compute_w_emse(
            methods=methods,
            trials=trials,
            config=config,
            dataset_sampler=dataset_sampler,
        )
        
        # Extract summary
        row = {'n_samples': n_samples}
        for method_name, stats in results['summary'].items():
            row[method_name] = stats['mse_mean']
            row[f'{method_name}_std'] = stats['mse_std']
        
        all_results.append(row)
        clear_stat_cache()
    
    df = pd.DataFrame(all_results)
    
    # Save and plot
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'sample_size_sweep',
                'sample_sizes': sample_sizes,
                'methods': methods,
                'trials': trials,
            }
        )
        
        csv_path = results_dir / f"{config.dataset.name}_sample_size_sweep.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_w_mse(
            df, 'n_samples',
            y_cols=methods,
            show_std=config.plotting.show_std,
            yscale=config.plotting.y_scale,
            symlog_linthresh=config.plotting.y_symlog_linthresh,
            ax=ax,
        )
        ax.set_xlabel('Target Sample Size')
        
        fig_path = results_dir / f"{config.dataset.name}_sample_size_sweep.pdf"
        save_figure(fig, str(fig_path))
        plt.close(fig)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Parameter sweep experiments')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--sweep', type=str, choices=['alpha', 'rho', 'sample_size'], required=True,
                        help='Type of sweep: alpha (dirichlet), rho (tweak_one), or sample_size')
    parser.add_argument('--values', type=float, nargs='+', default=None,
                        help='Values to sweep over')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to evaluate (default: rdals; optional: rlls bbsl mlls cpmcn)')
    parser.add_argument('--trials', type=int, default=None, help='Number of trials')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = reload_config(args.config)
    else:
        config = get_config()
    
    if args.sweep == 'alpha':
        # Dirichlet alpha sweep
        values = args.values or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        config.shift.name = 'dirichlet'  # Ensure dirichlet shift
        sweep_alpha(
            alpha_values=values,
            methods=args.methods,
            trials=args.trials,
            config=config,
        )
    elif args.sweep == 'rho':
        # Tweak-one rho sweep
        values = args.values or [0.1, 0.3, 0.5, 0.7, 0.9]
        sweep_rho(
            rho_values=values,
            methods=args.methods,
            trials=args.trials,
            config=config,
        )
    else:
        # Sample size sweep
        values = args.values or [100, 500, 1000, 2000, 5000, 10000]
        values = [int(v) for v in values]
        sweep_sample_size(
            sample_sizes=values,
            methods=args.methods,
            trials=args.trials,
            config=config,
        )


if __name__ == '__main__':
    main()
