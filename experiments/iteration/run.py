"""
Run iteration analysis experiments.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse

import matplotlib.pyplot as plt

from src.config import Config, get_config, reload_config

from .aggregate import compute_iteration_mse_trials
from .plot import plot_iteration_curves


def run_iteration_analysis(
    trials: int = None,
    max_iters: int = 10,
    config: Config = None,
    save_results: bool = True,
) -> None:
    """
    Run iteration analysis experiment.
    
    Compares convergence speed of RDALS, MLLS, and CPMCN.
    """
    if config is None:
        config = get_config()
    
    if trials is None:
        trials = config.experiment.num_trials
    
    print(f"Running iteration analysis with {trials} trials...")
    print(f"Dataset: {config.dataset.name}")
    print("=" * 60)
    
    # Compute aggregated iteration paths
    df = compute_iteration_mse_trials(trials=trials, config=config)
    
    print("\n" + "=" * 60)
    print("Aggregated Iteration Curves (first 10 iterations):")
    print("-" * 60)
    print(df.head(10).to_string(index=False))
    
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'iteration_analysis',
                'methods': ['rdals', 'mlls', 'cpmcn'],
                'trials': trials,
            }
        )
        
        # Save CSV
        csv_path = results_dir / f"{config.dataset.name}_iteration_paths.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Plot
        fig_path = results_dir / f"{config.dataset.name}_iteration_curves.pdf"
        plot_iteration_curves(df, max_iters=max_iters, save_path=str(fig_path))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Iteration analysis experiment')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--trials', type=int, default=None, help='Number of trials')
    parser.add_argument('--max-iters', type=int, default=10, help='Max iterations to plot')
    
    args = parser.parse_args()
    
    if args.config:
        config = reload_config(args.config)
    else:
        config = get_config()
    
    run_iteration_analysis(
        trials=args.trials,
        max_iters=args.max_iters,
        config=config,
    )


if __name__ == '__main__':
    main()
