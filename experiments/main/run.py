"""
Main experiment runner for label shift estimation.
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

from src.config import Config, get_config, reload_config
from src.evaluation.metrics import compute_w_emse, results_to_dataframe, METHODS


def run_experiment(
    config_path: str = None,
    methods: list = None,
    trials: int = None,
    seed: int = None,
    train_downstream: bool = None,
    save_results: bool = True,
) -> dict:
    """
    Run main label shift estimation experiment.
    
    Args:
        config_path: Path to config.yaml
        methods: List of method names to evaluate
        trials: Number of trials (overrides config)
        seed: Random seed (if set, trial i uses seed+i for reproducibility)
        train_downstream: Whether to train downstream classifier (None = use config)
        save_results: Whether to save results to file
    
    Returns:
        Results dictionary
    """
    # Load config
    if config_path:
        config = reload_config(config_path)
    else:
        config = get_config()
    
    if trials is not None:
        config.experiment.num_trials = trials
    
    # Use config setting if not specified via argument
    if train_downstream is None:
        train_downstream = config.downstream.train
    
    if methods is None:
        methods = ['rdals', 'rlls', 'bbsl', 'mlls', 'cpmcn']
    
    print(f"Running experiment with {len(methods)} methods, {config.experiment.num_trials} trials")
    print(f"Dataset: {config.dataset.name}")
    print(f"Shift: {config.shift.name} (alpha={config.shift.alpha})")
    print("=" * 60)
    
    # Run evaluation
    results = compute_w_emse(
        methods=methods,
        trials=config.experiment.num_trials,
        seed=seed,
        train_downstream=train_downstream,
        config=config,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("-" * 60)
    
    df = results_to_dataframe(results)
    print(df.to_string(index=False))
    
    # Save results
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'main',
                'methods': methods,
                'train_downstream': train_downstream,
            }
        )
        
        csv_path = results_dir / f"{config.dataset.name}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run label shift estimation experiment')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to evaluate (e.g., rdals rlls bbsl)')
    parser.add_argument('--trials', type=int, default=None, help='Number of trials')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (trial i uses seed+i)')
    parser.add_argument('--downstream', action='store_true', default=None,
                        help='Train downstream classifier (overrides config.yaml)')
    parser.add_argument('--no-downstream', action='store_true',
                        help='Disable downstream training (overrides config.yaml)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    # Determine downstream setting: CLI flag > config.yaml
    if args.downstream:
        train_downstream = True
    elif args.no_downstream:
        train_downstream = False
    else:
        train_downstream = None  # Will use config.yaml setting
    
    run_experiment(
        config_path=args.config,
        methods=args.methods,
        trials=args.trials,
        seed=args.seed,
        train_downstream=train_downstream,
        save_results=not args.no_save,
    )


if __name__ == '__main__':
    main()
