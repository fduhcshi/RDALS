"""
Evaluation metrics for label shift estimation.
"""

from typing import Dict, Any, List, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import Config, get_config
from ..data.shift import create_label_shift_datasets
from ..methods.rdals import rdals_method
from ..methods.baselines import (
    bs_rlls, bs_bbsl, bs_mlls, bs_cpmcn, bs_naive, bs_oracle, clear_stat_cache
)


# Method registry
METHODS: Dict[str, Callable] = {
    'rdals': rdals_method,
    'rlls': bs_rlls,
    'bbsl': bs_bbsl,
    'mlls': bs_mlls,
    'cpmcn': bs_cpmcn,
    'naive': bs_naive,
    'oracle': bs_oracle,
}


def compute_mse(w_hat: np.ndarray, w_true: np.ndarray) -> float:
    """Compute mean squared error between estimated and true weights."""
    return float(np.mean((w_hat - w_true) ** 2))


def compute_l1_error(q_hat: np.ndarray, q_true: np.ndarray) -> float:
    """Compute L1 error between estimated and true distributions."""
    return float(np.sum(np.abs(q_hat - q_true)))


def compute_w_emse(
    methods: List[str] = None,
    trials: int = None,
    seed: int = None,
    train_downstream: bool = False,
    config: Config = None,
    dataset_sampler: Callable = None,
) -> Dict[str, Any]:
    """
    Compute EMSE (Expected Mean Squared Error) for multiple methods.
    
    Args:
        methods: List of method names to evaluate
        trials: Number of trials
        seed: Random seed (trial i uses seed+i for reproducibility)
        train_downstream: Whether to train downstream classifier
        config: Configuration object
        dataset_sampler: Optional custom dataset sampler function
    
    Returns:
        Dictionary with per-method statistics
    """
    if config is None:
        config = get_config()
    
    if trials is None:
        trials = config.experiment.num_trials
    
    if methods is None:
        methods = ['rdals', 'rlls', 'bbsl', 'mlls']
    
    # Filter to available methods
    methods = [m for m in methods if m in METHODS]
    
    if dataset_sampler is None:
        def dataset_sampler():
            return create_label_shift_datasets(config)
    
    # Store per-trial results
    per_trial: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    
    for i in range(trials):
        # Set seed for this trial
        if seed is not None:
            trial_seed = seed + i
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)
            print(f"Trial {i + 1}/{trials} (seed={trial_seed})...")
        else:
            print(f"Trial {i + 1}/{trials}...")
        
        # Clear cache for each trial
        clear_stat_cache()
        
        # Sample new dataset
        source_dset, target_dset, p_true, q_true = dataset_sampler()
        
        for method_name in methods:
            method_fn = METHODS[method_name]
            try:
                result = method_fn(
                    source_dset, target_dset, p_true, q_true,
                    train_downstream=train_downstream,
                    config=config,
                )
                
                w_hat = np.array(result['w_hat'], dtype=float)
                w_true = np.array(result['w_true'], dtype=float)
                q_hat = np.array(result['q_hat'], dtype=float)
                
                mse = compute_mse(w_hat, w_true)
                l1_q = compute_l1_error(q_hat, q_true)
                
                trial_result = {
                    'mse': mse,
                    'l1_q': l1_q,
                    'time_sec': result.get('time_sec', 0.0),
                }
                
                if train_downstream:
                    trial_result['acc'] = result.get('acc', np.nan)
                    trial_result['macro_f1'] = result.get('macro_f1', np.nan)
                
                per_trial[method_name].append(trial_result)
                
            except Exception as e:
                print(f"  [WARN] Method {method_name} failed: {e}")
                per_trial[method_name].append({
                    'mse': np.nan,
                    'l1_q': np.nan,
                    'time_sec': np.nan,
                })
        
        print("=" * 50)
    
    # Aggregate results
    exclude_ratio = config.experiment.exclude_extreme_ratio
    summary = {}
    
    for method_name in methods:
        results = per_trial[method_name]
        mse_values = np.array([r['mse'] for r in results], dtype=float)
        
        # Filter out NaN
        valid_mask = np.isfinite(mse_values)
        valid_mse = mse_values[valid_mask]
        
        n_total = len(valid_mse)
        n_exclude = int(np.floor(exclude_ratio * n_total))
        
        if n_exclude > 0 and n_total > n_exclude:
            # Exclude worst results
            sorted_idx = np.argsort(valid_mse)
            keep_idx = sorted_idx[:-n_exclude]
            kept_mse = valid_mse[keep_idx]
        else:
            kept_mse = valid_mse
        
        if len(kept_mse) > 0:
            summary[method_name] = {
                'mse_mean': float(np.mean(kept_mse)),
                'mse_std': float(np.std(kept_mse)),
                'mse_median': float(np.median(kept_mse)),
                'n_total': n_total,
                'n_kept': len(kept_mse),
            }
        else:
            summary[method_name] = {
                'mse_mean': np.nan,
                'mse_std': np.nan,
                'mse_median': np.nan,
                'n_total': 0,
                'n_kept': 0,
            }
        
        # Add downstream metrics if available
        if train_downstream:
            acc_values = np.array([r.get('acc', np.nan) for r in results], dtype=float)
            valid_acc = acc_values[np.isfinite(acc_values)]
            if len(valid_acc) > 0:
                summary[method_name]['acc_mean'] = float(np.mean(valid_acc))
                summary[method_name]['acc_std'] = float(np.std(valid_acc))
    
    return {
        'summary': summary,
        'per_trial': per_trial,
    }


def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert results dictionary to DataFrame."""
    summary = results['summary']
    rows = []
    for method_name, stats in summary.items():
        row = {'method': method_name}
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows)
