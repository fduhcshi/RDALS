"""
Aggregation utilities for iteration analysis.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import Config, get_config
from src.data.shift import create_label_shift_datasets
from src.methods.baselines import clear_stat_cache

from .runners import run_rdals_iteration, run_mlls_iteration, run_cpmcn_iteration


def compute_iteration_mse_trials(
    trials: int = None,
    config: Config = None,
) -> pd.DataFrame:
    """
    Run multiple trials and aggregate iteration-vs-MSE paths.
    
    For each method, drops worst 10% of trials (by final MSE) before averaging.
    
    Returns:
        DataFrame with columns: iter, rdals, mlls, cpmcn
    """
    if config is None:
        config = get_config()
    
    if trials is None:
        trials = config.experiment.num_trials
    
    methods = ['rdals', 'mlls', 'cpmcn']
    runners = {
        'rdals': run_rdals_iteration,
        'mlls': run_mlls_iteration,
        'cpmcn': run_cpmcn_iteration,
    }
    
    paths_by_method: Dict[str, List[List[float]]] = {m: [] for m in methods}
    finals_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    
    for i in range(trials):
        print(f"Trial {i + 1}/{trials} (iteration curves)...")
        
        clear_stat_cache()
        source_dset, target_dset, p_true, q_true = create_label_shift_datasets(config)
        
        for name in methods:
            try:
                history = runners[name](source_dset, target_dset, p_true, q_true, config)
                paths_by_method[name].append(history)
                final = float(history[-1]) if len(history) > 0 else float('nan')
                finals_by_method[name].append(final)
            except Exception as e:
                print(f"  [WARN] {name} failed: {e}")
                paths_by_method[name].append([])
                finals_by_method[name].append(float('nan'))
        
        print("=" * 50)
    
    # Aggregate with trimming
    mean_paths: Dict[str, List[float]] = {}
    
    for m in methods:
        all_paths = paths_by_method[m]
        finals = np.array(finals_by_method[m], dtype=float)
        
        # Valid trial indices
        valid_idx = [
            idx for idx, (path, f) in enumerate(zip(all_paths, finals))
            if len(path) > 0 and np.isfinite(f)
        ]
        
        if len(valid_idx) == 0:
            print(f"[WARN] Method {m} has no valid trials")
            continue
        
        # Drop worst 10%
        vals_valid = finals[valid_idx]
        k_drop = int(np.floor(0.10 * len(valid_idx)))
        kept_idx = list(valid_idx)
        
        if k_drop > 0:
            order = np.argsort(vals_valid)
            worst_rel = order[-k_drop:]
            worst_set = {valid_idx[j] for j in worst_rel}
            kept_idx = [idx for idx in valid_idx if idx not in worst_set]
        
        print(f"Method {m}: n_total={len(all_paths)}, n_kept={len(kept_idx)}")
        
        if len(kept_idx) == 0:
            continue
        
        # Average per iteration
        max_len = max(len(all_paths[idx]) for idx in kept_idx)
        agg = []
        for t in range(max_len):
            vals_t = [
                all_paths[idx][t] for idx in kept_idx
                if len(all_paths[idx]) > t
            ]
            if len(vals_t) == 0:
                agg.append(float('nan'))
            else:
                agg.append(float(np.mean(vals_t)))
        
        mean_paths[m] = agg
    
    # Build DataFrame
    max_len = max((len(v) for v in mean_paths.values()), default=0)
    data = {'iter': list(range(1, max_len + 1))}
    
    for m in methods:
        seq = mean_paths.get(m, [])
        data[m] = [seq[i] if i < len(seq) else np.nan for i in range(max_len)]
    
    return pd.DataFrame(data)
