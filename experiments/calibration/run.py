"""
Run calibration comparison experiments.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config, get_config, reload_config
from src.data.shift import create_label_shift_datasets
from src.methods.rdals import rdals_method
from src.methods.baselines import bs_mlls, bs_cpmcn, clear_stat_cache

from .runner import mlls_variant, cpmcn_variant
from .plot import plot_vertical_strip


def run_calibration_comparison(
    trials: int = None,
    config: Config = None,
    save_results: bool = True,
) -> Dict[str, any]:
    """
    Run calibration comparison experiment.
    
    Compares RDALS with MLLS and CPMCN using different calibration methods:
    - None: No calibration (softmax only)
    - TS: Temperature Scaling
    - NBVS: No-Bias Vector Scaling
    - BCTS: Bias-Corrected Temperature Scaling
    - VS: Vector Scaling
    """
    if config is None:
        config = get_config()
    
    if trials is None:
        trials = config.experiment.num_trials
    
    # Method registry
    methods = {
        'rdals': lambda s, t, p, q: rdals_method(s, t, p, q, config=config),
        'mlls_none': lambda s, t, p, q: mlls_variant(s, t, p, q, 'none', config),
        'mlls_ts': lambda s, t, p, q: mlls_variant(s, t, p, q, 'ts', config),
        'mlls_nbvs': lambda s, t, p, q: mlls_variant(s, t, p, q, 'nbvs', config),
        'mlls_bcts': lambda s, t, p, q: bs_mlls(s, t, p, q, config=config),
        'mlls_vs': lambda s, t, p, q: mlls_variant(s, t, p, q, 'vs', config),
        'cpmcn_none': lambda s, t, p, q: cpmcn_variant(s, t, p, q, 'none', config),
        'cpmcn_ts': lambda s, t, p, q: cpmcn_variant(s, t, p, q, 'ts', config),
        'cpmcn_nbvs': lambda s, t, p, q: cpmcn_variant(s, t, p, q, 'nbvs', config),
        'cpmcn_bcts': lambda s, t, p, q: bs_cpmcn(s, t, p, q, config=config),
        'cpmcn_vs': lambda s, t, p, q: cpmcn_variant(s, t, p, q, 'vs', config),
    }
    
    # Collect per-trial MSE
    per_trial: Dict[str, List[float]] = {name: [] for name in methods}
    
    for i in range(trials):
        print(f"Trial {i + 1}/{trials}...")
        
        clear_stat_cache()
        source_dset, target_dset, p_true, q_true = create_label_shift_datasets(config)
        
        for name, method_fn in methods.items():
            try:
                result = method_fn(source_dset, target_dset, p_true, q_true)
                w_hat = np.array(result['w_hat'], dtype=float)
                w_true = np.array(result['w_true'], dtype=float)
                mse = float(np.mean((w_hat - w_true) ** 2))
                per_trial[name].append(mse)
            except Exception as e:
                print(f"  [WARN] {name} failed: {e}")
                per_trial[name].append(np.nan)
        
        print("=" * 50)
    
    # Aggregate with trimming
    exclude_ratio = config.experiment.exclude_extreme_ratio
    summary_rows = []
    kept_values: Dict[str, List[float]] = {}
    
    for name in methods:
        values = np.array([v for v in per_trial[name] if np.isfinite(v)], dtype=float)
        n_total = len(values)
        n_exclude = int(np.floor(exclude_ratio * n_total))
        
        if n_exclude > 0 and n_total > n_exclude:
            sorted_idx = np.argsort(values)
            values = values[sorted_idx[:-n_exclude]]
        
        kept_values[name] = values.tolist()
        
        summary_rows.append({
            'method': name,
            'mse_mean': float(np.mean(values)) if len(values) > 0 else np.nan,
            'mse_std': float(np.std(values)) if len(values) > 0 else np.nan,
            'mse_median': float(np.median(values)) if len(values) > 0 else np.nan,
            'n_kept': len(values),
        })
    
    df = pd.DataFrame(summary_rows)
    
    print("\n" + "=" * 60)
    print("Calibration Comparison Summary:")
    print("-" * 60)
    print(df.to_string(index=False))
    
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'calibration_comparison',
                'methods': list(methods.keys()),
                'trials': trials,
            }
        )
        
        # Save summary CSV
        csv_path = results_dir / f"{config.dataset.name}_calibration_summary.csv"
        df.to_csv(csv_path, index=False)
        
        # Save per-trial CSV
        trial_df = pd.DataFrame({
            name: [per_trial[name][i] if i < len(per_trial[name]) else np.nan 
                   for i in range(trials)]
            for name in methods
        })
        trial_path = results_dir / f"{config.dataset.name}_calibration_trials.csv"
        trial_df.to_csv(trial_path, index=False)
        
        # Build DataFrame for plot_vertical_strip (Method, Calibration, MSE)
        plot_rows = []
        for name in methods:
            if name == 'rdals':
                method_label = 'RDALS'
                calib_label = 'None'
            else:
                # Parse method_calibration format (e.g., 'mlls_ts' -> 'MLLS', 'TS')
                parts = name.split('_', 1)
                method_label = parts[0].upper()
                calib_label = parts[1].upper() if len(parts) > 1 else 'None'
                if calib_label == 'NONE':
                    calib_label = 'None'
            
            mse_val = df.loc[df['method'] == name, 'mse_mean'].values
            if len(mse_val) > 0 and np.isfinite(mse_val[0]):
                plot_rows.append({
                    'Method': method_label,
                    'Calibration': calib_label,
                    'MSE': float(mse_val[0]),
                })
        
        plot_df = pd.DataFrame(plot_rows)
        
        # Save calibration CSV for later replotting
        calib_csv_path = results_dir / f"{config.dataset.name}_calibration.csv"
        plot_df.to_csv(calib_csv_path, index=False)
        
        # Plot vertical strip chart
        fig_path = results_dir / f"{config.dataset.name}_calibration.pdf"
        plot_vertical_strip(plot_df, save_path=str(fig_path))
        plt.close()
        
        print(f"\nResults saved to: {results_dir}")
    
    return {
        'summary': df,
        'per_trial': per_trial,
        'kept_values': kept_values,
    }


def main():
    parser = argparse.ArgumentParser(description='Calibration comparison experiment')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--trials', type=int, default=None, help='Number of trials')
    
    args = parser.parse_args()
    
    if args.config:
        config = reload_config(args.config)
    else:
        config = get_config()
    
    run_calibration_comparison(
        trials=args.trials,
        config=config,
    )


if __name__ == '__main__':
    main()
