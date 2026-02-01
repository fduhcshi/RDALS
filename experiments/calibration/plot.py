"""
Plotting utilities for calibration comparison experiments.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

try:
    import seaborn as sns
except ImportError:
    sns = None


def plot_vertical_strip(
    df: pd.DataFrame,
    save_path: str = None,
    title: str = None,
    figsize: Tuple[float, float] = (5, 4),
    yscale: str = 'symlog',
    y_linthresh: float = 1e-4,
    y_linscale: float = 1.0,
    y_base: float = 10,
):
    """
    Plot vertical strip chart for calibration comparison.
    
    Args:
        df: DataFrame with columns 'Method', 'Calibration', 'MSE'
        save_path: Path to save figure
        title: Optional title
        figsize: Figure size
        yscale: Y-axis scale ('symlog', 'log', 'linear')
        y_linthresh: Linear threshold for symlog scale
        y_linscale: Linear scale for symlog scale
        y_base: Base for symlog scale
    """
    if sns is None:
        raise ImportError("seaborn is required for plot_vertical_strip")
    
    required = {'Method', 'Calibration', 'MSE'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    plot_df = df.copy()
    plot_df['Calibration'] = plot_df['Calibration'].fillna('None').astype(str)

    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': 12})

    def _lighten_color(color, amount: float = 0.6):
        r, g, b = mcolors.to_rgb(color)
        r = 1.0 - (1.0 - r) * amount
        g = 1.0 - (1.0 - g) * amount
        b = 1.0 - (1.0 - b) * amount
        return (r, g, b)

    preferred_hue_order = ['None', 'TS', 'NBVS', 'BCTS', 'VS']
    present_calibs = [c for c in preferred_hue_order if c in set(plot_df['Calibration'].unique())]
    if len(present_calibs) == 0:
        hue_order = list(plot_df['Calibration'].unique())
    else:
        extra = [c for c in plot_df['Calibration'].unique() if c not in present_calibs]
        hue_order = present_calibs + extra
    pal_list = sns.color_palette('Set2', n_colors=len(hue_order))
    palette = {c: pal_list[i] for i, c in enumerate(hue_order)}

    preferred_method_order = ['RDALS', 'MLLS', 'CPMCN']
    present_methods = [m for m in preferred_method_order if m in set(plot_df['Method'].unique())]
    method_order = present_methods if len(present_methods) > 0 else list(plot_df['Method'].unique())
    method_pos = {m: i for i, m in enumerate(method_order)}

    sns.scatterplot(
        data=plot_df,
        x='Method',
        y='MSE',
        hue='Calibration',
        hue_order=hue_order,
        style='Calibration',
        style_order=hue_order,
        palette=palette,
        s=160,
        zorder=10,
        edgecolor='k',
    )

    ax = plt.gca()
    for calib in hue_order:
        sub = plot_df[plot_df['Calibration'] == calib]
        if sub.shape[0] < 2:
            continue
        # Only connect MLLS <-> CPMCN (no RDALS <-> MLLS line)
        needed_methods = ['MLLS', 'CPMCN']
        xs = []
        ys = []
        for m in needed_methods:
            if m not in method_pos:
                continue
            row = sub[sub['Method'] == m]
            if row.shape[0] == 0:
                continue
            xs.append(method_pos[m])
            ys.append(float(row['MSE'].iloc[0]))
        if len(xs) == 2:
            line_color = _lighten_color(palette.get(calib, 'gray'), amount=0.7)
            ax.plot(xs, ys, color=line_color, linewidth=2.0, alpha=0.9, zorder=3, label='_nolegend_')

    ours_mask = plot_df['Method'].astype(str).str.lower().isin({'ours', 'rdals'})
    if bool(ours_mask.any()):
        ours_val = float(plot_df.loc[ours_mask, 'MSE'].mean())
        plt.axhline(y=ours_val, color='gray', linestyle='--', alpha=0.5, zorder=0)

    if 'RDALS' in method_pos:
        _rdals = plot_df[plot_df['Method'] == 'RDALS']
        _rdals_none = _rdals[_rdals['Calibration'] == 'None']
        _rdals_pick = _rdals_none if _rdals_none.shape[0] > 0 else _rdals
        if _rdals_pick.shape[0] > 0:
            rdals_y = float(_rdals_pick['MSE'].iloc[0])
            rdals_x = method_pos['RDALS']
            ax.annotate(
                'Ours',
                xy=(rdals_x, rdals_y),
                xytext=(6, 6),
                textcoords='offset points',
                ha='left',
                va='bottom',
                color='gray',
                fontsize=16,
                zorder=20,
            )

    plt.ylabel('MSE', fontsize=18)
    plt.xlabel('', fontsize=18)

    if yscale == 'symlog':
        try:
            ax.set_yscale('symlog', linthresh=y_linthresh, linscale=y_linscale, base=y_base)
        except TypeError:
            ax.set_yscale('symlog', linthresh=y_linthresh)

        vals = plot_df['MSE'].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size > 0:
            vmin = float(vals.min())
            vmax = float(vals.max())
            bottom = max(vmin / 1.5, 1e-12)
            top = vmax * 1.5
            if top > bottom:
                ax.set_ylim(bottom, top)
            low_exp = int(np.floor(np.log(vmin) / np.log(float(y_base))))
            high_exp = int(np.ceil(np.log(vmax) / np.log(float(y_base))))
            ticks = [float(y_base) ** k for k in range(low_exp, high_exp + 1)]
            if len(ticks) > 0:
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda y, pos: ('0' if y == 0 else f"{y:.6g}"))
                )
    elif yscale is not None and yscale != 'linear':
        ax.set_yscale(yscale)

    ax.tick_params(axis='both', which='major', labelsize=16, width=2)

    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(
            handles=handles,
            labels=labels,
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0,
            title='Calibration',
            fontsize=14,
            title_fontsize=14,
            frameon=False,
        )
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    sns.despine()
    
    if title:
        plt.title(title)
    
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_calibration_boxplot(
    data: Dict[str, List[float]],
    method_groups: Dict[str, List[str]] = None,
    figsize: Tuple[float, float] = (8, 5),
    save_path: str = None,
) -> plt.Axes:
    """
    Plot boxplot for calibration comparison.
    
    Args:
        data: Dictionary mapping method names to MSE values
        method_groups: Dictionary mapping group names to method lists
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib axes
    """
    if method_groups is None:
        method_groups = {
            'Ours': ['rdals'],
            'MLLS': [k for k in data.keys() if k.startswith('mlls_')],
            'CPMCN': [k for k in data.keys() if k.startswith('cpmcn_')],
        }
    
    # Flatten order
    order = []
    for group_methods in method_groups.values():
        order.extend(group_methods)
    order = [m for m in order if m in data]
    
    values = [data[m] for m in order]
    
    # Positions with gaps between groups
    positions = []
    widths = 0.5
    step = widths + 0.02
    cur = 1.0
    
    for group_name, group_methods in method_groups.items():
        group_in_order = [m for m in group_methods if m in order]
        if not group_in_order:
            continue
        for i in range(len(group_in_order)):
            positions.append(cur + i * step)
        cur = positions[-1] + 2 * step
    
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(values, positions=positions, widths=widths, patch_artist=True, showfliers=False)
    
    # Colors
    colors = []
    our_color = '#d62728'
    mlls_cmap = plt.get_cmap('Blues')
    cpmcn_cmap = plt.get_cmap('Oranges')
    
    color_idx = 0
    for group_name, group_methods in method_groups.items():
        group_in_order = [m for m in group_methods if m in order]
        n = len(group_in_order)
        if group_name == 'Ours':
            colors.extend([our_color] * n)
        elif group_name == 'MLLS':
            for i in range(n):
                t = 0.3 + 0.6 * (i / max(n - 1, 1))
                colors.append(mlls_cmap(t))
        elif group_name == 'CPMCN':
            for i in range(n):
                t = 0.3 + 0.6 * (i / max(n - 1, 1))
                colors.append(cpmcn_cmap(t))
        else:
            colors.extend(['#7f7f7f'] * n)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_ylabel('MSE')
    ax.set_xlabel('Calibration Method')
    
    # X-axis labels
    def _label(name: str) -> str:
        if name == 'rdals':
            return 'Ours'
        if '_' in name:
            suffix = name.split('_', 1)[1]
            return suffix.upper() if suffix.lower() != 'none' else 'None'
        return name
    
    ax.set_xticks(positions)
    ax.set_xticklabels([_label(m) for m in order])
    
    # Legend
    legend_handles = []
    if any(m.startswith('rdals') or m == 'rdals' for m in order):
        legend_handles.append(Patch(facecolor=our_color, alpha=0.6, label='Ours'))
    if any(m.startswith('mlls_') for m in order):
        legend_handles.append(Patch(facecolor=mlls_cmap(0.7), alpha=0.6, label='MLLS'))
    if any(m.startswith('cpmcn_') for m in order):
        legend_handles.append(Patch(facecolor=cpmcn_cmap(0.7), alpha=0.6, label='CPMCN'))
    
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved figure to: {save_path}")
    
    return ax
