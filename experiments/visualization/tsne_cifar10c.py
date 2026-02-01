"""
t-SNE visualization for CIFAR-10 vs CIFAR-10-C.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

from src.config import Config, get_config, reload_config
from .utils import SimpleFeatureExtractor, sample_paired_cifar10_and_c


CIFAR10_CLASS_NAMES = {
    0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer',
    5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck',
}


def run_tsne_cifar10c(
    corruption: str = 'brightness',
    severity: int = 5,
    classes: List[int] = None,
    n_per_class: int = 100,
    seed: int = 42,
    config: Config = None,
    save_results: bool = True,
) -> None:
    """
    Run t-SNE visualization for CIFAR-10 vs CIFAR-10-C.
    """
    if config is None:
        config = get_config()
    
    if classes is None:
        classes = [2, 5]  # Bird, Dog
    
    print(f"CIFAR-10 classes: {classes}")
    print(f"Corruption: {corruption}, severity: {severity}")
    print(f"Samples per class per domain: {n_per_class}")
    
    # Sample paired images
    images, labels, domains = sample_paired_cifar10_and_c(
        classes=classes,
        n_per_class=n_per_class,
        corruption=corruption,
        severity=severity,
        seed=seed,
        config=config,
    )
    
    N = images.shape[0]
    print(f"Total samples: {N} (source={np.sum(domains==0)}, target={np.sum(domains==1)})")
    
    # Raw pixel features
    raw_flat = images.astype(np.float32) / 255.0
    raw_flat = raw_flat.reshape(N, -1)
    
    # ResNet features
    extractor = SimpleFeatureExtractor(config=config)
    feats = extractor.extract(
        images,
        batch_size=config.model.batch_size,
        num_workers=config.model.num_workers,
    )
    
    # t-SNE
    print("Running t-SNE on raw pixels...")
    tsne_raw = TSNE(n_components=2, random_state=seed, init='pca', perplexity=30).fit_transform(raw_flat)
    
    print("Running t-SNE on ResNet features...")
    tsne_feat = TSNE(n_components=2, random_state=seed, init='pca', perplexity=30).fit_transform(feats)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    titles = ['t-SNE on Raw Pixels', 't-SNE on ResNet Features']
    emb_list = [tsne_raw, tsne_feat]
    
    domain_colors = {0: '#AA2B46', 1: '#3B5DA3'}
    domain_labels = {0: 'Source', 1: 'Target'}
    
    class_marker_cycle = ['o', '^', 's', 'D', 'P', 'X']
    class_markers = {cls: class_marker_cycle[i % len(class_marker_cycle)] for i, cls in enumerate(sorted(classes))}
    
    for ax, emb, title in zip(axes, emb_list, titles):
        for cls in classes:
            for d in [0, 1]:
                mask = (labels == cls) & (domains == d)
                if not np.any(mask):
                    continue
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    marker=class_markers[cls],
                    facecolors='none',
                    edgecolors=domain_colors[d],
                    linewidths=2,
                    s=60,
                )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    # Legend
    domain_handles = [
        Patch(facecolor=domain_colors[d], edgecolor='none', label=domain_labels[d])
        for d in [0, 1]
    ]
    class_handles = [
        Line2D([0], [0], marker=class_markers[cls], markersize=12, color='black',
               linestyle='none', markeredgewidth=1.5, markerfacecolor='none',
               label=CIFAR10_CLASS_NAMES.get(cls, f'Class {cls}'))
        for cls in classes
    ]
    
    all_handles = domain_handles + class_handles
    fig.legend(handles=all_handles, loc='lower center', ncol=len(all_handles),
               fontsize=14, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    
    if save_results:
        results_dir = config.create_results_dir(
            extra_meta={
                'experiment': 'tsne_cifar10c',
                'corruption': corruption,
                'severity': severity,
                'classes': classes,
                'n_per_class': n_per_class,
                'seed': seed,
            }
        )
        
        fig_path = results_dir / f"cifar10_vs_cifar10c_{corruption}_s{severity}_tsne.pdf"
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to: {fig_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization for CIFAR-10 vs CIFAR-10-C')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--corruption', type=str, default='brightness', help='Corruption type')
    parser.add_argument('--severity', type=int, default=5, help='Severity level (1-5)')
    parser.add_argument('--classes', type=int, nargs='+', default=[2, 5], help='Class indices')
    parser.add_argument('--n-per-class', type=int, default=100, help='Samples per class')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.config:
        config = reload_config(args.config)
    else:
        config = get_config()
    
    run_tsne_cifar10c(
        corruption=args.corruption,
        severity=args.severity,
        classes=args.classes,
        n_per_class=args.n_per_class,
        seed=args.seed,
        config=config,
    )


if __name__ == '__main__':
    main()
