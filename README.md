# RDALS: Regularized Discriminative Alignment for Deep Representations under Label Shift

A reference implementation of RDALS, a label shift estimation method based on Linear Discriminant Analysis (LDA) projection and regularized least squares optimization.

This repository is intended to make the RDALS implementation easy to inspect and run. It also includes optional comparison utilities for several common baselines, but it is not packaged as a fully tuned benchmark suite.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
RDALS/
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── src/                         # Core source code
│   ├── config.py                # Configuration loader (dataclasses + YAML)
│   ├── data/                    # Data loading and label shift generation
│   │   ├── datasets.py          # CIFAR-10/100, MNIST dataset loaders
│   │   ├── shift.py             # Label shift sampling (dirichlet, tweak_one)
│   │   └── cifar10c.py          # CIFAR-10-C corruption support
│   ├── models/                  # Feature extraction
│   │   └── extractor.py         # Pre-trained model feature extraction
│   ├── methods/                 # Label shift estimation methods
│   │   ├── rdals.py             # RDALS implementation
│   │   └── baselines.py         # RLLS, BBSL, MLLS, CPMCN
│   ├── evaluation/              # Evaluation utilities
│   │   └── metrics.py           # MSE computation, multi-trial runner
│   └── utils/                   # Utilities
│       └── plotting.py          # Plotting functions
├── experiments/                 # Experiment scripts
│   ├── main/                    # Main experiments
│   │   ├── run.py               # Single-point evaluation
│   │   └── sweep.py             # Parameter sweep (alpha, rho, sample_size)
│   ├── ablation/                # Ablation studies
│   ├── calibration/             # Calibration method comparison
│   ├── iteration/               # Iteration convergence analysis
│   └── visualization/           # Visualization tools (t-SNE, reweight)
└── results/                     # Output directory (auto-created)
```

## Quick Start

### Run RDALS

```bash
# Run RDALS with the default example configuration
python -m experiments.main.run

# Quick test (single trial, fixed seed)
python -m experiments.main.run --trials 1 --seed 42
```

### Optional Baseline Comparisons

Baseline implementations are provided for convenience and for reproducing comparison-style workflows. They are not run by default.

```bash
# Run RDALS together with selected baselines
python -m experiments.main.run --methods rdals rlls bbsl mlls cpmcn --trials 50
```

## Configuration

Edit `config.yaml` to customize:
- **Dataset**: `cifar10`, `cifar100`, `mnist`
- **Model**: `resnet18`, `resnet50`, etc.
- **Shift type**: `dirichlet` or `tweak_one`
- **Estimation parameters**: regularization, projection dimensions
- **Baseline parameters**: RLLS, MLLS, BBSL, CPMCN settings

The checked-in `config.yaml` is a lightweight example configuration for code inspection and basic execution checks. It should not be read as a claim of optimal hyperparameters for RDALS, as a fully tuned setting for every baseline, or as a guarantee of reproducing exact paper tables.

## Methods

| Method | Description |
|--------|-------------|
| **RDALS** | LDA projection + regularized least squares |
| **RLLS** | Regularized Learning under Label Shift |
| **BBSL** | Black Box Shift Learning |
| **MLLS** | Maximum Likelihood Label Shift |
| **CPMCN** | Calibrated Predictions Matching Class priors Network |

## Notes on Baselines

The baseline code is included so readers can inspect the comparison scaffold and run optional comparisons from the same interface. Baseline behavior can be sensitive to calibration splits, head-training hyperparameters, random seeds, and dataset/model choices. For serious benchmarking, tune each baseline according to its own recommended protocol and report the corresponding configuration.

## License

MIT License
