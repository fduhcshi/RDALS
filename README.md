# RDALS: Regularized Discriminative Alignment for Deep Representations under Label Shift

A novel framework for label shift estimation using Linear Discriminant Analysis (LDA) projection and regularized least squares optimization.

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
│   │   ├── rdals.py             # Our method: RDALS
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

### Run Main Experiment

```bash
# Run with default settings
python -m experiments.main.run

# Specify methods and trials
python -m experiments.main.run --methods rdals rlls bbsl mlls --trials 50

# Quick test (single trial, fixed seed)
python -m experiments.main.run --trials 1 --seed 42
```

### Run Parameter Sweep

```bash
# Sweep Dirichlet alpha (smaller alpha = larger shift)
python -m experiments.main.sweep --sweep alpha --values 0.1 0.5 1.0 2.0 5.0 --trials 50

# Sweep tweak_one rho (larger rho = larger shift)
python -m experiments.main.sweep --sweep rho --values 0.1 0.3 0.5 0.7 0.9 --trials 50

# Sweep sample size
python -m experiments.main.sweep --sweep sample_size --values 100 500 1000 2000 5000 --trials 50
```

## Configuration

Edit `config.yaml` to customize:
- **Dataset**: `cifar10`, `cifar100`, `mnist`
- **Model**: `resnet18`, `resnet50`, etc.
- **Shift type**: `dirichlet` or `tweak_one`
- **Estimation parameters**: regularization, projection dimensions
- **Baseline parameters**: RLLS, MLLS, BBSL, CPMCN settings

## Methods

| Method | Description |
|--------|-------------|
| **RDALS** (Ours) | LDA projection + regularized least squares |
| **RLLS** | Regularized Learning under Label Shift |
| **BBSL** | Black Box Shift Learning |
| **MLLS** | Maximum Likelihood Label Shift |
| **CPMCN** | Calibrated Predictions Matching Class priors Network |

## License

MIT License
