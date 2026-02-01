"""
Per-iteration MSE tracking for different methods.
"""

from typing import List

import numpy as np
import torch
from scipy.optimize import minimize

try:
    import abstention.calibration as abst_cal
    HAS_ABSTENTION = True
except ImportError:
    HAS_ABSTENTION = False

from src.config import Config, get_config
from src.models.extractor import FeatureExtractor
from src.methods.rdals import LabelShiftEstimator
from src.methods.baselines import _get_stats_cached


def run_rdals_iteration(
    source_dset,
    target_dset,
    p_true: np.ndarray,
    q_true: np.ndarray,
    config: Config = None,
) -> List[float]:
    """
    Run RDALS optimization and record MSE at each iteration.
    
    Returns:
        List of MSE values per iteration
    """
    if config is None:
        config = get_config()
    
    device = torch.device(config.model.device)
    num_classes = config.dataset.num_classes
    
    # Extract features
    extractor = FeatureExtractor(config=config)
    Z_S, Y_S = extractor.extract_features(source_dset)
    Z_T, Y_T = extractor.extract_features(target_dset)
    
    # Initialize estimator
    estimator = LabelShiftEstimator(
        lda_components=num_classes - 1,
        regularizer_lambda=config.estimation.regularizer_lambda,
    )
    estimator.fit_lda(Z_S, Y_S)
    
    # Build A and b
    A = estimator._build_A(Z_S, Y_S)
    b = estimator._build_b(Z_T)
    k = estimator.k
    
    # Compute true weights
    p_emp = np.bincount(Y_S.astype(int), minlength=num_classes).astype(float)
    p_emp = p_emp / max(p_emp.sum(), 1.0)
    eps = 1e-12
    denom = np.maximum(p_emp, eps)
    w_true = q_true / p_true
    
    # Track history
    history: List[float] = []
    
    def objective(Q: np.ndarray) -> float:
        residual = A @ Q - b
        l2_loss = float(np.sum(residual ** 2))
        reg = estimator.lambda_reg * float(np.sum(Q ** 2))
        return l2_loss + reg
    
    def callback(Qk: np.ndarray) -> None:
        Q = np.array(Qk, dtype=float)
        Q[Q < 0.0] = 0.0
        s = float(np.sum(Q))
        if s > 0.0:
            Q = Q / s
        w_hat = Q / denom
        mse = float(np.mean((w_hat - w_true) ** 2))
        history.append(mse)
    
    # Initial guess
    if config.estimation.q_init_type == 2 and estimator.p_true is not None:
        Q_init = estimator.p_true.copy()
    else:
        Q_init = np.full(k, 1.0 / k, dtype=float)
    
    constraints = [{'type': 'eq', 'fun': lambda Q: np.sum(Q) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(k)]
    
    minimize(
        objective,
        Q_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={'disp': False},
    )
    
    return history


def run_mlls_iteration(
    source_dset,
    target_dset,
    p_true: np.ndarray,
    q_true: np.ndarray,
    config: Config = None,
) -> List[float]:
    """
    Run MLLS EM and record MSE at each iteration.
    """
    if not HAS_ABSTENTION:
        raise ImportError("abstention package required for MLLS")
    
    if config is None:
        config = get_config()
    
    stats = _get_stats_cached(source_dset, target_dset, p_true, q_true, config)
    k = stats['k']
    logits_calib = stats.get('logits_calib')
    logits_T = stats.get('logits_T')
    logits_S_full = stats.get('logits_S_full')
    Y_calib = stats.get('Y_calib')
    
    # Calibration
    labels_onehot = np.zeros((len(Y_calib), k), dtype=float)
    if len(Y_calib) > 0:
        labels_onehot[np.arange(len(Y_calib)), Y_calib.astype(int)] = 1.0
    
    calib = abst_cal.TempScaling(
        lbfgs_kwargs={},
        verbose=False,
        bias_positions=list(range(k)),
    )
    calibrate = calib(valid_preacts=logits_calib, valid_labels=labels_onehot)
    
    P_T = calibrate(logits_T)
    P_S = calibrate(logits_S_full)
    p_s = P_S.mean(axis=0)
    
    q = p_s.copy()
    q = q / (q.sum() + 1e-12)
    
    max_iters = config.baselines.mlls.em_max_iters
    tol = config.baselines.mlls.em_tol
    eps = config.baselines.mlls.em_eps
    
    p_s_safe = np.maximum(p_s, eps)
    w_true = q_true / p_true
    
    history: List[float] = []
    
    for _ in range(max_iters):
        prev = q.copy()
        ratio = q / p_s_safe
        num = P_T * ratio[np.newaxis, :]
        den = np.sum(num, axis=1, keepdims=True) + eps
        r = num / den
        q = r.mean(axis=0)
        q = np.maximum(q, 0.0)
        s = q.sum()
        if s > 0.0:
            q = q / s
        
        w_hat = q / p_s_safe
        mse = float(np.mean((w_hat - w_true) ** 2))
        history.append(mse)
        
        if np.linalg.norm(q - prev, ord=1) < tol:
            break
    
    return history


def run_cpmcn_iteration(
    source_dset,
    target_dset,
    p_true: np.ndarray,
    q_true: np.ndarray,
    config: Config = None,
) -> List[float]:
    """
    Run CPMCN L-BFGS-B and record MSE at each iteration.
    """
    if not HAS_ABSTENTION:
        raise ImportError("abstention package required for CPMCN")
    
    if config is None:
        config = get_config()
    
    stats = _get_stats_cached(source_dset, target_dset, p_true, q_true, config)
    k = stats['k']
    logits_calib = stats.get('logits_calib')
    logits_T = stats.get('logits_T')
    Y_calib = stats.get('Y_calib')
    Y_S = stats.get('Y_S')
    
    # Calibration
    labels_onehot = np.zeros((len(Y_calib), k), dtype=float)
    if len(Y_calib) > 0:
        labels_onehot[np.arange(len(Y_calib)), Y_calib.astype(int)] = 1.0
    
    calib = abst_cal.TempScaling(
        lbfgs_kwargs={},
        verbose=False,
        bias_positions=list(range(k)),
    )
    calibrate = calib(valid_preacts=logits_calib, valid_labels=labels_onehot)
    P_T = calibrate(logits_T)
    
    # Source prior
    p_base = np.bincount(Y_S.astype(int), minlength=k).astype(float)
    p_base = p_base / max(p_base.sum(), 1.0)
    
    eps = 1e-12
    w_true = q_true / p_true
    
    history: List[float] = []
    
    def objective(w):
        w = np.maximum(w, 0.0)
        g = np.maximum(P_T @ w, eps)
        R = P_T / g[:, np.newaxis]
        p_q_w = R.mean(axis=0)
        diff = p_base - p_q_w
        return float(np.sum(diff ** 2))
    
    def callback(wk: np.ndarray) -> None:
        w = np.maximum(np.array(wk, dtype=float), 0.0)
        z = float(np.dot(w, p_base)) + eps
        if z > 0.0:
            w_norm = w / z
        else:
            w_norm = w
        mse = float(np.mean((w_norm - w_true) ** 2))
        history.append(mse)
    
    w0 = np.ones(k, dtype=float)
    bounds = [(eps, None)] * k
    
    res = minimize(
        objective,
        w0,
        method='L-BFGS-B',
        bounds=bounds,
        callback=callback,
    )
    
    # If callback was never called, add final point
    if not history:
        w = res.x if res is not None and hasattr(res, 'x') else w0
        w = np.maximum(w, 0.0)
        z = float(np.dot(w, p_base)) + eps
        w_norm = w / z if z > 0.0 else w
        mse = float(np.mean((w_norm - w_true) ** 2))
        history.append(mse)
    
    return history
