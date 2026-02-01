"""
Core runner for calibration comparison experiments.
Compares different calibration methods (None, TS, NBVS, BCTS, VS) with MLLS and CPMCN.
"""

import time
from typing import Dict, Any, List, Callable, Optional

import numpy as np
from scipy.special import softmax

try:
    import abstention.calibration as abst_cal
    HAS_ABSTENTION = True
except ImportError:
    HAS_ABSTENTION = False

from src.config import Config, get_config
from src.methods.baselines import _get_stats_cached, _build_stats


def _fit_calibrator(
    calib_type: str,
    logits_calib: np.ndarray,
    y_calib: np.ndarray,
    k: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a calibration function based on calibration type.
    
    Args:
        calib_type: Calibration type (none, ts, nbvs, bcts, vs)
        logits_calib: Calibration logits
        y_calib: Calibration labels
        k: Number of classes
    
    Returns:
        Calibration function: logits -> probabilities
    """
    calib_type = str(calib_type).lower()
    
    if calib_type == 'none':
        return lambda logits: softmax(logits, axis=1)
    
    if not HAS_ABSTENTION:
        raise ImportError("abstention package required for calibration")
    
    labels_onehot = np.zeros((len(y_calib), k), dtype=float)
    if len(y_calib) > 0:
        labels_onehot[np.arange(len(y_calib)), y_calib.astype(int)] = 1.0
    
    if calib_type == 'ts':
        calib = abst_cal.TempScaling(lbfgs_kwargs={}, verbose=False, bias_positions=[])
    elif calib_type == 'nbvs':
        calib = abst_cal.NoBiasVectorScaling(lbfgs_kwargs={}, verbose=False)
    elif calib_type == 'bcts':
        calib = abst_cal.TempScaling(lbfgs_kwargs={}, verbose=False, bias_positions=list(range(k)))
    elif calib_type == 'vs':
        calib = abst_cal.VectorScaling(lbfgs_kwargs={}, verbose=False)
    else:
        raise ValueError(f"Unknown calibration type: {calib_type}")
    
    calibrate = calib(valid_preacts=logits_calib, valid_labels=labels_onehot)
    return calibrate


def mlls_variant(
    source_dset,
    target_dset,
    p_true: np.ndarray,
    q_true: np.ndarray,
    calib_type: str,
    config: Config = None,
) -> Dict[str, Any]:
    """
    MLLS with specified calibration method.
    """
    if config is None:
        config = get_config()
    
    stats = _get_stats_cached(source_dset, target_dset, p_true, q_true, config)
    k = stats['k']
    logits_calib = stats.get('logits_calib')
    logits_T = stats.get('logits_T')
    logits_S_full = stats.get('logits_S_full')
    Y_calib = stats.get('Y_calib')
    base_time = float(stats.get('prep_time_sec', 0.0))
    
    t0 = time.time()
    
    calibrate = _fit_calibrator(calib_type, logits_calib, Y_calib, k)
    P_T = calibrate(logits_T)
    P_S = calibrate(logits_S_full)
    p_s = P_S.mean(axis=0)
    
    # EM loop
    q = p_s.copy()
    q = q / (q.sum() + 1e-12)
    
    max_iters = config.baselines.mlls.em_max_iters
    tol = config.baselines.mlls.em_tol
    eps = config.baselines.mlls.em_eps
    
    p_s_safe = np.maximum(p_s, eps)
    
    for _ in range(max_iters):
        prev = q.copy()
        ratio = q / p_s_safe
        num = P_T * ratio[np.newaxis, :]
        den = np.sum(num, axis=1, keepdims=True) + eps
        r = num / den
        q = r.mean(axis=0)
        q = np.maximum(q, 0.0)
        s = q.sum()
        if s > 0:
            q = q / s
        if np.linalg.norm(q - prev, ord=1) < tol:
            break
    
    q_hat = q
    w_hat = q_hat / p_s_safe
    w_true = q_true / p_true
    
    time_sec = base_time + (time.time() - t0)
    
    return {
        'q_hat': q_hat,
        'w_hat': w_hat,
        'w_true': w_true,
        'time_sec': time_sec,
    }


def cpmcn_variant(
    source_dset,
    target_dset,
    p_true: np.ndarray,
    q_true: np.ndarray,
    calib_type: str,
    config: Config = None,
) -> Dict[str, Any]:
    """
    CPMCN with specified calibration method.
    """
    from scipy.optimize import minimize
    
    if config is None:
        config = get_config()
    
    stats = _get_stats_cached(source_dset, target_dset, p_true, q_true, config)
    k = stats['k']
    logits_calib = stats.get('logits_calib')
    logits_T = stats.get('logits_T')
    Y_calib = stats.get('Y_calib')
    Y_S = stats.get('Y_S')
    base_time = float(stats.get('prep_time_sec', 0.0))
    
    t0 = time.time()
    
    calibrate = _fit_calibrator(calib_type, logits_calib, Y_calib, k)
    P_T = calibrate(logits_T)
    
    # Source prior
    p_base = np.bincount(Y_S.astype(int), minlength=k).astype(float)
    p_base = p_base / max(p_base.sum(), 1.0)
    
    eps = 1e-12
    
    def objective(w):
        w = np.maximum(w, 0.0)
        g = np.maximum(P_T @ w, eps)
        R = P_T / g[:, np.newaxis]
        p_q_w = R.mean(axis=0)
        diff = p_base - p_q_w
        return float(np.sum(diff ** 2))
    
    w0 = np.ones(k, dtype=float)
    bounds = [(eps, None)] * k
    
    res = minimize(objective, w0, method='L-BFGS-B', bounds=bounds)
    w = res.x if res is not None and hasattr(res, 'x') else w0
    w = np.maximum(w, 0.0)
    
    z = float(np.dot(w, p_base)) + eps
    w = w / z
    
    q_hat = p_base * w
    s = q_hat.sum()
    if s > 0:
        q_hat = q_hat / s
    
    w_hat = w
    w_true = q_true / p_true
    
    time_sec = base_time + (time.time() - t0)
    
    return {
        'q_hat': q_hat,
        'w_hat': w_hat,
        'w_true': w_true,
        'time_sec': time_sec,
    }
