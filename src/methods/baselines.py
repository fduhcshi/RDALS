"""
Optional baseline methods for label shift estimation.

These implementations provide comparison utilities alongside RDALS. They share
feature extraction and experiment configuration with the main evaluation code.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from scipy.optimize import minimize
from torch.utils.data import Dataset

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

try:
    import abstention.calibration as abst_cal
    HAS_ABSTENTION = True
except ImportError:
    HAS_ABSTENTION = False

from ..config import Config, get_config
from ..models.extractor import FeatureExtractor
from ..models.classifier import train_weighted_head

# Cache for shared statistics across methods
_STAT_CACHE = {}


def clear_stat_cache():
    """Clear the statistics cache."""
    global _STAT_CACHE
    _STAT_CACHE.clear()


def _train_linear_head(Z_S, Y_S, k, epochs=None, lr=None, batch_size=None, device='cpu', momentum=None, weight_decay=None, config=None):
    """Train a linear classification head for baseline estimators."""
    d = Z_S.shape[1]
    model = nn.Linear(d, k).to(device)
    
    # Read defaults from config if available
    if config is not None and hasattr(config, 'baselines') and hasattr(config.baselines, 'head'):
        head_cfg = config.baselines.head
        if epochs is None:
            epochs = getattr(head_cfg, 'epochs', 5)
        if lr is None:
            lr = getattr(head_cfg, 'lr', 0.001)
        if batch_size is None:
            batch_size = getattr(head_cfg, 'batch_size', 128)
        if momentum is None:
            momentum = getattr(head_cfg, 'momentum', 0.9)
        if weight_decay is None:
            weight_decay = getattr(head_cfg, 'weight_decay', 5e-4)
    else:
        # Standalone defaults used when no Config object is supplied.
        if epochs is None:
            epochs = 5
        if lr is None:
            lr = 0.001
        if batch_size is None:
            batch_size = 128
        if momentum is None:
            momentum = 0.9
        if weight_decay is None:
            weight_decay = 5e-4
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.from_numpy(Z_S).float().to(device)
    y = torch.from_numpy(Y_S).long().to(device)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X[idx])
            loss = loss_fn(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _build_stats(source_dset, target_dset, p_true, q_true, config=None):
    """Build shared prediction statistics for baseline estimators."""
    if config is None:
        config = get_config()
    
    device = torch.device(config.model.device)
    k = config.dataset.num_classes
    extractor = FeatureExtractor(config=config)
    Z_S, Y_S = extractor.extract_features(source_dset)
    Z_T, Y_T = extractor.extract_features(target_dset)
    t0 = time.time()
    split_flag = config.baselines.split_train_calibration
    ratio = config.baselines.calibration_ratio
    if split_flag:
        rng = np.random.default_rng()
        calib_idx_list = []
        train_idx_list = []
        for c in range(k):
            c_idx = np.where(Y_S == c)[0]
            if c_idx.size == 0:
                continue
            perm = rng.permutation(c_idx)
            n_calib_c = int(len(c_idx) * ratio)
            calib_idx_list.append(perm[:n_calib_c])
            train_idx_list.append(perm[n_calib_c:])
        calib_idx = np.concatenate(calib_idx_list) if len(calib_idx_list) > 0 else np.array([], dtype=int)
        train_idx = np.concatenate(train_idx_list) if len(train_idx_list) > 0 else np.array([], dtype=int)
        Z_train, Y_train = Z_S[train_idx], Y_S[train_idx]
        Z_calib, Y_calib = Z_S[calib_idx], Y_S[calib_idx]
        head = _train_linear_head(Z_train, Y_train, k, device=device, config=config)
        with torch.no_grad():
            logits_S = head(torch.from_numpy(Z_calib).float().to(device)).cpu().numpy()
            logits_T = head(torch.from_numpy(Z_T).float().to(device)).cpu().numpy()
            logits_S_full = head(torch.from_numpy(Z_S).float().to(device)).cpu().numpy()
        pred_S = np.argmax(logits_S, axis=1)
        pred_T = np.argmax(logits_T, axis=1)
        m_train = len(pred_S)
    else:
        head = _train_linear_head(Z_S, Y_S, k, device=device, config=config)
        with torch.no_grad():
            logits_S = head(torch.from_numpy(Z_S).float().to(device)).cpu().numpy()
            logits_T = head(torch.from_numpy(Z_T).float().to(device)).cpu().numpy()
            logits_S_full = logits_S
        pred_S = np.argmax(logits_S, axis=1)
        pred_T = np.argmax(logits_T, axis=1)
        m_train = len(pred_S)
        calib_idx = None
        train_idx = None
        Y_calib = Y_S
    t_prep_end = time.time()
    C = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if split_flag:
                C[i, j] = float(np.sum((pred_S == i) & (Y_calib == j))) / m_train
            else:
                C[i, j] = float(np.sum((pred_S == i) & (Y_S == j))) / m_train
    mu_train = np.zeros(k, dtype=float)
    for i in range(k):
        mu_train[i] = float(np.sum(pred_S == i)) / m_train
    mu_y = np.zeros(k, dtype=float)
    m_test = len(pred_T)
    for i in range(k):
        mu_y[i] = float(np.sum(pred_T == i)) / m_test
    prep_time_sec = float(t_prep_end - t0)
    stat_time_sec = float(time.time() - t_prep_end)
    shared_time_sec = prep_time_sec + stat_time_sec
    return {
        'C': C,
        'mu_train': mu_train,
        'mu_y': mu_y,
        'k': k,
        'm_train': m_train,
        'p_true': p_true,
        'q_true': q_true,
        'calib_idx': calib_idx,
        'train_idx': train_idx,
        'logits_calib': logits_S,
        'logits_T': logits_T,
        'logits_S_full': logits_S_full,
        'Y_calib': Y_calib,
        'Y_S': Y_S,
        'Z_S': Z_S,
        'prep_time_sec': prep_time_sec,
        'stat_time_sec': stat_time_sec,
        'shared_time_sec': shared_time_sec,
    }


def _get_stats_cached(source_dset, target_dset, p_true, q_true, config=None):
    """Return cached shared statistics for a source/target dataset pair."""
    if config is None:
        config = get_config()
    key = (
        id(source_dset),
        id(target_dset),
        config.baselines.split_train_calibration,
        config.baselines.calibration_ratio,
    )
    entry = _STAT_CACHE.get(key)
    if entry is not None:
        return entry
    entry = _build_stats(source_dset, target_dset, p_true, q_true, config)
    _STAT_CACHE[key] = entry
    return entry


def _compute_3deltaC(n_class, n_train, delta):
    """Compute the RLLS finite-sample regularization factor."""
    return 3*(2*np.log(2*n_class/delta)/(3*n_train) + np.sqrt(2*np.log(2*n_class/delta)/n_train))


def bs_rlls(source_dset, target_dset, p_true, q_true, train_downstream=False, config=None, seed=None):
    """Estimate label shift with RLLS."""
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for RLLS method")
    if config is None:
        config = get_config()
    
    stats = _get_stats_cached(source_dset, target_dset, p_true, q_true, config)
    C = stats['C']
    mu_train = stats['mu_train']
    mu_y = stats['mu_y']
    k = stats['k']
    m_train = stats['m_train']
    _prep = stats.get('prep_time_sec', None)
    _stat = stats.get('stat_time_sec', None)
    base_time = float(_prep) + float(_stat) if (_prep is not None and _stat is not None) else float(stats.get('shared_time_sec', 0.0))
    
    alpha = config.baselines.rlls.alpha
    delta = config.baselines.rlls.delta
    rho_cfg = config.baselines.rlls.rho
    if rho_cfg is None:
        rho = float(alpha) * float(_compute_3deltaC(k, m_train, float(delta)))
    else:
        rho = float(rho_cfg)
    
    theta = cp.Variable(k)
    b = mu_y - mu_train
    t_extra0 = time.time()
    objective = cp.Minimize(cp.norm(C @ theta - b, 2) + rho * cp.norm(theta, 2))
    constraints = [theta >= -1]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    w = 1 + np.array(theta.value).reshape(-1)
    w = np.maximum(w, 1e-8)
    
    # Estimate the empirical source prior from source dataset labels.
    try:
        if hasattr(source_dset, 'indices') and hasattr(source_dset, 'dataset') and hasattr(source_dset.dataset, 'targets'):
            y_src = np.array(source_dset.dataset.targets, dtype=int)[np.array(source_dset.indices, dtype=int)]
        else:
            raise AttributeError
    except Exception:
        y_src = np.array([], dtype=int)
    counts = np.bincount(y_src, minlength=k).astype(float)
    if counts.sum() == 0:
        p_emp = np.full(k, 1.0 / k)
    else:
        p_emp = counts / counts.sum()
    eps = 1e-12
    z = float(np.dot(w, p_emp)) + eps
    w_hat = w / z
    Q_hat = p_emp * w_hat
    s = Q_hat.sum()
    if s > 0:
        Q_hat = Q_hat / s
    w_true = q_true / p_true
    time_sec = base_time + float(time.time() - t_extra0)
    
    result = {'q_hat': Q_hat, 'w_hat': w_hat, 'w_true': w_true, 'time_sec': time_sec}
    if train_downstream:
        extra = train_weighted_head(source_dset, target_dset, w_hat, config=config, seed=seed)
        result.update(extra)
    return result


def bs_oracle(source_dset, target_dset, p_true, q_true, train_downstream=False, config=None, seed=None):
    """Return the true target prior as an oracle reference."""
    if config is None:
        config = get_config()
    q_hat = np.array(q_true, dtype=float).copy()
    w_true = np.array(q_true, dtype=float) / np.array(p_true, dtype=float)
    w_hat = w_true.copy()
    time_sec = 0.0
    result = {'q_hat': q_hat, 'w_hat': w_hat, 'w_true': w_true, 'time_sec': time_sec}
    if train_downstream:
        extra = train_weighted_head(source_dset, target_dset, w_hat, config=config, seed=seed)
        result.update(extra)
    return result


def bs_bbsl(source_dset, target_dset, p_true, q_true, train_downstream=False, config=None, seed=None):
    """Estimate label shift with BBSL/BBSE."""
    if config is None:
        config = get_config()
    
    stats = _get_stats_cached(source_dset, target_dset, p_true, q_true, config)
    C = stats['C']
    mu_y = stats['mu_y']
    _prep = stats.get('prep_time_sec', None)
    _stat = stats.get('stat_time_sec', None)
    base_time = float(_prep) + float(_stat) if (_prep is not None and _stat is not None) else float(stats.get('shared_time_sec', 0.0))
    t_extra0 = time.time()
    w_lin = np.matmul(np.linalg.pinv(C), mu_y)
    if config.baselines.bbse.nonneg:
        w = np.maximum(w_lin, 0)
    else:
        w = w_lin
    
    # Estimate the empirical source prior from source dataset labels.
    try:
        if hasattr(source_dset, 'indices') and hasattr(source_dset, 'dataset') and hasattr(source_dset.dataset, 'targets'):
            y_src = np.array(source_dset.dataset.targets, dtype=int)[np.array(source_dset.indices, dtype=int)]
        else:
            raise AttributeError
    except Exception:
        y_src = np.array([], dtype=int)
    counts = np.bincount(y_src, minlength=C.shape[0]).astype(float)
    if counts.sum() == 0:
        p_emp = np.full(C.shape[0], 1.0 / C.shape[0])
    else:
        p_emp = counts / counts.sum()
    eps = 1e-12
    z = float(np.dot(w, p_emp)) + eps
    w_hat = w / z
    q_hat = p_emp * w_hat
    if config.baselines.bbse.renormalize:
        s = q_hat.sum()
        if s > 0:
            q_hat = q_hat / s
    w_true = q_true / p_true
    time_sec = base_time + float(time.time() - t_extra0)
    
    result = {'q_hat': q_hat, 'w_hat': w_hat, 'w_true': w_true, 'time_sec': time_sec}
    if train_downstream:
        extra = train_weighted_head(source_dset, target_dset, w_hat, config=config, seed=seed)
        result.update(extra)
    return result


def bs_naive(source_dset, target_dset, p_true, q_true, train_downstream=False, config=None, seed=None):
    """Use target predicted-label frequencies as a naive prior estimate."""
    if config is None:
        config = get_config()
    
    stats = _get_stats_cached(source_dset, target_dset, p_true, q_true, config)
    mu_y = stats['mu_y']
    k = stats['k']
    _prep = stats.get('prep_time_sec', None)
    _stat = stats.get('stat_time_sec', None)
    base_time = float(_prep) + float(_stat) if (_prep is not None and _stat is not None) else float(stats.get('shared_time_sec', 0.0))
    q_hat = np.array(mu_y, dtype=float).copy()
    q_hat = np.maximum(q_hat, 0)
    s = q_hat.sum()
    if s > 0:
        q_hat = q_hat / s
    
    # Estimate the empirical source prior from source dataset labels.
    try:
        if hasattr(source_dset, 'indices') and hasattr(source_dset, 'dataset') and hasattr(source_dset.dataset, 'targets'):
            y_src = np.array(source_dset.dataset.targets, dtype=int)[np.array(source_dset.indices, dtype=int)]
        else:
            raise AttributeError
    except Exception:
        y_src = np.array([], dtype=int)
    counts = np.bincount(y_src, minlength=k).astype(float)
    if counts.sum() == 0:
        p_emp = np.full(k, 1.0 / k)
    else:
        p_emp = counts / counts.sum()
    denom = np.maximum(p_emp, 1e-12)
    w_hat = q_hat / denom
    w_true = q_true / p_true
    time_sec = base_time
    
    result = {'q_hat': q_hat, 'w_hat': w_hat, 'w_true': w_true, 'time_sec': time_sec}
    if train_downstream:
        extra = train_weighted_head(source_dset, target_dset, np.ones_like(w_hat), config=config, seed=seed)
        result.update(extra)
    return result


def bs_mlls(source_dset, target_dset, p_true, q_true, train_downstream=False, config=None, seed=None):
    """Estimate label shift with MLLS and calibrated predictions."""
    if not HAS_ABSTENTION:
        raise ImportError("abstention package is required for MLLS method")
    if config is None:
        config = get_config()
    
    stats = _get_stats_cached(source_dset, target_dset, p_true, q_true, config)
    k = stats['k']
    logits_calib = stats.get('logits_calib', None)
    logits_T = stats.get('logits_T', None)
    logits_S_full = stats.get('logits_S_full', None)
    Y_calib = stats.get('Y_calib', None)
    base_time = float(stats.get('prep_time_sec', stats.get('shared_time_sec', 0.0)))
    
    if logits_calib is None or logits_T is None or Y_calib is None or logits_S_full is None:
        # Recompute logits if the shared-statistics cache is unavailable.
        device = torch.device(config.model.device)
        extractor = FeatureExtractor(config=config)
        Z_S, Y_S = extractor.extract_features(source_dset)
        Z_T, Y_T = extractor.extract_features(target_dset)
        split_flag = config.baselines.split_train_calibration
        ratio = config.baselines.calibration_ratio
        if split_flag:
            rng = np.random.default_rng(seed=42)
            calib_idx_list, train_idx_list = [], []
            for c in range(k):
                c_idx = np.where(Y_S == c)[0]
                if c_idx.size == 0:
                    continue
                perm = rng.permutation(c_idx)
                n_calib_c = int(len(c_idx) * ratio)
                calib_idx_list.append(perm[:n_calib_c])
                train_idx_list.append(perm[n_calib_c:])
            calib_idx = np.concatenate(calib_idx_list) if len(calib_idx_list) > 0 else np.array([], dtype=int)
            train_idx = np.concatenate(train_idx_list) if len(train_idx_list) > 0 else np.array([], dtype=int)
            Z_train, Y_train = Z_S[train_idx], Y_S[train_idx]
            Z_calib_arr, Y_calib = Z_S[calib_idx], Y_S[calib_idx]
        else:
            Z_train, Y_train = Z_S, Y_S
            Z_calib_arr, Y_calib = Z_S, Y_S
        head = _train_linear_head(Z_train, Y_train, k, device=device)
        with torch.no_grad():
            logits_calib = head(torch.from_numpy(Z_calib_arr).float().to(device)).cpu().numpy()
            logits_T = head(torch.from_numpy(Z_T).float().to(device)).cpu().numpy()
            logits_S_full = head(torch.from_numpy(Z_S).float().to(device)).cpu().numpy()
    
    # Bias-corrected temperature scaling calibration.
    t_extra0 = time.time()
    labels_onehot = np.zeros((len(Y_calib), k), dtype=float)
    if len(Y_calib) > 0:
        labels_onehot[np.arange(len(Y_calib)), Y_calib] = 1.0
    calib = abst_cal.TempScaling(
        lbfgs_kwargs={},
        verbose=False,
        bias_positions=list(range(k))
    )
    calibrate = calib(valid_preacts=logits_calib, valid_labels=labels_onehot)
    P_T = calibrate(logits_T)
    P_S = calibrate(logits_S_full)
    p_s = P_S.mean(axis=0)
    
    # EM update for the target class prior.
    q = np.array(p_s, dtype=float).copy()
    q = q / (q.sum() + 1e-12)
    max_iters = config.baselines.mlls.em_max_iters
    tol = config.baselines.mlls.em_tol
    eps = config.baselines.mlls.em_eps
    p_s_safe = np.maximum(p_s, eps)
    for _ in range(max_iters):
        prev = q.copy()
        ratio = (q / p_s_safe)
        num = P_T * ratio[None, :]
        den = np.sum(num, axis=1, keepdims=True) + eps
        r = num / den
        q = r.mean(axis=0)
        q = np.maximum(q, 0)
        s = q.sum()
        if s > 0:
            q = q / s
        if np.linalg.norm(q - prev, ord=1) < tol:
            break
    q_hat = q
    p_s_safe = np.maximum(p_s, eps)
    w_hat = q_hat / p_s_safe
    w_true = q_true / p_true
    time_sec = base_time + float(time.time() - t_extra0)
    
    result = {'q_hat': q_hat, 'w_hat': w_hat, 'w_true': w_true, 'time_sec': time_sec}
    if train_downstream:
        extra = train_weighted_head(source_dset, target_dset, w_hat, config=config, seed=seed)
        result.update(extra)
    return result


def bs_cpmcn(source_dset, target_dset, p_true, q_true, train_downstream=False, config=None, seed=None):
    """Estimate label shift with CPMCN."""
    if not HAS_ABSTENTION:
        raise ImportError("abstention package is required for CPMCN method")
    if config is None:
        config = get_config()
    
    stats = _get_stats_cached(source_dset, target_dset, p_true, q_true, config)
    k = stats['k']
    logits_calib = stats.get('logits_calib', None)
    logits_T = stats.get('logits_T', None)
    Y_calib = stats.get('Y_calib', None)
    base_time = float(stats.get('prep_time_sec', stats.get('shared_time_sec', 0.0)))
    
    if logits_calib is None or logits_T is None or Y_calib is None:
        # Recompute logits if the shared-statistics cache is unavailable.
        device = torch.device(config.model.device)
        extractor = FeatureExtractor(config=config)
        Z_S, Y_S = extractor.extract_features(source_dset)
        Z_T, Y_T = extractor.extract_features(target_dset)
        split_flag = config.baselines.split_train_calibration
        ratio = config.baselines.calibration_ratio
        if split_flag:
            rng = np.random.default_rng(seed=42)
            calib_idx_list, train_idx_list = [], []
            for c in range(k):
                c_idx = np.where(Y_S == c)[0]
                if c_idx.size == 0:
                    continue
                perm = rng.permutation(c_idx)
                n_calib_c = int(len(c_idx) * ratio)
                calib_idx_list.append(perm[:n_calib_c])
                train_idx_list.append(perm[n_calib_c:])
            calib_idx = np.concatenate(calib_idx_list) if len(calib_idx_list) > 0 else np.array([], dtype=int)
            train_idx = np.concatenate(train_idx_list) if len(train_idx_list) > 0 else np.array([], dtype=int)
            Z_train, Y_train = Z_S[train_idx], Y_S[train_idx]
            Z_calib_arr, Y_calib = Z_S[calib_idx], Y_S[calib_idx]
        else:
            Z_train, Y_train = Z_S, Y_S
            Z_calib_arr, Y_calib = Z_S, Y_S
        head = _train_linear_head(Z_train, Y_train, k, device=device)
        with torch.no_grad():
            logits_calib = head(torch.from_numpy(Z_calib_arr).float().to(device)).cpu().numpy()
            logits_T = head(torch.from_numpy(Z_T).float().to(device)).cpu().numpy()
    
    # Bias-corrected temperature scaling calibration.
    t_extra0 = time.time()
    labels_onehot = np.zeros((len(Y_calib), k), dtype=float)
    if len(Y_calib) > 0:
        labels_onehot[np.arange(len(Y_calib)), Y_calib] = 1.0
    calib = abst_cal.TempScaling(
        lbfgs_kwargs={},
        verbose=False,
        bias_positions=list(range(k))
    )
    calibrate = calib(valid_preacts=logits_calib, valid_labels=labels_onehot)
    P_T = calibrate(logits_T)
    
    # Estimate the empirical source prior from source dataset labels.
    try:
        if hasattr(source_dset, 'indices') and hasattr(source_dset, 'dataset') and hasattr(source_dset.dataset, 'targets'):
            y_src = np.array(source_dset.dataset.targets, dtype=int)[np.array(source_dset.indices, dtype=int)]
        else:
            raise AttributeError
    except Exception:
        y_src = np.array(Y_calib, dtype=int)
    counts = np.bincount(y_src, minlength=k).astype(float)
    if counts.sum() == 0:
        p_base = np.full(k, 1.0 / k)
    else:
        p_base = counts / counts.sum()
    
    # CPMCN objective over nonnegative importance weights.
    eps = 1e-12
    def objective(w):
        w = np.maximum(w, 0.0)
        g = np.maximum(P_T @ w, eps)
        R = P_T / g[:, None]
        p_q_w = R.mean(axis=0)
        diff = p_base - p_q_w
        return float(np.sum(diff * diff))
    
    w0 = np.ones(k, dtype=float)
    bounds = [(eps, None)] * k
    res = minimize(objective, w0, method='L-BFGS-B', bounds=bounds)
    w = res.x if (res is not None and hasattr(res, 'x')) else w0
    w = np.maximum(w, 0.0)
    z = float(np.dot(w, p_base)) + eps
    w = w / z
    
    q_hat = p_base * w
    s = q_hat.sum()
    if s > 0:
        q_hat = q_hat / s
    w_hat = w
    w_true = q_true / p_true
    
    time_sec = base_time + float(time.time() - t_extra0)
    
    result = {'q_hat': q_hat, 'w_hat': w_hat, 'w_true': w_true, 'time_sec': time_sec}
    if train_downstream:
        extra = train_weighted_head(source_dset, target_dset, w_hat, config=config, seed=seed)
        result.update(extra)
    return result
