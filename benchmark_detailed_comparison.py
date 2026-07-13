"""
benchmark_detailed_comparison.py
==================================

Compares TNStream vs TNStream+CUSUM.

Generates:
  - 1 plot per dataset: quality over time with drift markers
  - 1 summary plot: aggregated metrics across all datasets
"""

from __future__ import annotations

import os
import sys
import time
import warnings
import multiprocessing
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, ".")

try:
    from tnstream import TNStream
    from tnstream_cusum import TNStreamCUSUM
except ImportError as e:
    print(f"\n❌ ERROR: Could not import TNStream modules.")
    print(f"\n   This script must be run from your thesis directory where:")
    print(f"   - tnstream.py is located")
    print(f"   - tnstream_cusum.py is located")
    print(f"\n   Current working directory: {os.getcwd()}")
    print(f"   Error: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants (from benchmark.py)
# ---------------------------------------------------------------------------

SEEDS_DEFAULT     = [42, 314, 2718, 9001, 20250]
N_POINTS          = 5_000
EVAL_EVERY        = 50
WINDOW_W          = 500
RECOVERY_HOLD     = 3
RECOVERY_TARGET   = 0.90
DRIFT_PRE_W       = 2
DRIFT_DURING_W    = 1
DRIFT_POST_W      = 3

# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamDataset:
    X:            np.ndarray
    y:            np.ndarray
    drift_points: List[int]
    name:         str


def make_abrupt_shift(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    drift_every = n // 4
    drift_points = [drift_every * i for i in range(1, 4)]
    regimes = [
        np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]),
        np.array([[0.8, 0.8], [0.2, 0.8], [0.5, 0.2]]),
        np.array([[0.1, 0.5], [0.9, 0.5], [0.5, 0.1]]),
        np.array([[0.3, 0.7], [0.7, 0.3], [0.5, 0.5]]),
    ]
    X, y, regime = [], [], 0
    for i in range(n):
        if i in drift_points:
            regime += 1
        k = rs.randint(3)
        X.append(rs.normal(regimes[regime][k], 0.055))
        y.append(k)
    return StreamDataset(np.array(X), np.array(y), drift_points, "AbruptShift")


def make_abrupt_split(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    drift_pt = n // 2
    X, y = [], []
    for i in range(n):
        if i < drift_pt:
            k = rs.randint(2)
            centres = [[0.25, 0.5], [0.75, 0.5]]
            X.append(rs.normal(centres[k], 0.07))
            y.append(k)
        else:
            k = rs.randint(3)
            centres = [[0.25, 0.5], [0.65, 0.3], [0.85, 0.7]]
            X.append(rs.normal(centres[k], 0.06))
            y.append(k)
    return StreamDataset(np.array(X), np.array(y), [drift_pt], "AbruptSplit")


def make_abrupt_merge(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    drift_pt = n // 2
    X, y = [], []
    for i in range(n):
        if i < drift_pt:
            k = rs.randint(3)
            centres = [[0.2, 0.5], [0.8, 0.5], [0.5, 0.15]]
            X.append(rs.normal(centres[k], 0.055))
            y.append(k)
        else:
            k = rs.randint(2)
            centres = [[0.5, 0.5], [0.5, 0.15]]
            X.append(rs.normal(centres[k], 0.09))
            y.append(k)
    return StreamDataset(np.array(X), np.array(y), [drift_pt], "AbruptMerge")


def make_gradual_rbf(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    base = np.array([[0.20, 0.25], [0.75, 0.25], [0.45, 0.75]])
    regime_offset = np.array([[0.04, 0.02], [-0.03, 0.05], [0.02, -0.04]])
    drift_every = n // 4
    drift_points = [drift_every * i for i in range(1, 4)]
    centres = base.copy()
    regime_sign = 1.0
    X, y = [], []
    for i in range(n):
        if i > 0 and i % drift_every == 0:
            regime_sign *= -1.0
            centres = base + regime_sign * regime_offset
        phase = 2.0 * np.pi * (i / max(n, 1))
        smooth = np.array([
            [0.02 * np.sin(phase), 0.015 * np.cos(phase)],
            [0.015 * np.cos(phase), 0.02 * np.sin(phase)],
            [0.018 * np.sin(phase + 1.2), 0.016 * np.cos(phase + 0.6)],
        ])
        live = centres + smooth
        k = rs.randint(3)
        X.append(rs.normal(live[k], 0.05))
        y.append(k)
    return StreamDataset(np.array(X), np.array(y), drift_points, "GradualRBF")


def make_birth_death(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    all_centres = rs.uniform(0.12, 0.88, (7, 2))
    active = [0, 1, 2]
    weights = rs.dirichlet(np.ones(3))
    drift_every = n // 4
    drift_points = [drift_every * i for i in range(1, 4)]
    last_drift = -10000
    X, y = [], []
    for i in range(n):
        if i > 0 and i % drift_every == 0:
            retire = int(rs.choice(active))
            active.remove(retire)
            new_id = int(rs.randint(len(all_centres)))
            while new_id in active:
                new_id = int(rs.randint(len(all_centres)))
            active.append(new_id)
            weights = rs.dirichlet(np.ones(len(active)))
            last_drift = i
        k_local = int(rs.choice(len(active), p=weights))
        k = int(active[k_local])
        bursty = 0 <= (i - last_drift) < 120
        spread = 0.05 if not bursty else 0.085
        x = rs.uniform(0, 1, 2) if (bursty and rs.rand() < 0.18) else rs.normal(all_centres[k], spread)
        X.append(x)
        y.append(k)
    return StreamDataset(np.array(X), np.array(y), drift_points, "BirthDeath")


def make_reappearing_modes(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    centres = np.array([[0.18, 0.22], [0.42, 0.74], [0.72, 0.20], [0.80, 0.76]])
    mode_pairs = [(0, 1), (2, 3), (0, 3), (1, 2)]
    drift_every = n // 4
    drift_points = [drift_every * i for i in range(1, 4)]
    pair_idx = 0
    X, y = [], []
    for i in range(n):
        if i > 0 and i % drift_every == 0:
            pair_idx = (pair_idx + 1) % len(mode_pairs)
        a, b = mode_pairs[pair_idx]
        k = a if rs.rand() < 0.55 else b
        X.append(rs.normal(centres[k], 0.052))
        y.append(k)
    return StreamDataset(np.array(X), np.array(y), drift_points, "ReappearingModes")


def make_variable_density(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    centres = np.array([[0.18, 0.18], [0.82, 0.24], [0.50, 0.78]])
    spreads = np.array([0.03, 0.06, 0.09])
    drift_every = n // 4
    drift_points = [drift_every * i for i in range(1, 4)]
    theta = 0.0
    X, y = [], []
    for i in range(n):
        if i > 0 and i % drift_every == 0:
            theta += np.pi / 5.0
            spreads = np.roll(spreads, 1)
            centres = np.clip(centres + rs.normal(0, 0.04, centres.shape), 0.06, 0.94)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        k = rs.randint(3)
        d = spreads[k]
        cov = rot @ np.diag([d**2, (1.6*d)**2]) @ rot.T
        X.append(rs.multivariate_normal(centres[k], cov))
        y.append(k)
    return StreamDataset(np.array(X), np.array(y), drift_points, "VariableDensity")


def make_rotating_moons(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    drift_every = n // 4
    drift_points = [drift_every * i for i in range(1, 4)]
    angle = 0.0
    X, y = [], []
    for i in range(n):
        if i > 0 and i % drift_every == 0:
            angle += np.pi / 3.0
        k = rs.randint(2)
        t = rs.uniform(0, np.pi)
        if k == 0:
            pt = np.array([np.cos(t), np.sin(t)]) * 0.3 + np.array([0.5, 0.5])
        else:
            pt = np.array([1 - np.cos(t), 0.5 - np.sin(t)]) * 0.3 + np.array([0.35, 0.35])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pt = rot @ (pt - 0.5) + 0.5
        pt += rs.normal(0, 0.025, 2)
        X.append(pt)
        y.append(k)
    return StreamDataset(np.array(X), np.array(y), drift_points, "RotatingMoons")


def make_speed_variant(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    drift_pts = [n // 3, 2 * n // 3]
    centres = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]])
    X, y = [], []
    for i in range(n):
        centres += rs.normal(0, 0.0003, centres.shape)
        centres = np.clip(centres, 0.05, 0.95)
        if i in drift_pts:
            centres += rs.choice([-1, 1], centres.shape) * rs.uniform(0.08, 0.15, centres.shape)
            centres = np.clip(centres, 0.05, 0.95)
        k = rs.randint(3)
        X.append(rs.normal(centres[k], 0.05))
        y.append(k)
    return StreamDataset(np.array(X), np.array(y), drift_pts, "SpeedVariant")


ALL_DATASETS = [
    make_abrupt_shift,
    make_abrupt_split,
    make_abrupt_merge,
    make_gradual_rbf,
    make_birth_death,
    make_reappearing_modes,
    make_variable_density,
    make_rotating_moons,
    make_speed_variant,
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def safe_mean(arr):
    """Mean ignoring non-finite values."""
    finite = [v for v in arr if np.isfinite(v)]
    return np.mean(finite) if finite else float("nan")


def compute_purity(labels_true, labels_pred, unassigned_mask=None):
    """Purity: fraction of points in largest GT cluster per pred cluster."""
    if len(np.unique(labels_pred)) == 0:
        return float("nan")
    if unassigned_mask is not None and np.all(unassigned_mask):
        return float("nan")
    
    if unassigned_mask is not None:
        mask = ~unassigned_mask
        labels_true = labels_true[mask]
        labels_pred = labels_pred[mask]
    
    if len(labels_pred) == 0:
        return float("nan")
    
    purity_scores = []
    for pred_label in np.unique(labels_pred):
        mask_pred = labels_pred == pred_label
        true_in_pred = labels_true[mask_pred]
        if len(true_in_pred) > 0:
            majority_count = np.bincount(true_in_pred).max()
            purity_scores.append(majority_count / len(true_in_pred))
    
    return np.mean(purity_scores) if purity_scores else float("nan")


def compute_coverage(labels_true, labels_pred, unassigned_mask=None):
    """Coverage: fraction of GT points found in any pred cluster."""
    if unassigned_mask is not None:
        assigned = np.sum(~unassigned_mask)
        total = len(labels_true)
        return assigned / total if total > 0 else float("nan")
    return 1.0


def compute_quality(labels_true, labels_pred, unassigned_mask=None):
    """Quality = mean(purity, coverage, ARI)."""
    purity = compute_purity(labels_true, labels_pred, unassigned_mask)
    coverage = compute_coverage(labels_true, labels_pred, unassigned_mask)
    ari = adjusted_rand_score(labels_true, labels_pred) if len(np.unique(labels_pred)) > 0 else float("nan")
    return np.nanmean([purity, coverage, ari])


# ---------------------------------------------------------------------------
# Benchmark-style recovery computation (STRICT)
# ---------------------------------------------------------------------------

def compute_recovery_strict(values, drift_windows, total_windows, min_degradation=0.02):
    """
    Compute response lag and recovery lag using benchmark.py's strict definition.
    
    Response lag: First window where quality ≥ target
    Recovery lag: First window of a RECOVERY_HOLD-window sustained block ≥ target
    Target: Median of post-drift tail * RECOVERY_TARGET (0.90)
    Only measures if meaningful degradation (≥ 2%) occurred.
    """
    response_lags = []
    recoveries = []
    
    for idx, dw in enumerate(drift_windows):
        # Segment between this drift and the next (or end)
        start = dw
        stop = drift_windows[idx + 1] if idx + 1 < len(drift_windows) else total_windows
        
        # Need enough room to measure recovery
        if stop - start < RECOVERY_HOLD + 2:
            response_lags.append(float("nan"))
            recoveries.append(float("nan"))
            continue
        
        seg = values[start:stop]
        finite = seg[np.isfinite(seg)]
        if len(finite) < RECOVERY_HOLD + 2:
            response_lags.append(float("nan"))
            recoveries.append(float("nan"))
            continue
        
        # Check if meaningful degradation occurred
        pre_start = max(0, dw - DRIFT_PRE_W)
        pre_vals = values[pre_start:dw]
        pre_q = float(np.nanmean(pre_vals)) if len(pre_vals) > 0 else float("nan")
        
        early_post = seg[:min(5, len(seg))]
        min_post = float(np.nanmin(early_post[np.isfinite(early_post)])) if np.any(np.isfinite(early_post)) else float("nan")
        
        if np.isfinite(pre_q) and np.isfinite(min_post) and (pre_q - min_post) < min_degradation:
            response_lags.append(float("nan"))
            recoveries.append(float("nan"))
            continue
        
        # Target: median of tail * RECOVERY_TARGET
        tail = finite[-min(5, len(finite)):]
        target = float(np.median(tail)) * RECOVERY_TARGET
        
        # Response lag: first window ≥ target
        resp = float("nan")
        for s in range(len(seg)):
            if np.isfinite(seg[s]) and seg[s] >= target:
                resp = float(s + 1)
                break
        response_lags.append(resp)
        
        # Recovery lag: first sustained RECOVERY_HOLD-window block ≥ target
        rec = float("nan")
        for s in range(len(seg) - RECOVERY_HOLD + 1):
            chunk = seg[s:s + RECOVERY_HOLD]
            if np.all(np.isfinite(chunk)) and np.all(chunk >= target):
                rec = float(s + 1)
                break
        recoveries.append(rec)
    
    return response_lags, recoveries


# ---------------------------------------------------------------------------
# Run single algorithm on single dataset
# ---------------------------------------------------------------------------

@dataclass
class DriftMetrics:
    algorithm: str
    dataset: str
    seed: int
    quality_series: np.ndarray
    drift_windows: List[int]
    quality_mean: float
    quality_drop: float
    response_lag_w: float
    recovery_w: float


def run_algorithm(algo_class, ds: StreamDataset, seed: int, disable_tqdm: bool = False) -> DriftMetrics:
    """Run one algorithm on one dataset, return quality time-series + drift windows."""
    
    X = ds.X
    y = ds.y
    n = len(X)
    
    algo = algo_class()
    
    quality_series = []
    
    window_idx = lambda pt: pt // EVAL_EVERY
    
    # Feed data cumulatively to the streaming algorithm
    for i in tqdm(range(n), desc=f"Processing {ds.name} with {algo_class.__name__}", unit="point", disable=disable_tqdm):
        algo.update(X[i])
        
        # Evaluate every EVAL_EVERY points
        if (i + 1) % EVAL_EVERY == 0:
            window_start = max(0, i + 1 - EVAL_EVERY)
            window_data = X[window_start:i + 1]
            window_true = y[window_start:i + 1]
            
            labels_pred = algo.get_labels(window_data)
            
            # Compute quality on valid points
            if len(labels_pred) > 0 and np.any(labels_pred >= 0):
                valid_mask = labels_pred >= 0
                if np.any(valid_mask):
                    quality = compute_quality(window_true[valid_mask], labels_pred[valid_mask])
                    quality_series.append(quality if np.isfinite(quality) else float("nan"))
                else:
                    quality_series.append(float("nan"))
            else:
                quality_series.append(float("nan"))
    
    quality_series = np.array(quality_series)
    drift_ws = [window_idx(dp) for dp in ds.drift_points if dp < n]
    
    # Compute metrics using BENCHMARK's strict recovery definition
    total_w = len(quality_series)
    
    # Phase quality
    pre_qs, during_qs = [], []
    for dw in drift_ws:
        pre_start = max(0, dw - DRIFT_PRE_W)
        dur_end = min(dw + DRIFT_DURING_W, total_w)
        pre_qs += [quality_series[w] for w in range(pre_start, dw) if np.isfinite(quality_series[w])]
        during_qs += [quality_series[w] for w in range(dw, dur_end) if np.isfinite(quality_series[w])]
    
    pre_q = safe_mean(pre_qs)
    during_q = safe_mean(during_qs)
    quality_drop = (pre_q - during_q) if (np.isfinite(pre_q) and np.isfinite(during_q)) else float("nan")
    
    # Recovery lags using BENCHMARK's strict computation
    resp_lags, recs = compute_recovery_strict(quality_series, drift_ws, total_w)
    
    return DriftMetrics(
        algorithm=algo_class.__name__,
        dataset=ds.name,
        seed=seed,
        quality_series=quality_series,
        drift_windows=drift_ws,
        quality_mean=safe_mean(quality_series),
        quality_drop=quality_drop,
        response_lag_w=safe_mean(resp_lags),
        recovery_w=safe_mean(recs),
    )


# Worker function for parallel execution
def _worker_run(algo_name: str, ds_fn: Callable, seed: int) -> DriftMetrics:
    """Worker function for ProcessPoolExecutor."""
    from tnstream import TNStream
    from tnstream_cusum import TNStreamCUSUM
    
    algo_class = TNStream if algo_name == "TNStream" else TNStreamCUSUM
    ds = ds_fn(seed)
    
    return run_algorithm(algo_class, ds, seed, disable_tqdm=True)


@dataclass
class AggregatedMetrics:
    algorithm: str
    dataset: str
    quality_mean: float
    quality_std: float
    quality_drop_mean: float
    quality_drop_std: float
    response_lag_w_mean: float
    response_lag_w_std: float
    recovery_w_mean: float
    recovery_w_std: float
    quality_series_mean: np.ndarray
    quality_series_std: np.ndarray


def aggregate_metrics(all_metrics: List[DriftMetrics]) -> Dict[Tuple[str, str], AggregatedMetrics]:
    """Aggregate metrics across seeds for each algorithm-dataset combination."""
    aggregated = {}
    
    grouped = defaultdict(list)
    for metric in all_metrics:
        key = (metric.algorithm, metric.dataset)
        grouped[key].append(metric)
    
    for (algo, ds), metrics_list in grouped.items():
        quality_means = [m.quality_mean for m in metrics_list]
        quality_drops = [m.quality_drop for m in metrics_list if np.isfinite(m.quality_drop)]
        response_lags = [m.response_lag_w for m in metrics_list if np.isfinite(m.response_lag_w)]
        recovery_ws = [m.recovery_w for m in metrics_list if np.isfinite(m.recovery_w)]
        
        quality_mean = safe_mean(quality_means)
        quality_std = np.std([q for q in quality_means if np.isfinite(q)]) if len([q for q in quality_means if np.isfinite(q)]) > 1 else 0.0
        
        quality_drop_mean = safe_mean(quality_drops) if quality_drops else float("nan")
        quality_drop_std = np.std(quality_drops) if len(quality_drops) > 1 else 0.0
        
        response_lag_mean = safe_mean(response_lags) if response_lags else float("nan")
        response_lag_std = np.std(response_lags) if len(response_lags) > 1 else 0.0
        
        recovery_w_mean = safe_mean(recovery_ws) if recovery_ws else float("nan")
        recovery_w_std = np.std(recovery_ws) if len(recovery_ws) > 1 else 0.0
        
        # Aggregate quality series
        max_len = max(len(m.quality_series) for m in metrics_list)
        quality_series_agg = np.full((len(metrics_list), max_len), np.nan)
        
        for i, m in enumerate(metrics_list):
            quality_series_agg[i, :len(m.quality_series)] = m.quality_series
        
        quality_series_mean = np.nanmean(quality_series_agg, axis=0)
        quality_series_std = np.nanstd(quality_series_agg, axis=0)
        
        aggregated[(algo, ds)] = AggregatedMetrics(
            algorithm=algo,
            dataset=ds,
            quality_mean=quality_mean,
            quality_std=quality_std,
            quality_drop_mean=quality_drop_mean,
            quality_drop_std=quality_drop_std,
            response_lag_w_mean=response_lag_mean,
            response_lag_w_std=response_lag_std,
            recovery_w_mean=recovery_w_mean,
            recovery_w_std=recovery_w_std,
            quality_series_mean=quality_series_mean,
            quality_series_std=quality_series_std,
        )
    
    return aggregated


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dataset_comparison_aggregated(tnstream_agg: AggregatedMetrics, 
                                       cusum_agg: AggregatedMetrics,
                                       output_path: str, 
                                       drift_points: List[int]):
    """
    Plot quality time-series for both algorithms on a dataset with drift markers.
    Shows mean ± std across seeds.
    """
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Convert drift point indices to window indices
    window_idx = lambda pt: pt // EVAL_EVERY
    drift_windows = [window_idx(dp) for dp in drift_points]
    
    # X-axis: windows
    x_tn = np.arange(len(tnstream_agg.quality_series_mean))
    x_cs = np.arange(len(cusum_agg.quality_series_mean))
    
    # Plot quality series with confidence bands
    ax.plot(x_tn, tnstream_agg.quality_series_mean, label="TNStream", 
            color="#2E86AB", linewidth=2.5, marker='o', markersize=3, alpha=0.85)
    ax.fill_between(x_tn, 
                    tnstream_agg.quality_series_mean - tnstream_agg.quality_series_std,
                    tnstream_agg.quality_series_mean + tnstream_agg.quality_series_std,
                    color="#2E86AB", alpha=0.15)
    
    ax.plot(x_cs, cusum_agg.quality_series_mean, label="TNStream+CUSUM", 
            color="#A23B72", linewidth=2.5, marker='s', markersize=3, alpha=0.85)
    ax.fill_between(x_cs,
                    cusum_agg.quality_series_mean - cusum_agg.quality_series_std,
                    cusum_agg.quality_series_mean + cusum_agg.quality_series_std,
                    color="#A23B72", alpha=0.15)
    
    # Mark drift events
    colors_drift = ["#E63946", "#F77F00", "#06A77D"]
    for idx, drift_window in enumerate(drift_windows):
        ax.axvline(drift_window, color=colors_drift[idx % len(colors_drift)], 
                  linestyle='--', linewidth=3, alpha=0.9,
                  label=f"Drift {idx + 1}" if idx == 0 else "")
    
    ax.set_xlabel("Window (50 points per window)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Quality (Purity/Coverage/ARI)", fontsize=11, fontweight='bold')
    ax.set_title(
        f"{tnstream_agg.dataset} — TNStream vs TNStream+CUSUM Drift Reaction (BENCHMARK Metrics)\n"
        f"Quality: TN={tnstream_agg.quality_mean:.3f}±{tnstream_agg.quality_std:.3f}, "
        f"CUSUM={cusum_agg.quality_mean:.3f}±{cusum_agg.quality_std:.3f}\n"
        f"Recovery (w, sustained): TN={tnstream_agg.recovery_w_mean:.1f}±{tnstream_agg.recovery_w_std:.1f}, "
        f"CUSUM={cusum_agg.recovery_w_mean:.1f}±{cusum_agg.recovery_w_std:.1f}",
        fontsize=12, fontweight='bold', pad=15
    )
    
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_summary_aggregated(aggregated_metrics: Dict[Tuple[str, str], AggregatedMetrics], 
                           output_path: str):
    """
    Summary plot: comparison metrics across datasets.
    Shows quality drop, recovery lag (sustained), and average quality with error bars.
    """
    
    dataset_names = sorted(set(ds for (_, ds) in aggregated_metrics.keys()))
    algo_names = ["TNStream", "TNStreamCUSUM"]
    
    quality_means = {algo: [] for algo in algo_names}
    quality_stds = {algo: [] for algo in algo_names}
    quality_drops = {algo: [] for algo in algo_names}
    quality_drop_stds = {algo: [] for algo in algo_names}
    recovery_ws = {algo: [] for algo in algo_names}
    recovery_w_stds = {algo: [] for algo in algo_names}
    
    for ds in dataset_names:
        for algo in algo_names:
            key = (algo, ds)
            if key in aggregated_metrics:
                agg = aggregated_metrics[key]
                quality_means[algo].append(agg.quality_mean)
                quality_stds[algo].append(agg.quality_std)
                quality_drops[algo].append(agg.quality_drop_mean)
                quality_drop_stds[algo].append(agg.quality_drop_std)
                recovery_ws[algo].append(agg.recovery_w_mean)
                recovery_w_stds[algo].append(agg.recovery_w_std)
            else:
                quality_means[algo].append(float("nan"))
                quality_stds[algo].append(0.0)
                quality_drops[algo].append(float("nan"))
                quality_drop_stds[algo].append(0.0)
                recovery_ws[algo].append(float("nan"))
                recovery_w_stds[algo].append(0.0)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    # Plot 1: Overall Quality
    ax = axes[0]
    bars1 = ax.bar(x - width/2, quality_means["TNStream"], width, 
                   yerr=quality_stds["TNStream"], label="TNStream", 
                   color="#2E86AB", alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, quality_means["TNStreamCUSUM"], width,
                   yerr=quality_stds["TNStreamCUSUM"], label="TNStream+CUSUM", 
                   color="#A23B72", alpha=0.8, capsize=3)
    ax.set_ylabel("Quality", fontsize=11, fontweight='bold')
    ax.set_title("Overall Quality", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Plot 2: Quality Drop During Drift
    ax = axes[1]
    bars1 = ax.bar(x - width/2, quality_drops["TNStream"], width,
                   yerr=quality_drop_stds["TNStream"], label="TNStream", 
                   color="#2E86AB", alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, quality_drops["TNStreamCUSUM"], width,
                   yerr=quality_drop_stds["TNStreamCUSUM"], label="TNStream+CUSUM", 
                   color="#A23B72", alpha=0.8, capsize=3)
    ax.set_ylabel("Quality Drop", fontsize=11, fontweight='bold')
    ax.set_title("Quality Drop During Drift (↓ is better)", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Recovery Windows (SUSTAINED)
    ax = axes[2]
    bars1 = ax.bar(x - width/2, recovery_ws["TNStream"], width,
                   yerr=recovery_w_stds["TNStream"], label="TNStream", 
                   color="#2E86AB", alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, recovery_ws["TNStreamCUSUM"], width,
                   yerr=recovery_w_stds["TNStreamCUSUM"], label="TNStream+CUSUM", 
                   color="#A23B72", alpha=0.8, capsize=3)
    ax.set_ylabel("Recovery Windows", fontsize=11, fontweight='bold')
    ax.set_title("Sustained Recovery Lag (↓ is better)", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle("Drift Reaction Comparison: TNStream vs TNStream+CUSUM (BENCHMARK Metrics, 5 seeds)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run comparison and generate plots."""
    
    seeds = SEEDS_DEFAULT
    
    output_dir = "benchmark_detailed_comparison_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("  Detailed Drift Comparison: TNStream vs TNStream+CUSUM")
    print("  (Using BENCHMARK's strict recovery definition)")
    print("=" * 70)
    print(f"  Seeds: {seeds}")
    print(f"  Datasets: {[ds(42).name for ds in ALL_DATASETS]}")
    print(f"  Recovery metric: Sustained {RECOVERY_HOLD}-window block at target")
    print(f"  Target: Post-drift tail median * {RECOVERY_TARGET}")
    print()
    
    all_metrics = []
    
    # Prepare all work items
    work_items = []
    for ds_fn in ALL_DATASETS:
        for seed in seeds:
            work_items.append(("TNStream", ds_fn, seed))
            work_items.append(("TNStreamCUSUM", ds_fn, seed))
    
    total_runs = len(work_items)
    
    num_workers = min(4, multiprocessing.cpu_count())
    print(f"  Using {num_workers} CPU cores for parallel processing\n")
    
    # Run all work items in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_worker_run, algo_name, ds_fn, seed): (algo_name, ds_fn, seed)
            for algo_name, ds_fn, seed in work_items
        }
        
        with tqdm(total=total_runs, desc="Processing datasets", unit="run") as pbar:
            for future in as_completed(futures):
                try:
                    metrics = future.result()
                    all_metrics.append(metrics)
                    pbar.update(1)
                except Exception as e:
                    algo_name, ds_fn, seed = futures[future]
                    print(f"\n❌ Error processing {ds_fn(seed).name} (seed={seed}) with {algo_name}: {e}")
                    pbar.update(1)
    
    print()
    
    # Aggregate results across seeds
    aggregated_metrics = aggregate_metrics(all_metrics)
    
    # Generate plots for each dataset
    for ds_fn in ALL_DATASETS:
        ds = ds_fn(42)
        ds_name = ds.name
        drift_points = ds.drift_points
        tn_agg = aggregated_metrics.get(("TNStream", ds_name))
        cusum_agg = aggregated_metrics.get(("TNStreamCUSUM", ds_name))
        
        if tn_agg and cusum_agg:
            print(f"\n  Plotting: {ds_name}")
            output_plot = os.path.join(output_dir, f"benchmark_drift_{ds_name}.png")
            plot_dataset_comparison_aggregated(tn_agg, cusum_agg, output_plot, drift_points)
    
    # Summary plot
    print("\n  Generating summary plot ...", end="", flush=True)
    summary_plot = os.path.join(output_dir, "benchmark_drift_summary.png")
    plot_summary_aggregated(aggregated_metrics, summary_plot)
    print(" ✓")
    
    print()
    print("=" * 70)
    print(f"  All outputs saved to: {output_dir}/")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
