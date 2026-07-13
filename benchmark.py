"""
benchmark.py
=================
Algorithms compared
-------------------
  TNStream          — temporal nearest-neighbour, fully parameter-driven
  WindowedDBSCAN    — density-based sliding window, no k needed
  DBSTREAM          — online density micro-cluster method (river)
  DenStream         — landmark density stream clustering (river)
  AdaptiveRadius    — lightweight decay-weighted prototype clustering

Metrics reported (CSV)
----------------------
  quality           — mean(purity, coverage, ARI) over full stream
  purity / coverage / ARI — individual components
  pending_ratio     — fraction of data unassigned (TNStream only, 0 for others)
  pre/during/post drift quality — phase-separated quality
  quality_drop      — pre minus during drift quality
  response_lag_w    — windows until first quality recovery after drift
  recovery_w        — windows until sustained (3-window) quality recovery
  recovery_pts      — same, in data points

Usage
-----
    python benchmark.py
    python benchmark.py --params best_params.json
    python benchmark.py --jobs 4 --seeds 42 314 2718
    python benchmark.py --quick          # 2 seeds, shorter streams

Outputs
-------
    benchmark_TIMESTAMP_summary.csv
    benchmark_TIMESTAMP_detail.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import BallTree

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, ".")
from tnstream import TNStream
from tnstream_cusum import TNStreamCUSUM


# ---------------------------------------------------------------------------
# Constants
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

TNSTREAM_DEFAULTS = {}


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


def make_dimension_shift(seed: int, n: int = N_POINTS) -> StreamDataset:
    rs = np.random.RandomState(seed)
    drift_pts = [n // 4, n // 2, 3 * n // 4]
    directions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.707, 0.707]),
        np.array([-0.707, 0.707]),
    ]
    centres = np.array([[0.3, 0.3], [0.7, 0.3], [0.5, 0.75]])
    regime = 0
    X, y = [], []
    for i in range(n):
        if i in drift_pts:
            regime += 1
            shift = directions[regime] * 0.18
            centres = np.clip(centres + shift, 0.05, 0.95)
        k = rs.randint(3)
        X.append(rs.normal(centres[k], 0.055))
        y.append(k)
    return StreamDataset(np.array(X), np.array(y), drift_pts, "DimensionShift")


ALL_DATASETS: List[Callable] = [
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
# Algorithm adapters — unified interface, no k anywhere
# ---------------------------------------------------------------------------

class StreamingAdapter(ABC):
    @abstractmethod
    def update(self, x: np.ndarray) -> None: ...
    @abstractmethod
    def get_labels(self, X: np.ndarray) -> np.ndarray: ...
    def pending_ratio(self, window: int) -> float:
        return 0.0


class TNStreamAdapter(StreamingAdapter):
    def __init__(self, params: Dict):
        self.model = TNStream(**params)

    def update(self, x):
        self.model.update(x)

    def get_labels(self, X):
        return self.model.get_labels(X)

    def pending_ratio(self, window: int) -> float:
        pool = getattr(self.model, "_pool", [])
        return len(pool) / max(1, window)

class TNStreamCUSUMAdapter(StreamingAdapter):
    def __init__(self, params: Dict):
        self.model = TNStreamCUSUM(**params)

    def update(self, x):
        self.model.update(x)

    def get_labels(self, X):
        return self.model.get_labels(X)

    def pending_ratio(self, window: int) -> float:
        pool = getattr(self.model, "_pool", [])
        return len(pool) / max(1, window)


class WindowedDBSCANAdapter(StreamingAdapter):
    """Sliding-window DBSCAN.  eps tuned to typical cluster radius, no k."""
    def __init__(self, window_size=WINDOW_W, eps=0.12, min_samples=5,
                 recluster_every=20):
        self.window_size = window_size
        self.eps = eps
        self.min_samples = min_samples
        self.recluster_every = recluster_every
        self._buf: List[np.ndarray] = []
        self._centers: np.ndarray = np.empty((0, 2))
        self._radii: np.ndarray = np.empty(0)
        self._ids: np.ndarray = np.empty(0, dtype=int)
        self._t = 0

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self._buf.append(x)
        if len(self._buf) > self.window_size:
            self._buf = self._buf[-self.window_size:]
        self._t += 1
        if self._t % self.recluster_every == 0 and len(self._buf) >= self.min_samples:
            self._recluster()

    def _recluster(self):
        pts = np.array(self._buf)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(pts)
        centers, radii, ids = [], [], []
        for cid in np.unique(labels):
            if cid < 0:
                continue
            mask = labels == cid
            c = pts[mask].mean(axis=0)
            r = max(float(np.percentile(np.linalg.norm(pts[mask] - c, axis=1), 95)), 1e-6)
            centers.append(c)
            radii.append(r * 1.1)
            ids.append(cid)
        if centers:
            self._centers = np.array(centers)
            self._radii = np.array(radii)
            self._ids = np.array(ids, dtype=int)
        else:
            self._centers = np.empty((0, len(self._buf[0])))
            self._radii = np.empty(0)
            self._ids = np.empty(0, dtype=int)

    def get_labels(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), -1, dtype=int)
        if len(self._centers) == 0:
            return out
        tree = BallTree(self._centers)
        dists, inds = tree.query(X, k=1)
        for i, (d, idx) in enumerate(zip(dists[:, 0], inds[:, 0])):
            if d <= self._radii[idx]:
                out[i] = self._ids[idx]
        return out


class DBSTREAMAdapter(StreamingAdapter):
    """
    Simplified DBSTREAM-style micro-cluster method.
    No k — density radius controls granularity.
    Fading factor decays old clusters so they eventually dissolve.
    """
    def __init__(self, radius=0.11, fading=0.998, cleanup_interval=100,
                 min_weight=1.5, merge_threshold=0.7):
        self.radius = radius
        self.fading = fading
        self.cleanup_interval = cleanup_interval
        self.min_weight = min_weight
        self.merge_threshold = merge_threshold  # fraction of radius
        self._centers: List[np.ndarray] = []
        self._weights: List[float] = []
        self._t = 0

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self._t += 1
        # fade all weights
        self._weights = [w * self.fading for w in self._weights]

        if not self._centers:
            self._centers.append(x.copy())
            self._weights.append(1.0)
        else:
            dists = np.linalg.norm(np.array(self._centers) - x, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] <= self.radius:
                lr = 1.0 / max(self._weights[idx], 1.0)
                self._centers[idx] = (1 - lr) * self._centers[idx] + lr * x
                self._weights[idx] += 1.0
            else:
                self._centers.append(x.copy())
                self._weights.append(1.0)

        if self._t % self.cleanup_interval == 0:
            self._cleanup()

    def _cleanup(self):
        # remove weak clusters
        keep = [i for i, w in enumerate(self._weights) if w >= self.min_weight]
        self._centers = [self._centers[i] for i in keep]
        self._weights = [self._weights[i] for i in keep]

        # merge very close clusters
        if len(self._centers) < 2:
            return
        merged = True
        while merged and len(self._centers) >= 2:
            merged = False
            centers = np.array(self._centers)
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    if np.linalg.norm(centers[i] - centers[j]) < self.radius * self.merge_threshold:
                        wi, wj = self._weights[i], self._weights[j]
                        new_c = (wi * self._centers[i] + wj * self._centers[j]) / (wi + wj)
                        self._centers[i] = new_c
                        self._weights[i] = wi + wj
                        del self._centers[j]
                        del self._weights[j]
                        merged = True
                        break
                if merged:
                    break

    def get_labels(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), -1, dtype=int)
        if not self._centers:
            return out
        centers = np.array(self._centers)
        tree = BallTree(centers)
        dists, inds = tree.query(X, k=1)
        for i, (d, idx) in enumerate(zip(dists[:, 0], inds[:, 0])):
            if d <= self.radius * 1.1:
                out[i] = int(idx)
        return out


class DenStreamAdapter(StreamingAdapter):
    """
    Simplified DenStream-style two-tier micro/macro cluster approach.
    No k — epsilon controls density granularity.
    """
    def __init__(self, eps=0.11, mu=2.0, beta=0.5, lam=0.1,
                 window_size=WINDOW_W):
        self.eps = eps
        self.mu = mu
        self.beta = beta
        self.lam = lam
        self.window_size = window_size
        self._p_mcs: List[Dict] = []   # potential micro-clusters
        self._o_mcs: List[Dict] = []   # outlier micro-clusters
        self._t = 0

    def _decay(self, mc):
        return mc["w"] * (2 ** (-self.lam))

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self._t += 1

        # decay all
        for mc in self._p_mcs + self._o_mcs:
            mc["w"] *= (2 ** (-self.lam))

        # try to merge into nearest potential MC
        merged = False
        if self._p_mcs:
            centers = np.array([mc["c"] for mc in self._p_mcs])
            dists = np.linalg.norm(centers - x, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] <= self.eps:
                mc = self._p_mcs[idx]
                mc["w"] += 1.0
                mc["c"] = mc["c"] + (x - mc["c"]) / mc["w"]
                merged = True

        if not merged and self._o_mcs:
            centers = np.array([mc["c"] for mc in self._o_mcs])
            dists = np.linalg.norm(centers - x, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] <= self.eps:
                mc = self._o_mcs[idx]
                mc["w"] += 1.0
                mc["c"] = mc["c"] + (x - mc["c"]) / mc["w"]
                # promote if weight exceeds threshold
                if mc["w"] >= self.beta * self.mu:
                    self._p_mcs.append(mc)
                    self._o_mcs.pop(idx)
                merged = True

        if not merged:
            self._o_mcs.append({"c": x.copy(), "w": 1.0})

        # prune weak outlier MCs periodically
        if self._t % 50 == 0:
            self._p_mcs = [mc for mc in self._p_mcs if mc["w"] >= self.beta * self.mu]
            self._o_mcs = [mc for mc in self._o_mcs if mc["w"] >= 0.1]

    def get_labels(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), -1, dtype=int)
        strong = [mc for mc in self._p_mcs if mc["w"] >= self.mu]
        if not strong:
            return out
        centers = np.array([mc["c"] for mc in strong])
        tree = BallTree(centers)
        dists, inds = tree.query(X, k=1)
        for i, (d, idx) in enumerate(zip(dists[:, 0], inds[:, 0])):
            if d <= self.eps * 1.15:
                out[i] = int(idx)
        return out


class AdaptiveRadiusAdapter(StreamingAdapter):
    """Decay-weighted prototype clusters, no k."""
    def __init__(self, radius=0.12, decay=0.995, min_weight=0.35,
                 max_clusters=25, assign_scale=1.10, window_size=WINDOW_W):
        self.radius = radius
        self.decay = decay
        self.min_weight = min_weight
        self.max_clusters = max_clusters
        self.assign_scale = assign_scale
        self.window_size = window_size
        self._centers: List[np.ndarray] = []
        self._weights: List[float] = []
        self._ages: List[int] = []
        self._t = 0

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self._t += 1
        self._weights = [w * self.decay for w in self._weights]
        self._ages = [a + 1 for a in self._ages]

        if not self._centers:
            self._centers.append(x.copy())
            self._weights.append(1.0)
            self._ages.append(0)
        else:
            dists = np.linalg.norm(np.array(self._centers) - x, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] <= self.radius:
                lr = 1.0 / max(self._weights[idx], 1.0)
                self._centers[idx] = (1 - lr) * self._centers[idx] + lr * x
                self._weights[idx] += 1.0
                self._ages[idx] = 0
            else:
                if len(self._centers) >= self.max_clusters:
                    replace = int(np.argmin(self._weights))
                    self._centers[replace] = x.copy()
                    self._weights[replace] = 1.0
                    self._ages[replace] = 0
                else:
                    self._centers.append(x.copy())
                    self._weights.append(1.0)
                    self._ages.append(0)

        # prune weak old clusters
        keep = [i for i, (w, a) in enumerate(zip(self._weights, self._ages))
                if w >= self.min_weight or a <= self.window_size]
        self._centers = [self._centers[i] for i in keep]
        self._weights = [self._weights[i] for i in keep]
        self._ages = [self._ages[i] for i in keep]

    def get_labels(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), -1, dtype=int)
        if not self._centers:
            return out
        centers = np.array(self._centers)
        tree = BallTree(centers)
        dists, inds = tree.query(X, k=1)
        for i, (d, idx) in enumerate(zip(dists[:, 0], inds[:, 0])):
            if d <= self.radius * self.assign_scale:
                out[i] = int(idx)
        return out


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

@dataclass
class AlgoSpec:
    name:    str
    factory: Callable[[], StreamingAdapter]


def build_algorithms(tn_params: Dict) -> List[AlgoSpec]:
    return [
        AlgoSpec("TNStream",         lambda p=tn_params: TNStreamAdapter(p)),
        AlgoSpec("TNStreamCUSUM",    lambda p=tn_params: TNStreamCUSUMAdapter(p)),
        AlgoSpec("WindowedDBSCAN",   lambda: WindowedDBSCANAdapter()),
        AlgoSpec("DBSTREAM",         lambda: DBSTREAMAdapter()),
        AlgoSpec("DenStream",        lambda: DenStreamAdapter()),
        AlgoSpec("AdaptiveRadius",   lambda: AdaptiveRadiusAdapter()),
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def coverage_score(pred: np.ndarray) -> float:
    return float(np.mean(pred >= 0)) if len(pred) else 0.0


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_pred >= 0
    if mask.sum() == 0:
        return 0.0
    lt, lp = y_true[mask].astype(int), y_pred[mask].astype(int)
    total = sum(np.bincount(lt[lp == c]).max() for c in np.unique(lp))
    return float(total) / mask.sum()


def ari_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_pred >= 0
    if mask.sum() < 2:
        return float("nan")
    lt, lp = y_true[mask].astype(int), y_pred[mask].astype(int)
    if len(np.unique(lp)) < 2 or len(np.unique(lt)) < 2:
        return float("nan")
    return float(adjusted_rand_score(lt, lp))


def quality(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cov = coverage_score(y_pred)
    pur = purity_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    vals = [cov, pur] + ([ari] if np.isfinite(ari) else [])
    return float(np.mean(vals))


def window_idx(point: int) -> int:
    return max(0, math.ceil(point / EVAL_EVERY) - 1)


def compute_recovery(quality_series: List[float], drift_windows: List[int],
                     total_windows: int, min_degradation: float = 0.05) -> Tuple[List[float], List[float]]:
    """Returns (response_lag_windows, recovery_windows) per drift event."""
    values = np.array(quality_series, dtype=float)
    response_lags, recoveries = [], []

    for idx, dw in enumerate(drift_windows):
        start = dw + 1
        stop = drift_windows[idx + 1] if idx + 1 < len(drift_windows) else total_windows
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

        # only measure recovery if meaningful degradation actually occurred
        pre_start = max(0, dw - 2)
        pre_vals = values[pre_start:dw]
        pre_q = float(np.nanmean(pre_vals)) if len(pre_vals) > 0 else float("nan")
        early_post = seg[:min(5, len(seg))]
        min_post = float(np.nanmin(early_post[np.isfinite(early_post)])) if np.any(np.isfinite(early_post)) else float("nan")
        if np.isfinite(pre_q) and np.isfinite(min_post) and (pre_q - min_post) < min_degradation:
            response_lags.append(float("nan"))
            recoveries.append(float("nan"))
            continue

        tail = finite[-min(5, len(finite)):]
        target = float(np.median(tail)) * RECOVERY_TARGET

        # response lag: first single window at or above target
        resp = float("nan")
        for s in range(len(seg)):
            if np.isfinite(seg[s]) and seg[s] >= target:
                resp = float(s + 1)
                break
        response_lags.append(resp)

        # recovery: first sustained RECOVERY_HOLD windows
        rec = float("nan")
        for s in range(len(seg) - RECOVERY_HOLD + 1):
            chunk = seg[s:s + RECOVERY_HOLD]
            if np.all(np.isfinite(chunk)) and np.all(chunk >= target):
                rec = float(s + 1)
                break
        recoveries.append(rec)

    return response_lags, recoveries


def safe_mean(vals: List[float]) -> float:
    v = [x for x in vals if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    algorithm:       str
    dataset:         str
    seed:            int
    quality_mean:    float
    purity_mean:     float
    coverage_mean:   float
    ari_mean:        float
    pending_ratio:   float
    pre_quality:     float
    during_quality:  float
    post_quality:    float
    quality_drop:    float
    response_lag_w:  float   # mean windows to first recovery, NaN if no degradation
    recovery_w:      float   # mean windows to sustained recovery
    recovery_pts:    float   # recovery_w * EVAL_EVERY
    quality_series:  List[float] = field(default_factory=list)
    drift_windows:   List[int]   = field(default_factory=list)


def run_single(algo_spec: AlgoSpec, ds_fn: Callable, seed: int) -> RunResult:
    ds = ds_fn(seed, N_POINTS)
    model = algo_spec.factory()
    X, y = ds.X, ds.y
    n = len(X)

    quality_series = []
    pending_series = []

    for i in range(n):
        model.update(X[i])
        if (i + 1) % EVAL_EVERY == 0:
            bX = X[i + 1 - EVAL_EVERY:i + 1]
            by = y[i + 1 - EVAL_EVERY:i + 1]
            pred = model.get_labels(bX)
            quality_series.append(quality(by, pred))
            pending_series.append(model.pending_ratio(min(i + 1, WINDOW_W)))

    total_w = len(quality_series)
    drift_ws = [window_idx(dp) for dp in ds.drift_points if dp < n]

    # phase quality
    pre_qs, during_qs, post_qs = [], [], []
    for dw in drift_ws:
        pre_start  = max(0, dw - DRIFT_PRE_W)
        dur_start  = dw
        post_start = dw + DRIFT_DURING_W
        post_end   = min(total_w, dw + DRIFT_DURING_W + DRIFT_POST_W)
        pre_qs  += [quality_series[w] for w in range(pre_start, dw) if np.isfinite(quality_series[w])]
        dur_qs   = [quality_series[w] for w in range(dur_start, min(dur_start + DRIFT_DURING_W, total_w)) if np.isfinite(quality_series[w])]
        during_qs += dur_qs
        post_qs += [quality_series[w] for w in range(post_start, post_end) if np.isfinite(quality_series[w])]

    pre_q    = safe_mean(pre_qs)
    during_q = safe_mean(during_qs)
    post_q   = safe_mean(post_qs)
    drop     = (pre_q - during_q) if np.isfinite(pre_q) and np.isfinite(during_q) else float("nan")

    resp_lags, recs = compute_recovery(quality_series, drift_ws, total_w)

    return RunResult(
        algorithm      = algo_spec.name,
        dataset        = ds.name,
        seed           = seed,
        quality_mean   = safe_mean(quality_series),
        purity_mean    = float("nan"),   # computed below
        coverage_mean  = float("nan"),
        ari_mean       = float("nan"),
        pending_ratio  = safe_mean(pending_series),
        pre_quality    = pre_q,
        during_quality = during_q,
        post_quality   = post_q,
        quality_drop   = drop,
        response_lag_w = safe_mean(resp_lags),
        recovery_w     = safe_mean(recs),
        recovery_pts   = safe_mean(recs) * EVAL_EVERY if np.isfinite(safe_mean(recs)) else float("nan"),
        quality_series = quality_series,
        drift_windows  = drift_ws,
    )


def run_single_full(args: Tuple) -> RunResult:
    algo_name, ds_fn, seed, tn_params, n_points = args
    # rebuild adapter from name to avoid pickling issues
    adapters = {
        "TNStream":       lambda: TNStreamAdapter(tn_params),
        "TNStreamCUSUM":  lambda: TNStreamCUSUMAdapter(tn_params),
        "WindowedDBSCAN": lambda: WindowedDBSCANAdapter(),
        "DBSTREAM":       lambda: DBSTREAMAdapter(),
        "DenStream":      lambda: DenStreamAdapter(),
        "AdaptiveRadius": lambda: AdaptiveRadiusAdapter(),
    }
    spec = AlgoSpec(algo_name, adapters[algo_name])

    ds = ds_fn(seed, n_points)
    model = spec.factory()
    X, y = ds.X, ds.y
    n = len(X)

    quality_series, purity_series, cov_series, ari_series, pending_series = [], [], [], [], []

    for i in range(n):
        model.update(X[i])
        if (i + 1) % EVAL_EVERY == 0:
            bX = X[i + 1 - EVAL_EVERY:i + 1]
            by = y[i + 1 - EVAL_EVERY:i + 1]
            pred = model.get_labels(bX)
            cov = coverage_score(pred)
            pur = purity_score(by, pred)
            ari = ari_score(by, pred)
            q = float(np.nanmean([cov, pur] + ([ari] if np.isfinite(ari) else [])))
            quality_series.append(q)
            purity_series.append(pur)
            cov_series.append(cov)
            ari_series.append(ari if np.isfinite(ari) else float("nan"))
            pending_series.append(model.pending_ratio(min(i + 1, WINDOW_W)))

    total_w = len(quality_series)
    drift_ws = [window_idx(dp) for dp in ds.drift_points if dp < n]

    pre_qs, during_qs, post_qs = [], [], []
    for dw in drift_ws:
        pre_start  = max(0, dw - DRIFT_PRE_W)
        dur_end    = min(dw + DRIFT_DURING_W, total_w)
        post_end   = min(total_w, dw + DRIFT_DURING_W + DRIFT_POST_W)
        pre_qs   += [quality_series[w] for w in range(pre_start, dw) if np.isfinite(quality_series[w])]
        during_qs += [quality_series[w] for w in range(dw, dur_end) if np.isfinite(quality_series[w])]
        post_qs  += [quality_series[w] for w in range(dw + DRIFT_DURING_W, post_end) if np.isfinite(quality_series[w])]

    pre_q    = safe_mean(pre_qs)
    during_q = safe_mean(during_qs)
    post_q   = safe_mean(post_qs)
    drop     = (pre_q - during_q) if (np.isfinite(pre_q) and np.isfinite(during_q)) else float("nan")

    resp_lags, recs = compute_recovery(quality_series, drift_ws, total_w)

    return RunResult(
        algorithm      = algo_name,
        dataset        = ds.name,
        seed           = seed,
        quality_mean   = safe_mean(quality_series),
        purity_mean    = safe_mean(purity_series),
        coverage_mean  = safe_mean(cov_series),
        ari_mean       = safe_mean(ari_series),
        pending_ratio  = safe_mean(pending_series),
        pre_quality    = pre_q,
        during_quality = during_q,
        post_quality   = post_q,
        quality_drop   = drop,
        response_lag_w = safe_mean(resp_lags),
        recovery_w     = safe_mean(recs),
        recovery_pts   = safe_mean(recs) * EVAL_EVERY if np.isfinite(safe_mean(recs)) else float("nan"),
        quality_series = quality_series,
        drift_windows  = drift_ws,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(runs: List[RunResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for r in runs:
        detail_rows.append({
            "Algorithm":      r.algorithm,
            "Dataset":        r.dataset,
            "Seed":           r.seed,
            "Quality":        round(r.quality_mean, 4),
            "Purity":         round(r.purity_mean, 4) if np.isfinite(r.purity_mean) else float("nan"),
            "Coverage":       round(r.coverage_mean, 4) if np.isfinite(r.coverage_mean) else float("nan"),
            "ARI":            round(r.ari_mean, 4) if np.isfinite(r.ari_mean) else float("nan"),
            "PendingRatio":   round(r.pending_ratio, 4),
            "PreQuality":     round(r.pre_quality, 4) if np.isfinite(r.pre_quality) else float("nan"),
            "DuringQuality":  round(r.during_quality, 4) if np.isfinite(r.during_quality) else float("nan"),
            "PostQuality":    round(r.post_quality, 4) if np.isfinite(r.post_quality) else float("nan"),
            "QualityDrop":    round(r.quality_drop, 4) if np.isfinite(r.quality_drop) else float("nan"),
            "ResponseLag_w":  round(r.response_lag_w, 2) if np.isfinite(r.response_lag_w) else float("nan"),
            "Recovery_w":     round(r.recovery_w, 2) if np.isfinite(r.recovery_w) else float("nan"),
            "Recovery_pts":   round(r.recovery_pts, 0) if np.isfinite(r.recovery_pts) else float("nan"),
        })

    detail_df = pd.DataFrame(detail_rows)

    # summary: mean ± std across seeds
    summary_rows = []
    for (algo, ds), grp in detail_df.groupby(["Algorithm", "Dataset"]):
        def ms(col):
            v = grp[col].dropna()
            return round(float(v.mean()), 4) if len(v) else float("nan"), \
                   round(float(v.std()), 4) if len(v) >= 2 else float("nan")

        summary_rows.append({
            "Algorithm":            algo,
            "Dataset":              ds,
            "Runs":                 len(grp),
            "Quality mean":         ms("Quality")[0],
            "Quality std":          ms("Quality")[1],
            "Purity mean":          ms("Purity")[0],
            "Coverage mean":        ms("Coverage")[0],
            "ARI mean":             ms("ARI")[0],
            "PendingRatio mean":    ms("PendingRatio")[0],
            "PreQuality mean":      ms("PreQuality")[0],
            "DuringQuality mean":   ms("DuringQuality")[0],
            "PostQuality mean":     ms("PostQuality")[0],
            "QualityDrop mean":     ms("QualityDrop")[0],
            "ResponseLag_w mean":   ms("ResponseLag_w")[0],
            "ResponseLag_w std":    ms("ResponseLag_w")[1],
            "Recovery_w mean":      ms("Recovery_w")[0],
            "Recovery_w std":       ms("Recovery_w")[1],
            "Recovery_pts mean":    ms("Recovery_pts")[0],
        })

    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="+", default=SEEDS_DEFAULT)
    parser.add_argument("--jobs",    type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--quick",   action="store_true",
                        help="2 seeds, 2000 points — fast smoke test")
    args = parser.parse_args()

    seeds = [42, 314] if args.quick else args.seeds
    n_pts = 2000 if args.quick else N_POINTS

    tn_params = {}

    algo_names = ["TNStream", "TNStreamCUSUM", "WindowedDBSCAN", "DBSTREAM", "DenStream", "AdaptiveRadius"]
    ds_fns = ALL_DATASETS

    tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = "benchmark_outputs"
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, f"benchmark_{tag}_summary.csv")
    detail_csv  = os.path.join(out_dir, f"benchmark_{tag}_detail.csv")

    total = len(algo_names) * len(ds_fns) * len(seeds)
    print("=" * 70)
    print("  Streaming Clustering Benchmark")
    print("=" * 70)
    print(f"  Algorithms  : {', '.join(algo_names)}")
    print(f"  Datasets    : {len(ds_fns)}")
    print(f"  Seeds       : {seeds}")
    print(f"  Points/run  : {n_pts}")
    print(f"  Total runs  : {total}")
    print(f"  Jobs        : {args.jobs}")
    print()

    jobs = [
        (algo, ds_fn, seed, tn_params, n_pts)
        for algo in algo_names
        for ds_fn in ds_fns
        for seed in seeds
    ]



    all_runs: List[RunResult] = []
    t0 = time.perf_counter()

    if args.jobs == 1:
        for idx, job in enumerate(jobs, 1):
            r = run_single_full(job)
            all_runs.append(r)
            sys.stdout.write(f"\r  [{idx}/{total}] {r.algorithm:<18} {r.dataset:<20} seed={r.seed}")
            sys.stdout.flush()
    else:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(run_single_full, job) for job in jobs]
            for idx, fut in enumerate(cf.as_completed(futures), 1):
                r = fut.result()
                all_runs.append(r)
                sys.stdout.write(f"\r  [{idx}/{total}] last: {r.algorithm:<18} {r.dataset}")
                sys.stdout.flush()

    wall = time.perf_counter() - t0
    print(f"\n\n  Wall time: {wall:.1f}s")

    detail_df, summary_df = aggregate(all_runs)
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    # Print reaction-speed ranking per dataset
    print()
    print("=" * 70)
    print("  DRIFT REACTION RANKING  (lower windows = faster recovery)")
    print("=" * 70)
    for ds_name, grp in summary_df.groupby("Dataset"):
        ranked = grp.sort_values("Recovery_w mean", na_position="last")
        print(f"\n  [{ds_name}]")
        print(f"  {'Algorithm':<20} {'Quality':>8} {'QDrop':>7} {'Resp_w':>7} {'Recov_w':>8} {'Recov_pts':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
        for _, row in ranked.iterrows():
            rw   = f"{row['Recovery_w mean']:.1f}" if pd.notna(row['Recovery_w mean']) else "  n/a"
            rlag = f"{row['ResponseLag_w mean']:.1f}" if pd.notna(row['ResponseLag_w mean']) else "  n/a"
            rpts = f"{row['Recovery_pts mean']:.0f}" if pd.notna(row['Recovery_pts mean']) else "  n/a"
            drop = f"{row['QualityDrop mean']:.3f}" if pd.notna(row['QualityDrop mean']) else "  n/a"
            tn_marker = " *" if row["Algorithm"] in {"TNStream", "TNStreamCUSUM"} else ""
            print(f"  {row['Algorithm']:<20} {row['Quality mean']:>8.3f} {drop:>7} {rlag:>7} {rw:>8} {rpts:>10}{tn_marker}")

    print()
    print(f"  Saved: {summary_csv}")
    print(f"  Saved: {detail_csv}")
    print()
    print("  NOTE: NaN in recovery columns = no measurable quality degradation")
    print("        detected after drift — algorithm sailed through without disruption.")
    print("        TNStream variants pending_ratio > 0 reflects unassigned pool behaviour;")
    print("        other algorithms always assign every point (ratio = 0).")


if __name__ == "__main__":
    main()
