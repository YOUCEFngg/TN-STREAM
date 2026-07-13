"""
TNStream — Temporal Nearest Neighbour Stream Clustering
Zeng et al., arXiv:2505.00359

Implements all 11 sub-algorithms from the paper.  A few things worth
knowing before diving in:

  - LSH is only used in DefineMC (§4.3.1).  kTNC is exact-search only —
    the paper explicitly bans approximate methods there (§4.1).

  - The SNN "radius" is a distance, not a count.  Specifically it's the
    max distance from xi to any shared neighbour, taken over all qualifying
    pairs.  "Qualifying" means the pair shares ≥ tk neighbours and there
    are ≥ mk such pairs in the batch.

  - 2r merge threshold in AddMCtoMacroC comes straight from Alg. 7.

  - Parameters tk and mk gate SNN pair qualification (Table 7 in paper).
  
"""

import numpy as np
from sklearn.neighbors import BallTree, KDTree
from collections import defaultdict


# ---------------------------------------------------------------------------
# LSH (signed random projections)
#
# Only used in DefineMC to speed up range searches on the unassigned pool.
# kTNC never touches this — §4.1 is explicit about it.
# ---------------------------------------------------------------------------

class LSH:
    def __init__(self, n_hashes=10, n_tables=3, seed=42):
        self.n_hashes = n_hashes
        self.n_tables = n_tables
        self.seed     = seed
        self.planes   = None
        self.tables   = None
        self.data     = None

    def fit(self, X):
        n, d = X.shape
        rng  = np.random.RandomState(self.seed)
        self.planes = rng.randn(self.n_tables, self.n_hashes, d)
        self.data   = X.copy()
        self.tables = []
        for t in range(self.n_tables):
            keys = (X @ self.planes[t].T > 0).astype(np.uint8)
            tbl  = defaultdict(list)
            for i, k in enumerate(keys):
                tbl[tuple(k)].append(i)
            self.tables.append(tbl)
        return self

    def _key(self, t, x):
        return tuple((x @ self.planes[t].T > 0).astype(np.uint8))

    def query_candidates(self, x):
        cands = set()
        for t in range(self.n_tables):
            cands.update(self.tables[t].get(self._key(t, x), []))
        return list(cands)

    def range_search(self, x, r, X_ref):
        """Return indices from X_ref that fall within distance r of x."""
        cands = self.query_candidates(x)
        if not cands:
            cands = list(range(len(X_ref)))
        return [i for i in cands if np.linalg.norm(X_ref[i] - x) <= r]

    def approx_knn(self, X_query, k):
        k = min(k, len(self.data) - 1)
        n = len(X_query)
        D = np.full((n, k), np.inf)
        I = np.full((n, k), -1, dtype=int)
        for qi, x in enumerate(X_query):
            cands = np.array(self.query_candidates(x))
            if len(cands) < k:
                cands = np.arange(len(self.data))
            dists = np.linalg.norm(self.data[cands] - x, axis=1)
            dists[(self.data[cands] == x).all(axis=1)] = np.inf
            order = np.argsort(dists)[:k]
            take  = min(k, len(order))
            D[qi, :take] = dists[order[:take]]
            I[qi, :take] = cands[order[:take]]
        return D, I

    def add_point(self, x):
        idx = len(self.data)
        self.data = np.vstack([self.data, x])
        for t in range(self.n_tables):
            self.tables[t][self._key(t, x)].append(idx)


# ---------------------------------------------------------------------------
# Tree helpers
#
# KDTree for low-d data, BallTree otherwise — 'auto' picks for you.
# These are used everywhere kTNC needs exact nearest-neighbour queries.
# ---------------------------------------------------------------------------

def _build_tree(X, mode='auto'):
    d = X.shape[1]
    if mode == 'kd' or (mode == 'auto' and d <= 10):
        return KDTree(X)
    return BallTree(X)


# ---------------------------------------------------------------------------
# Tightest Neighbours  (Definition 1, §4.1)
#
# TN(k, xi) = { xj | xj ∈ KNN(k, xi)  AND  xi ∈ KNN(k, xj) }
#
# Mutual k-NN, essentially.  No LSH here — exact only.
# ---------------------------------------------------------------------------

def compute_tightest_neighbors(X, k, mode='auto'):
    n = len(X)
    if n < 2:
        return {i: set() for i in range(n)}
    k    = min(k, n - 1)
    tree = _build_tree(X, mode)
    _, inds = tree.query(X, k=k + 1)   # +1 because the point itself comes back
    knn  = {i: set(inds[i][1:]) for i in range(n)}
    return {i: {j for j in knn[i] if i in knn[j]} for i in range(n)}


# ---------------------------------------------------------------------------
# TNOF — Tightest Neighbours Outlier Factor  (Definitions 9 & 10)
#
# TNOF(xi) = mean_dist_to_TN / |TN(k, xi)|²
# Outlier if TNOF > mean(TNOF) + alpha * std(TNOF), or TN is empty.
# ---------------------------------------------------------------------------

def compute_tnof(X, tn):
    n    = len(X)
    tnof = np.full(n, np.inf)
    for i in range(n):
        nb = list(tn[i])
        if nb:
            tnof[i] = (np.linalg.norm(X[nb] - X[i], axis=1).mean()
                       / len(nb) ** 2)
    return tnof


def detect_outliers(X, tn, alpha=1.0):
    tnof = compute_tnof(X, tn)
    fin  = tnof[np.isfinite(tnof)]
    if not len(fin):
        return np.ones(len(X), dtype=bool)
    theta = fin.mean() + alpha * fin.std()
    mask  = tnof > theta
    # anything with an empty tight neighbourhood is always an outlier
    for i in range(len(X)):
        if not tn[i]:
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# kTNC — k Tightest Neighbours Clustering  (Algorithm 1, §4.1)
#
# Union-Find over TN connected components.  Equivalent to the paper's
# while-loop formulation by Theorem 6, but simpler to implement.
# Outliers get label -1.
# ---------------------------------------------------------------------------

def ktnc(X, k, alpha=1.0, mode='auto'):
    n = len(X)
    if n == 0:
        return np.array([], dtype=int)

    tn  = compute_tightest_neighbors(X, k, mode)
    out = detect_outliers(X, tn, alpha)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]   # path compression
            x = par[x]
        return x

    for i in range(n):
        if out[i]:
            continue
        for j in tn[i]:
            if not out[j]:
                ra, rb = find(i), find(j)
                if ra != rb:
                    par[ra] = rb

    labels, roots, cid = np.full(n, -1, dtype=int), {}, 0
    for i in range(n):
        if out[i]:
            continue
        r = find(i)
        if r not in roots:
            roots[r] = cid
            cid += 1
        labels[i] = roots[r]
    return labels


# ---------------------------------------------------------------------------
# SNN adaptive radius  (§4.3.1)
#
# We need a *distance* for the MC radius, not a neighbour count.
# The paper says: take the maximum distance from xi to any shared
# neighbour, over all pairs (xi, xj) that share ≥ tk neighbours,
# provided at least mk such pairs exist in the batch.
#
# Falls back to r_max when the batch is too small or sparse to produce
# enough qualifying pairs.
# ---------------------------------------------------------------------------

def snn_radius(X, k=5, r_max=0.5, tk=5, mk=4):
    n = len(X)
    if n < 2:
        return r_max

    k    = min(k, n - 1)
    tree = _build_tree(X)
    _, inds = tree.query(X, k=k + 1)
    knn  = {i: set(inds[i][1:]) for i in range(n)}

    max_dist = 0.0
    n_qual   = 0
    for i in range(n):
        for j in knn[i]:
            shared = knn[i] & knn[j]
            if len(shared) >= tk:
                n_qual += 1
                for s in shared:
                    d = float(np.linalg.norm(X[i] - X[s]))
                    if d > max_dist:
                        max_dist = d

    if n_qual < mk:
        return r_max   # not enough evidence — fall back to the hard cap

    return float(np.clip(max_dist, 1e-4, r_max))


# ---------------------------------------------------------------------------
# MicroCluster  (§4.3.1)
#
# A sphere in feature space: centre = mean of current points, radius
# comes from SNN.  Keeps raw (timestamp, vector) pairs so we can expire
# old data as the window slides.
# ---------------------------------------------------------------------------

class MicroCluster:
    _ctr = 0   # global counter just for readable IDs

    def __init__(self, center, radius):
        self.id       = MicroCluster._ctr
        MicroCluster._ctr += 1
        self.center   = center.copy()
        self.radius   = radius
        self.points   = []    # (timestamp, vector) pairs
        self.macro_id = -1    # -1 means not yet assigned to any macro

    def add(self, t, x):
        self.points.append((t, x))
        self._recompute_center()

    def expire(self, cutoff):
        """Drop points older than cutoff, recompute centre. Returns True if still alive."""
        self.points = [(t, x) for t, x in self.points if t > cutoff]
        if self.points:
            self._recompute_center()
        return bool(self.points)

    def _recompute_center(self):
        self.center = np.mean([v for _, v in self.points], axis=0)

    def contains(self, x):
        return bool(np.linalg.norm(x - self.center) <= self.radius)

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f"MC(id={self.id}, n={len(self)}, r={self.radius:.4f}, macro={self.macro_id})"


# ---------------------------------------------------------------------------
# TNStream — the main algorithm
#
# Parameters mirror Table 7 in the paper:
#   W       : sliding window length (points)
#   N       : min points to keep a micro-cluster alive
#   r_max   : hard upper bound on MC radius
#   k       : neighbourhood size for kTNC (exact search)
#   n_micro : min MCs required to form/keep a macro-cluster
#   snn_k   : k used when estimating the SNN radius in DefineMC
#   alpha   : TNOF outlier threshold multiplier
#   tk, mk  : SNN pair qualification thresholds (Table 7)
#   backend : 'kd' | 'ball' | 'lsh' — affects DefineMC only;
#             kTNC always uses exact search regardless
# ---------------------------------------------------------------------------

class TNStream:
    def __init__(self, W=500, N=5, r_max=0.15, k=4, n_micro=2,
                 snn_k=5, alpha=1.6, tk=3, mk=2,
                 backend='ball', n_hashes=10, n_tables=3):
        self.W        = W
        self.N        = N
        self.r_max    = r_max
        self.k        = k
        self.n_micro  = n_micro
        self.snn_k    = snn_k
        self.alpha    = alpha
        self.tk       = tk
        self.mk       = mk
        self.backend  = backend
        self.n_hashes = n_hashes
        self.n_tables = n_tables

        self._t        = 0         # global timestamp
        self._pool     = []        # unassigned points: [(timestamp, vector)]
        self._mcs      = []        # list[MicroCluster]
        self._macros   = {}        # macro_id -> list[MicroCluster]
        self._next_mid = 0         # macro ID counter (never reused)

    # -- public interface ----------------------------------------------------

    def update(self, x):
        """Ingest one point.  Runs the full pipeline on each call."""
        x      = np.asarray(x, dtype=float)
        self._t += 1
        cutoff  = self._t - self.W

        self._kill_mcs(cutoff)      # expire old data, prune small MCs
        self._kill_macros()         # clean up macros that lost too many MCs

        if not self._add_to_mc(self._t, x):
            # couldn't place x — park it in the pool
            self._pool.append((self._t, x))
            self._pool = [(t, p) for t, p in self._pool if t > cutoff]

        if len(self._pool) >= self.N:
            self._define_mc()       # try to crystallise new MCs from pool

        # MC centres are kept current by MicroCluster.add() / .expire(),
        # so no separate UpdateMC pass is needed here.

        self._add_mc_to_macro()     # pull unassigned MCs into nearby macros
        self._define_macro()        # form new macros from leftover MCs
        self._update_macros()       # re-run kTNC inside each existing macro
        self._kill_macros()         # one more cleanup pass after updates

    def fit(self, X):
        for x in X:
            self.update(x)
        return self

    def get_labels(self, X):
        """
        Assign each point in X to a macro-cluster.
        Returns an int array; -1 means outlier / unassigned.
        """
        X      = np.asarray(X, dtype=float)
        labels = np.full(len(X), -1, dtype=int)
        if not self._mcs:
            return labels

        centers = np.array([mc.center for mc in self._mcs])
        radii   = [mc.radius for mc in self._mcs]
        tree    = BallTree(centers)
        dists, inds = tree.query(X, k=1)

        for i, (d, idx) in enumerate(zip(dists[:, 0], inds[:, 0])):
            mc = self._mcs[idx]
            if d <= radii[idx] and mc.macro_id >= 0:
                labels[i] = mc.macro_id
        return labels

    @property
    def micro_clusters(self):
        return self._mcs

    @property
    def macro_clusters(self):
        return self._macros

    # -- DefineMC  (§4.3.1) --------------------------------------------------

    def _define_mc(self):
        """
        Compute an SNN radius from the current pool, then do a range search
        to find dense enough neighbourhoods (≥ N points within r).
        Each qualifying neighbourhood becomes a new MicroCluster.
        """
        if len(self._pool) < self.N:
            return

        pts = np.array([x for _, x in self._pool])
        r   = snn_radius(pts,
                         k=min(self.snn_k, len(pts) - 1),
                         r_max=self.r_max,
                         tk=self.tk, mk=self.mk)
        used = np.zeros(len(pts), dtype=bool)

        if self.backend == 'lsh' and len(pts) >= 4:
            lsh = LSH(self.n_hashes, self.n_tables).fit(pts)
            for i in range(len(pts)):
                if used[i]:
                    continue
                nb = np.array(lsh.range_search(pts[i], r, pts))
                nb = nb[~used[nb]] if len(nb) else nb
                if len(nb) >= self.N:
                    self._create_mc(pts, nb, r)
                    used[nb] = True
        else:
            mode = 'kd' if self.backend == 'kd' else 'ball'
            tree = _build_tree(pts, mode)
            for i in range(len(pts)):
                if used[i]:
                    continue
                nb = tree.query_radius(pts[i:i+1], r=r)[0]
                nb = nb[~used[nb]]
                if len(nb) >= self.N:
                    self._create_mc(pts, nb, r)
                    used[nb] = True

        self._pool = [p for i, p in enumerate(self._pool) if not used[i]]

    def _create_mc(self, pts, nb_indices, r):
        center = pts[nb_indices].mean(axis=0)
        mc     = MicroCluster(center, r)
        for j in nb_indices:
            mc.add(self._pool[j][0], self._pool[j][1])
        self._mcs.append(mc)

    # -- AddtoMC  (§4.3.2) ---------------------------------------------------

    def _add_to_mc(self, t, x):
        """
        Find the closest MC centre.  If it's within that MC's radius,
        absorb x into it.  Returns True on success.
        """
        if not self._mcs:
            return False
        centers = np.array([mc.center for mc in self._mcs])
        dists   = np.linalg.norm(centers - x, axis=1)
        idx     = int(np.argmin(dists))
        if dists[idx] <= self._mcs[idx].radius:
            self._mcs[idx].add(t, x)
            return True
        return False

    # -- DefineMacroC  (§4.3.3) ----------------------------------------------

    def _define_macro(self):
        """
        Run kTNC on the centres of unassigned MCs.  Any resulting component
        with ≥ n_micro members becomes a new macro-cluster.
        """
        unassigned = [i for i, mc in enumerate(self._mcs) if mc.macro_id == -1]
        if len(unassigned) < self.n_micro:
            return

        centers = np.array([self._mcs[i].center for i in unassigned])
        mode    = 'kd' if self.backend == 'kd' else 'ball'
        labels  = ktnc(centers, k=self.k, alpha=self.alpha, mode=mode)

        groups = defaultdict(list)
        for pos, lbl in enumerate(labels):
            if lbl >= 0:
                groups[lbl].append(unassigned[pos])

        for mc_indices in groups.values():
            if len(mc_indices) >= self.n_micro:
                mid = self._next_mid
                self._next_mid += 1
                self._macros[mid] = [self._mcs[i] for i in mc_indices]
                for i in mc_indices:
                    self._mcs[i].macro_id = mid

    # -- AddMCtoMacroC  (§4.3.4) ---------------------------------------------

    def _add_mc_to_macro(self):
        """
        For each unassigned MC, find the nearest already-assigned MC.
        If that MC is within 2r (Alg. 7 threshold), pull the unassigned
        one into the same macro.
        """
        if not self._macros:
            return

        assigned = [(i, mc) for i, mc in enumerate(self._mcs) if mc.macro_id >= 0]
        if not assigned:
            return

        assigned_centers = np.array([mc.center for _, mc in assigned])
        tree = BallTree(assigned_centers)

        for mc in self._mcs:
            if mc.macro_id >= 0:
                continue
            dists, inds = tree.query(mc.center.reshape(1, -1), k=1)
            d       = float(dists[0, 0])
            ref_mc  = assigned[int(inds[0, 0])][1]
            if d <= 2 * ref_mc.radius:
                mid = ref_mc.macro_id
                mc.macro_id = mid
                self._macros.setdefault(mid, []).append(mc)

    # -- UpdateMacroC  (§4.3.6) ----------------------------------------------

    def _update_macros(self):
        """
        Re-run kTNC inside each existing macro.  Keep the largest
        surviving component if it has ≥ n_micro MCs; otherwise dissolve
        the macro.  Splinter components large enough to stand alone get
        promoted to new macros.
        """
        to_delete = []
        to_add    = {}

        for mid, mcs in list(self._macros.items()):
            # only consider MCs still alive and still belonging to this macro
            mcs = [mc for mc in mcs if mc in self._mcs and mc.macro_id == mid]
            if len(mcs) < self.n_micro:
                to_delete.append(mid)
                continue

            centers = np.array([mc.center for mc in mcs])
            mode    = 'kd' if self.backend == 'kd' else 'ball'
            labels  = ktnc(centers, k=self.k, alpha=self.alpha, mode=mode)

            groups = defaultdict(list)
            for pos, lbl in enumerate(labels):
                if lbl >= 0:
                    groups[lbl].append(mcs[pos])

            if not groups:
                to_delete.append(mid)
                continue

            # largest component stays; smaller ones that are big enough spin off
            best = max(groups.values(), key=len)
            if len(best) < self.n_micro:
                to_delete.append(mid)
                continue

            self._macros[mid] = best
            for mc in mcs:
                mc.macro_id = -1
            for mc in best:
                mc.macro_id = mid

            for comp in groups.values():
                if comp is not best and len(comp) >= self.n_micro:
                    new_mid = self._next_mid
                    self._next_mid += 1
                    to_add[new_mid] = comp

        for mid in to_delete:
            for mc in self._macros.get(mid, []):
                if mc.macro_id == mid:
                    mc.macro_id = -1
            del self._macros[mid]

        for new_mid, comp in to_add.items():
            self._macros[new_mid] = comp
            for mc in comp:
                mc.macro_id = new_mid

    # -- KillMCs  (§4.3.7) ---------------------------------------------------

    def _kill_mcs(self, cutoff):
        """
        Expire old points from every MC.  Any MC that drops below N points
        is dissolved — its remaining points go back into the pool so they
        get another chance to form a cluster later.
        """
        surviving = []
        for mc in self._mcs:
            mc.expire(cutoff)
            if len(mc) >= self.N:
                surviving.append(mc)
            else:
                # return surviving points to the pool
                for t, x in mc.points:
                    if t > cutoff:
                        self._pool.append((t, x))
                # detach from macro — _kill_macros will notice the gap
                mc.macro_id = -1
        self._mcs = surviving

    # -- KillMacroCs  (§4.3.8) -----------------------------------------------

    def _kill_macros(self):
        """
        Drop any macro whose live MC count fell below n_micro.
        Those MCs become unassigned and will be re-evaluated next cycle.
        """
        mc_set    = set(self._mcs)
        to_delete = []

        for mid, mcs in self._macros.items():
            live = [mc for mc in mcs if mc in mc_set and mc.macro_id == mid]
            if len(live) < self.n_micro:
                to_delete.append(mid)
                for mc in live:
                    mc.macro_id = -1
            else:
                self._macros[mid] = live

        for mid in to_delete:
            del self._macros[mid]
