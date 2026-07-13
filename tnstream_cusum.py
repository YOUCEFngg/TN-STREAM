"""
TNStreamCUSUM — TNStream + AbruptCUSUM, structural-adaptation variant (v4)

Design history
--------------
v1 — hard reset on drift: wiped all state, forcing a cold-start ladder.
     Fast detection, terrible quality.
v2 — uniform window compression: compressed effective W during a burst.
     Improved response lag but hurt quality; compression killed nascent
     new-regime MCs alongside stale old-regime ones.
v3 — selective window compression: protected post-drift MCs from the burst
     cutoff.  Logically correct but parameters proved impossible to tune —
     burst_steps too small = invisible; too large = burst becomes the dip.

v4 (this file) — no window manipulation at all.  Instead:

  1. Better surprise signal
     Distance to nearest MC normalised by that MC's radius.  A point that
     lands inside its nearest MC scores < 1 (unsurprising); one that lands
     far outside scores >> 1 (genuinely surprising).  Returning 0 when no
     MCs exist suppresses cold-start false alarms entirely.

  2. Macro-stability gate
     CUSUM is not fed until at least one macro cluster exists.  Before that
     the model is still warming up and the surprise signal is meaningless.

  3. Temporary n_micro relaxation on drift
     On confirmed drift, the minimum MC count required to form/keep a macro
     is dropped by 1 for `relax_steps` updates.  This lets small groups of
     new-regime MCs coalesce into macros faster, without touching the window
     or destroying existing structure.  Old macros dissolve naturally as their
     MCs expire through the normal window — no forced surgery.

  4. Temporary pool sensitivity boost on drift
     The pool threshold for MC crystallisation is dropped to `N_fast` for
     `relax_steps` updates.  New-regime points form MCs sooner, accelerating
     the pipeline from raw points → MCs → macros → labels.

Together 3 and 4 attack the actual bottleneck (label latency after drift)
without touching the stability of what already exists.

Parameters (CUSUM-specific)
---------------------------
cusum_k_lambda    : adaptive lambda multiplier               (default 3.8)
                   Lowered from 4.5 — catches the early shoulder of a drift
                   signal; macro-gate still prevents cold-start FP.
cusum_k_slope     : slope threshold multiplier               (default 3.5)
cusum_ema_alpha   : EMA forgetting for reference mean         (default 0.07)
                   Raised from 0.05 — baseline adapts faster during stable
                   phases, sharpening contrast when real drift arrives.
cusum_delta       : dead-band in standardised units          (default 0.01)
cusum_confirm     : consecutive steps to confirm drift       (default 2)
                   Lowered from 3 — saves ~2 steps of lag per detection;
                   FP risk absorbed by the macro-stability gate.
cusum_min_samples : warmup guard before detector can fire    (default 50)

Parameters (adaptation-specific)
---------------------------------
relax_steps  : how many updates to hold the relaxed thresholds (default 180)
               Raised from 150 (approx W/2.8). More runway for new-regime
               MCs to consolidate before thresholds tighten again.
n_micro_relax: n_micro override during relaxation             (default 1)
               Drop to 1 so even a single new-regime MC can anchor a macro.
N_fast       : pool threshold during relaxation               (default 2)
               Lowered from 3 — minimum allowed (2 <= N_fast < N).
               New-regime points crystallise into MCs in half the time;
               only active post-detection so stable-phase quality is unaffected.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from tnstream import TNStream
from OCDD import AbruptCUSUM


class TNStreamCUSUM(TNStream):
    """
    TNStream with an AbruptCUSUM watchdog that reacts to confirmed abrupt
    drift by temporarily relaxing macro-formation and MC-crystallisation
    thresholds, accelerating label recovery without destabilising existing
    structure.

    Public interface — identical to TNStream:
        .update(x)
        .get_labels(X)
        .micro_clusters  /  .macro_clusters  (properties)

    Diagnostics:
        .drift_events     — list of DriftEvent objects (see inner class)
        .in_relax         — True while adaptation thresholds are relaxed
        .surprise_history — list of (t, surprise) for every update
    """

    # ------------------------------------------------------------------
    # Small struct so each drift event carries everything useful
    # ------------------------------------------------------------------
    class DriftEvent:
        __slots__ = (
            "t",                    # timestamp CUSUM fired
            "direction",            # 'up' | 'down'
            "score",                # abruptness z-score
            "acceleration",         # CUSUM 2nd derivative
            "n_mcs",                # live MC count at fire time
            "n_macros",             # live macro count at fire time
            "nearest_real_drift",   # filled in post-hoc by evaluate_timing()
            "lag_to_real",          # filled in post-hoc (positive = fired after drift)
            "verdict",              # 'true_positive' | 'false_positive' | 'early'
        )

        def __init__(self, t, direction, score, acceleration, n_mcs, n_macros):
            self.t                  = t
            self.direction          = direction
            self.score              = score
            self.acceleration       = acceleration
            self.n_mcs              = n_mcs
            self.n_macros           = n_macros
            self.nearest_real_drift = None
            self.lag_to_real        = None
            self.verdict            = "unverified"

        def __repr__(self):
            return (
                f"DriftEvent(t={self.t}, dir={self.direction}, "
                f"score={self.score:.2f}, accel={self.acceleration:.3f}, "
                f"mcs={self.n_mcs}, macros={self.n_macros}, "
                f"verdict={self.verdict}, lag={self.lag_to_real})"
            )

    # ------------------------------------------------------------------

    def __init__(
        self,
        # ---- TNStream params (pass-through) ----
        W=500, N=5, r_max=0.15, k=4, n_micro=2,
        snn_k=5, alpha=1.6, tk=3, mk=2,
        backend='ball', n_hashes=10, n_tables=3,
        # ---- CUSUM params ----
        cusum_k_lambda=3.5304030498554377,
        cusum_k_slope=2.509685764176742,
        cusum_ema_alpha=0.11986779639677939,
        cusum_delta=0.016777202486114567,
        cusum_confirm=2,
        cusum_min_samples=39,
        # ---- adaptation params ----
        relax_steps=238,
        n_micro_relax=1,
        N_fast=4,
        fast_window_fraction=0.21498970223954925,
        
    ):
        super().__init__(
            W=W, N=N, r_max=r_max, k=k, n_micro=n_micro,
            snn_k=snn_k, alpha=alpha, tk=tk, mk=mk,
            backend=backend, n_hashes=n_hashes, n_tables=n_tables,
        )
        

        self._cusum = AbruptCUSUM(
             k_lambda=cusum_k_lambda,
            k_slope=cusum_k_slope,
            ema_alpha=cusum_ema_alpha,
            delta=cusum_delta,
            confirm_steps=cusum_confirm,
            min_samples=cusum_min_samples,
        )

        self._relax_steps     = relax_steps
        self._n_micro_relax   = n_micro_relax
        self._N_fast          = max(2, min(N_fast, N - 1))
        self._relax_remaining = 0
        self._drift_boundary       = None
        self._fast_window          = None
        self._fast_window_fraction = fast_window_fraction
        self._boundary_cooldown    = 0

        # diagnostics
        self.drift_events:     list = []   # list[DriftEvent]
        self.surprise_history: list = []   # list[(t, float)]

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    @property
    def in_relax(self) -> bool:
        return self._relax_remaining > 0

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self._t += 1

        cutoff = self._t - self.W

        # active overrides during relaxation period
        if self._relax_remaining > 0:
            n_micro_eff = self._n_micro_relax
            N_eff       = self._N_fast
            self._relax_remaining -= 1
        else:
            n_micro_eff = self.n_micro
            N_eff       = self.N

        if self._boundary_cooldown > 0:
            self._boundary_cooldown -= 1
        else:
            self._drift_boundary = None
            self._fast_window    = None

        # standard TNStream pipeline — only the threshold args change
        self._kill_mcs(cutoff)
        self._kill_macros()

        if not self._add_to_mc(self._t, x):
            self._pool.append((self._t, x))
            self._pool = [(t, p) for t, p in self._pool if t > cutoff]

        if len(self._pool) >= N_eff:
            self._define_mc()

        self._add_mc_to_macro()
        self._define_macro_relaxed(n_micro_eff)
        self._update_macros()
        self._kill_macros_relaxed(n_micro_eff)

        # compute and log surprise signal
        surprise = self._surprise(x)
        self.surprise_history.append((self._t, surprise))

        # gate: don't run CUSUM until the model has at least one stable macro
        if not self._macros:
            return

        if self._cusum.update(surprise):
            ev = TNStreamCUSUM.DriftEvent(
                t            = self._t,
                direction    = self._cusum.direction,
                score        = self._cusum.abruptness_score,
                acceleration = self._cusum.acceleration,
                n_mcs        = len(self._mcs),
                n_macros     = len(self._macros),
            )
            self.drift_events.append(ev)
            self._relax_remaining = self._relax_steps
            if (self._boundary_cooldown == 0
                    and self._cusum.abruptness_score >= 2.3):
                self._drift_boundary   = self._t
                self._fast_window      = int(self.W * self._fast_window_fraction)
                self._boundary_cooldown = self._relax_steps
            self._cusum.reset()

    # -------------------------------------------------------------------------
    # Post-hoc timing evaluator
    # -------------------------------------------------------------------------

    def evaluate_timing(self, true_drift_points: list, tolerance: int = None):
        """
        Compare detected drift events against ground-truth drift timestamps
        and classify each as true_positive, false_positive, or early.

        Parameters
        ----------
        true_drift_points : list of int
            The actual drift timestamps from the dataset (ds.drift_points).
        tolerance : int, optional
            How many steps either side of a true drift counts as a match.
            Defaults to W // 4 (one quarter-window).

        Returns
        -------
        dict with keys:
            true_positive, false_positive, early, missed,
            precision, recall, events (the annotated DriftEvent list)
        """
        if tolerance is None:
            tolerance = self.W // 4

        true_drifts  = sorted(true_drift_points)
        matched_true = set()

        for ev in self.drift_events:
            best_dist = None
            best_td   = None
            for td in true_drifts:
                dist = ev.t - td   # positive = fired after drift
                if abs(dist) <= tolerance:
                    if best_dist is None or abs(dist) < abs(best_dist):
                        best_dist = dist
                        best_td   = td

            if best_td is not None:
                ev.nearest_real_drift = best_td
                ev.lag_to_real        = best_dist
                ev.verdict = "early" if best_dist < 0 else "true_positive"
                matched_true.add(best_td)
            else:
                ev.nearest_real_drift = None
                ev.lag_to_real        = None
                ev.verdict            = "false_positive"

        tp     = sum(1 for ev in self.drift_events if ev.verdict == "true_positive")
        fp     = sum(1 for ev in self.drift_events if ev.verdict == "false_positive")
        early  = sum(1 for ev in self.drift_events if ev.verdict == "early")
        missed = len(true_drifts) - len(matched_true)
        total  = len(self.drift_events)

        return {
            "true_positive" : tp,
            "false_positive": fp,
            "early"         : early,
            "missed"        : missed,
            "precision"     : round(tp / total,           3) if total       else float("nan"),
            "recall"        : round(tp / len(true_drifts),3) if true_drifts else float("nan"),
            "events"        : self.drift_events,
        }

    def print_drift_log(self, true_drift_points: list = None, tolerance: int = None):
        """
        Print a human-readable table of all drift detections.

        If true_drift_points is supplied, evaluate_timing() is called first
        so verdicts and lag columns are populated.

        Example output
        --------------
          t=1253  dir=up    score= 4.21  accel=0.183  mcs= 7  macros=2  TRUE_POSITIVE   lag=+3
          t= 312  dir=down  score= 2.10  accel=0.091  mcs= 3  macros=1  FALSE_POSITIVE  lag=n/a

        Usage in a benchmark loop
        -------------------------
            model = TNStreamCUSUM()
            for x in stream:
                model.update(x)
            model.print_drift_log(true_drift_points=ds.drift_points)
        """
        if true_drift_points is not None:
            self.evaluate_timing(true_drift_points, tolerance)

        if not self.drift_events:
            print("  [drift log] no drift events detected")
            return

        header = f"  {'t':>6}  {'dir':<5} {'score':>6}  {'accel':>6}  {'mcs':>4}  {'macros':>6}  {'verdict':<16}  lag"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for ev in self.drift_events:
            lag_str = f"{ev.lag_to_real:+d}" if ev.lag_to_real is not None else "n/a"
            print(
                f"  {ev.t:>6}  {ev.direction:<5} {ev.score:>6.2f}  "
                f"{ev.acceleration:>6.3f}  {ev.n_mcs:>4}  {ev.n_macros:>6}  "
                f"{ev.verdict:<16}  {lag_str}"
            )

        if true_drift_points is not None:
            s = self.evaluate_timing(true_drift_points, tolerance)
            print()
            print(
                f"  TP={s['true_positive']}  FP={s['false_positive']}  "
                f"early={s['early']}  missed={s['missed']}  "
                f"precision={s['precision']}  recall={s['recall']}"
            )

    # -------------------------------------------------------------------------
    # Relaxed-threshold overrides (swap n_micro temporarily, restore after)
    # -------------------------------------------------------------------------

    def _define_macro_relaxed(self, n_micro_eff: int):
        """Run _define_macro with a temporary n_micro override."""
        original     = self.n_micro
        self.n_micro = n_micro_eff
        self._define_macro()
        self.n_micro = original

    def _kill_macros_relaxed(self, n_micro_eff: int):
        """Run _kill_macros with a temporary n_micro override."""
        original     = self.n_micro
        self.n_micro = n_micro_eff
        self._kill_macros()
        self.n_micro = original
    def _kill_mcs(self, cutoff):
        """
        Override TNStream._kill_mcs to apply accelerated expiry to old-regime
        MCs after confirmed drift.

        MCs whose points are majority pre-drift-boundary use a shortened
        effective window (fast_window). As new-regime points accumulate into
        the MC, the pre-drift ratio drops below 0.5 and it automatically
        reverts to the full window. Self-healing — no manual cleanup needed.
        """
        surviving = []
        for mc in self._mcs:
            if not mc.points:
                continue

            effective_cutoff = cutoff  # default: normal full-window cutoff

            if self._drift_boundary is not None and self._fast_window is not None:
                total = len(mc.points)
                pre   = sum(1 for t, _ in mc.points if t < self._drift_boundary)
                if pre / total > 0.5:
                    # majority old-regime — use accelerated cutoff
                    effective_cutoff = self._t - self._fast_window

            if mc.expire(effective_cutoff):
                surviving.append(mc)

        self._mcs = surviving
    def _backdate_old_mcs(self, penalty_fraction: float = 0.5):
        """
        Only ages MCs whose most recent point predates the current moment
        by at least W//4 steps — i.e. clearly old-regime MCs, not freshly
        formed ones that might belong to the new regime.
        """
        penalty = int(self.W * penalty_fraction)
        safe_age = self.W // 4   # MC must be at least this old to be backdated
        for mc in self._mcs:
            if not mc.points:
                continue
            newest = mc.points[-1][0]
            if newest < (self._t - safe_age):   # clearly old, not newly formed
                mc.points = [(t - penalty, x) for t, x in mc.points]
    # -------------------------------------------------------------------------
    # Improved surprise signal
    # -------------------------------------------------------------------------

    def _surprise(self, x) -> float:
        """
        Normalised distance to the nearest MC centre.

        Returns 0.0 when no MCs exist — suppresses cold-start false alarms
        because CUSUM sees a flat zero signal during warmup rather than a
        noisy r_max spike that looks like sustained drift.

        Returns dist / radius otherwise:
            < 1.0  — point landed inside its nearest MC  (unsurprising)
            ~ 1.0  — point is right at the boundary
            > 1.0  — point landed outside                (mildly surprising)
            >> 1.0 — far outside all MCs                 (strong drift signal)
        """
        if not self._mcs:
            return 0.0
        centers = np.array([mc.center for mc in self._mcs])
        dists   = np.linalg.norm(centers - x, axis=1)
        idx     = int(np.argmin(dists))
        return float(dists[idx] / max(self._mcs[idx].radius, 1e-6))