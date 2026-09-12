"""Couriers that move between regions -- the fidelity gap this project named.

WHAT THIS PROJECT SAID ABOUT ITSELF
-----------------------------------
"Couriers still do not reposition, accept or decline in THIS simulator -- SE-3
models all three, and the two are not joined. That remains the biggest fidelity
gap."

"Regions are independent, so geo spillover cannot be represented at all. In a
real metro adjacent regions share couriers, and that is exactly the failure a geo
design is supposed to be protected against."

Those are the same gap stated twice. A geo experiment's whole claim is that
randomising at the region level removes interference; if couriers cross region
boundaries the claim is false, and a simulator whose couriers cannot move can
never show it failing. This module makes them move, using SE-3's policy rather
than a second implementation of one.

WHY IMPORT SE-3 RATHER THAN WRITE A REPOSITIONING RULE
------------------------------------------------------
A repositioning rule written here to demonstrate spillover would be a rule
written to demonstrate spillover. SE-3's was built and tuned for a different
question -- how much fill rate repositioning buys -- and measured against an
oracle placement, with its own finding that a broadcast recommendation is already
optimal in a scarce market. Borrowing it means the interference measured here is
a consequence of a policy somebody else's numbers justified.

Loaded BY FILE PATH: both projects have a top-level package called `src`, so a
plain sys.path insert resolves `src.control` against whichever `src` was imported
first. DATA-2 and SE-1 hit the same thing from their side of the ML-1 join.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SE3 = os.path.join(HERE, os.pardir, "delivery-dispatch-tracking")

_CACHE = {}


def se3_available() -> bool:
    return os.path.exists(os.path.join(SE3, "src", "control.py"))


def se3_control():
    if "ctl" in _CACHE:
        return _CACHE["ctl"]
    path = os.path.join(SE3, "src", "control.py")
    if not os.path.exists(path):
        raise ImportError("delivery-dispatch-tracking/src/control.py not found")
    spec = importlib.util.spec_from_file_location("se3_control_d3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE["ctl"] = mod
    return mod


class CourierPool:
    """Which region each courier sits in, and who is free to move.

    The Marketplace tracks a COUNT of couriers per region and a list of busy-until
    stamps. Repositioning only ever moves idle couriers, so the busy list never
    has to follow anybody: this pool decides where the idle ones go and hands back
    a new per-region capacity vector.

    `travel_minutes` is what stops every courier converging on the best region.
    Regions are laid out on a line, which is a weaker geometry than SE-3's grid
    and is stated rather than hidden -- it makes adjacent regions cheap to reach
    and distant ones expensive, which is the property the spillover result needs.
    """

    def __init__(self, n_regions: int, per_region: int, spacing_minutes: float = 6.0,
                 seed: int = 0):
        self.R = n_regions
        self.region = np.repeat(np.arange(n_regions), int(per_region))
        self.rng = np.random.default_rng(seed)
        self.travel = (np.abs(np.arange(n_regions)[:, None]
                              - np.arange(n_regions)[None, :]) * spacing_minutes)
        self.moves = 0

    def counts(self) -> np.ndarray:
        return np.bincount(self.region, minlength=self.R).astype(float)

    def _idle_mask(self, busy: np.ndarray) -> np.ndarray:
        """Mark (count - busy) couriers idle in each region.

        Which particular couriers are idle does not matter -- they are
        interchangeable -- but the COUNT does, and taking it from the
        marketplace's own busy vector keeps the two in step. Marking more
        couriers idle than exist would let a busy courier reposition and quietly
        create capacity.
        """
        counts = self.counts()
        idle = np.zeros(len(self.region), dtype=bool)
        for r in range(self.R):
            free = int(max(counts[r] - np.ceil(busy[r]), 0))
            if free <= 0:
                continue
            where = np.flatnonzero(self.region == r)
            idle[where[:free]] = True
        return idle

    def step(self, busy: np.ndarray, demand_rate: np.ndarray,
             surge: np.ndarray | None = None, targeted: bool = False,
             compliance: float = 0.55, shrink: float = 1.0) -> np.ndarray:
        """Move idle couriers once. Returns the new per-region capacity.

        `shrink` regularises the demand estimate toward its own mean before the
        policy sees it, and it is load-bearing rather than cosmetic.

        SE-3's policy decides on a RATIO of expected earnings, which is the right
        shape -- a courier compares here against there. But a ratio of two small
        counts is noise with a decimal point. The first version of this passed a
        raw trailing estimate straight through, and at 00:30 with the estimate
        still warming up it read [0.5, 0.0, 0.0, 1.1, 0.0, 1.3] orders per
        half-hour. The 1.1 against the 0.5 clears the policy's 1.25 threshold on
        a difference of SIX TENTHS OF AN ORDER, and thirty couriers moved on it.
        Over a day of half-hourly decisions that ratchets: 87 of 168 couriers
        ended in one region, which is a runaway rather than an equilibrium.

        Shrinking toward the pooled mean compresses the ratios, which breaks the
        ratchet: with it the most crowded region holds 12% of the fleet after
        three days, without it 38%.

        BE PRECISE ABOUT WHAT IT DOES NOT DO. It does not neutralise a single
        degenerate estimate -- that vector shrunk by its own mean still spans a
        3.69 ratio and still moves couriers. What it defuses is the REPETITION:
        once demand is non-trivial the prior is small relative to the signal, and
        the compounding that turned small noisy differences into a corner
        solution stops. A test pins both halves, because the first version of
        this docstring claimed the stronger thing and a test disagreed.

        It is the same empirical-Bayes move ML-1 makes for per-item elasticity,
        for the same reason.

        The fix is applied to the ESTIMATE rather than to SE-3's policy: the
        policy is not wrong to use a ratio, and editing another project's tuned
        decision rule to fix this project's input would put the repair in the
        wrong place.
        """
        ctl = se3_control()
        idle = self._idle_mask(busy)
        if not idle.any():
            return self.counts()
        counts = self.counts()
        demand_rate = np.asarray(demand_rate, float)
        if shrink > 0:
            demand_rate = demand_rate + shrink * float(demand_rate.mean())
        if surge is None:
            surge = np.ones(self.R)
        fn = ctl.reposition_targeted if targeted else ctl.reposition
        before = self.region.copy()
        self.region = fn(self.region, idle, demand_rate, counts, self.travel,
                         surge, self.rng, compliance=compliance)
        self.moves += int((self.region != before).sum())
        return self.counts()


def crossing_rate(treated_regions: np.ndarray, start: np.ndarray,
                  end: np.ndarray) -> float:
    """Share of couriers that ended in a region of the opposite arm.

    The number a geo design's assumption rests on. If it is zero the design's
    claim holds; if it is not, the control arm has been damaged by the treatment
    and the estimate is inflated by more than the treatment did.
    """
    moved_arm = treated_regions[end] != treated_regions[start]
    return float(moved_arm.mean())
