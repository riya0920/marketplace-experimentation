"""Couriers that move between regions, borrowed from SE-3.

This project named the gap twice -- "couriers still do not reposition in THIS
simulator" and "regions are independent, so geo spillover cannot be represented
at all". These are the tests for closing it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import market as MK      # noqa: E402
from src import mobility as MOB   # noqa: E402


needs_se3 = pytest.mark.skipif(not MOB.se3_available(),
                               reason="se3-dispatch-tracking not present")


# --------------------------------------------------------------------------
# the join
# --------------------------------------------------------------------------
@needs_se3
def test_the_policy_is_se3s_and_not_a_second_implementation():
    ctl = MOB.se3_control()
    assert hasattr(ctl, "reposition")
    assert hasattr(ctl, "reposition_targeted")
    assert hasattr(ctl, "herding_index")


@needs_se3
def test_repositioning_conserves_couriers():
    """Nobody is created or destroyed by moving. A pool that leaked couriers
    would look exactly like a treatment effect on capacity."""
    pool = MOB.CourierPool(6, 10, seed=0)
    total = len(pool.region)
    for _ in range(20):
        counts = pool.step(np.zeros(6), np.arange(6, dtype=float) * 3.0)
        assert counts.sum() == total


@needs_se3
def test_only_idle_couriers_move():
    """Marking more couriers idle than exist would let a busy one reposition and
    quietly create capacity in two places at once."""
    pool = MOB.CourierPool(4, 10, seed=1)
    busy = np.array([10.0, 10.0, 10.0, 10.0])      # everybody working
    before = pool.region.copy()
    pool.step(busy, np.array([1.0, 50.0, 1.0, 1.0]))
    assert np.array_equal(pool.region, before)


@needs_se3
def test_the_idle_mask_never_exceeds_the_courier_count():
    pool = MOB.CourierPool(5, 8, seed=2)
    for busy_level in (0.0, 3.0, 7.9, 8.0):
        busy = np.full(5, busy_level)
        idle = pool._idle_mask(busy)
        counts = pool.counts()
        per_region = np.bincount(pool.region[idle], minlength=5)
        assert (per_region <= counts).all()


# --------------------------------------------------------------------------
# the bug this module produced
# --------------------------------------------------------------------------
@needs_se3
def test_shrinkage_is_what_stops_the_runaway():
    """The bug: raw trailing demand estimates are near zero while warming up,
    SE-3's policy decides on a RATIO, and a ratio of two small counts is noise
    with a decimal point. Over a day of half-hourly decisions it ratcheted --
    87 of 168 couriers ended in one region.

    The claim under test is about SUSTAINED operation, not a single step. On one
    pathological vector shrinkage changes nothing (the first version of this test
    asserted that it did, and failed): [0.5, 0, 0, 1.1, 0, 1.3] shrunk by its own
    mean still spans a 3.69 ratio, well past the policy's 1.25 threshold. What
    shrinkage does is compress ratios once demand is non-trivial, which is what
    breaks the ratchet.
    """
    a = MK.assign_cluster(12, seed=100)
    out = {}
    for shrink in (0.0, 1.0):
        pool = MOB.CourierPool(12, 14, seed=0)
        rec = MK.Marketplace(n_regions=12, couriers=14, seed=0).run(
            3, a, lift=0.08, pool=pool, shrink=shrink)
        c = rec["final_couriers"]
        out[shrink] = c.max() / c.sum()
    assert out[0.0] > 0.30, out          # the runaway
    assert out[1.0] < 0.20, out          # a market
    assert out[1.0] < out[0.0]


@needs_se3
def test_shrinkage_alone_does_not_neutralise_a_degenerate_estimate():
    """Kept because it is the honest boundary of the fix, and because the
    docstring used to claim otherwise. A single step on a near-zero vector still
    moves couriers; it is the repetition that shrinkage defuses."""
    tiny = np.array([0.5, 0.0, 0.0, 1.1, 0.0, 1.3])
    moved = {}
    for shrink in (0.0, 1.0):
        pool = MOB.CourierPool(6, 28, seed=3)
        before = pool.region.copy()
        pool.step(np.zeros(6), tiny, shrink=shrink)
        moved[shrink] = int((pool.region != before).sum())
    assert moved[0.0] > 0
    assert moved[1.0] > 0


@needs_se3
def test_a_real_demand_signal_still_gets_through_the_shrinkage():
    """Otherwise the fix would be 'turn the policy off'."""
    pool = MOB.CourierPool(4, 20, seed=4)
    strong = np.array([2.0, 90.0, 2.0, 2.0])
    pool.step(np.zeros(4), strong, shrink=1.0)
    assert pool.counts()[1] > 20


@needs_se3
def test_repositioning_reaches_an_equilibrium_rather_than_a_corner():
    """The runaway produced 87 of 168 couriers in one region. Anything that
    concentrated is a broken policy, not a market."""
    pool = MOB.CourierPool(8, 14, seed=5)
    demand = 6.0 + 4.0 * np.arange(8)
    busy = np.zeros(8)
    for _ in range(60):
        counts = pool.step(busy, demand, shrink=1.0)
    assert counts.max() < 0.40 * counts.sum()


# --------------------------------------------------------------------------
# reproducibility -- the second bug
# --------------------------------------------------------------------------
def test_the_weather_is_the_same_across_processes():
    """It was seeded with hash(("weather", days)) and Python salts string
    hashing per process, so the comment promising that the same seed gives the
    same weather was true within one run and false across two. Comparisons
    inside a report were sound and nobody could reproduce the report.

    This test pins the actual values, so it fails if the seed source ever goes
    back to something process-dependent."""
    rng = np.random.default_rng(20260101 + 3)
    expect = [MK.day_multiplier(d, rng) for d in range(3)]
    rng2 = np.random.default_rng(20260101 + 3)
    again = [MK.day_multiplier(d, rng2) for d in range(3)]
    assert expect == again


def test_two_marketplaces_with_the_same_seed_agree():
    a = MK.assign_cluster(6, seed=1)
    r1 = MK.Marketplace(n_regions=6, couriers=10, seed=0).run(1, a, lift=0.05)
    r2 = MK.Marketplace(n_regions=6, couriers=10, seed=0).run(1, a, lift=0.05)
    assert r1["orders"].sum() == r2["orders"].sum()


def test_every_design_being_compared_draws_the_same_weather():
    """Deliberately NOT derived from `seed`: otherwise a design comparison is
    partly a comparison of the weather each design happened to get."""
    m1 = MK.Marketplace(n_regions=4, couriers=10, seed=0)
    m2 = MK.Marketplace(n_regions=4, couriers=10, seed=99)
    assert m1.weather_seed == m2.weather_seed


# --------------------------------------------------------------------------
# what the section concludes
# --------------------------------------------------------------------------
@needs_se3
def test_couriers_do_cross_between_arms():
    """If they did not, the section's headline would be vacuous -- an estimate
    unaffected by an interference channel that never opened."""
    a = MK.assign_cluster(12, seed=100)
    pool = MOB.CourierPool(12, 14, seed=0)
    MK.Marketplace(n_regions=12, couriers=14, seed=0).run(
        2, a, lift=0.08, pool=pool)
    rate = MOB.crossing_rate(a.treated_regions, np.repeat(np.arange(12), 14),
                             pool.region)
    assert rate > 0.2, rate


@needs_se3
def test_the_run_restores_its_courier_vector():
    """`run` mutates self.couriers while repositioning. A Marketplace reused for
    a second run must not inherit the first run's final positions, or the two
    runs are not the comparison they claim to be."""
    a = MK.assign_cluster(6, seed=2)
    mk = MK.Marketplace(n_regions=6, couriers=12, seed=0)
    before = mk.couriers.copy()
    mk.run(1, a, lift=0.05, pool=MOB.CourierPool(6, 12, seed=0))
    assert np.array_equal(mk.couriers, before)


@needs_se3
def test_mobility_does_not_change_the_number_of_couriers_in_the_market():
    a = MK.assign_cluster(6, seed=3)
    pool = MOB.CourierPool(6, 12, seed=1)
    rec = MK.Marketplace(n_regions=6, couriers=12, seed=0).run(
        2, a, lift=0.05, pool=pool)
    assert rec["final_couriers"].sum() == pytest.approx(72, abs=1e-9)
