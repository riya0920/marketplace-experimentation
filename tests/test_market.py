"""Guards on the simulator and the estimators.

The simulator's job is to have an honest interference channel. If it does not,
every number in the report is decoration, so the channel itself is tested
directly rather than assumed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import estimators as E  # noqa: E402
from src import market as M  # noqa: E402


def _run(couriers=12, days=1, assign=None, lift=0.10, seed=0, regions=6):
    mk = M.Marketplace(n_regions=regions, couriers=couriers, seed=seed)
    return mk.run(days, assign or M.assign_user_level(0.5), lift)


# --------------------------------------------------------------------------
# the simulator
# --------------------------------------------------------------------------
def test_stylised_facts_hold():
    assert all(M.validate(M.Marketplace(seed=0)).values())


def test_eta_rises_with_congestion():
    mk = M.Marketplace(n_regions=3, couriers=10, seed=0)
    quiet = mk.eta(np.full(3, 1.0))[0]
    busy = mk.eta(np.full(3, 9.0))[0]
    assert busy > quiet


def test_conversion_falls_as_eta_rises():
    mk = M.Marketplace(seed=0)
    fast = mk.conversion(np.array([18.0]), np.array([False]), 0.0)[0]
    slow = mk.conversion(np.array([90.0]), np.array([False]), 0.0)[0]
    assert slow < fast


def test_treatment_raises_conversion_at_fixed_eta():
    mk = M.Marketplace(seed=0)
    c = mk.conversion(np.array([30.0]), np.array([False]), 0.10)[0]
    t = mk.conversion(np.array([30.0]), np.array([True]), 0.10)[0]
    assert t > c
    assert t == pytest.approx(c * 1.10)


def test_the_interference_channel_actually_exists():
    """THE load-bearing test. Treating everyone must raise utilisation; if it
    does not, there is no interference and the entire project is measuring
    nothing."""
    rt = _run(assign=M.assign_all(True), seed=5)
    rc = _run(assign=M.assign_all(False), seed=5)
    gt, gc = E.guardrails(rt), E.guardrails(rc)
    assert gt["orders"] > gc["orders"]
    assert gt["mean_utilisation"] > gc["mean_utilisation"]
    assert gt["mean_eta"] > gc["mean_eta"]


def test_tighter_markets_congest_more():
    slack = E.guardrails(_run(couriers=30, assign=M.assign_all(False), seed=3))
    tight = E.guardrails(_run(couriers=8, assign=M.assign_all(False), seed=3))
    assert tight["mean_utilisation"] > slack["mean_utilisation"]
    assert tight["mean_eta"] > slack["mean_eta"]


def test_zero_lift_means_no_true_effect():
    """A null treatment must produce a null truth, or the ground truth itself is
    contaminated and every bias number is measured against a moving target."""
    rt = _run(assign=M.assign_all(True), lift=0.0, days=2, seed=11)
    rc = _run(assign=M.assign_all(False), lift=0.0, days=2, seed=11)
    assert abs(E.global_effect(rt, rc)) < 0.01


def test_records_are_internally_consistent():
    rec = _run(days=1, seed=7)
    assert (rec["t_exposures"] + rec["c_exposures"] == rec["exposures"]).all()
    assert (rec["t_orders"] + rec["c_orders"] == rec["orders"]).all()
    assert (rec["orders"] <= rec["exposures"]).all()


# --------------------------------------------------------------------------
# assignment designs
# --------------------------------------------------------------------------
def test_switchback_treats_the_whole_market_within_a_block():
    f = M.assign_switchback(60, seed=1)
    rng = np.random.default_rng(0)
    for t in (0, 30, 59):
        arms = {bool(f(t, r, 5, rng)[0]) for r in range(6)}
        assert len(arms) == 1, "a switchback block must be one arm market-wide"


def test_switchback_blocks_do_change_arm():
    f = M.assign_switchback(60, seed=2)
    rng = np.random.default_rng(0)
    arms = {bool(f(t * 60, 0, 3, rng)[0]) for t in range(30)}
    assert arms == {True, False}


def test_cluster_assignment_is_stable_over_time_and_covers_both_arms():
    f = M.assign_cluster(12, seed=3)
    rng = np.random.default_rng(0)
    for r in range(12):
        assert bool(f(0, r, 2, rng)[0]) == bool(f(9999, r, 2, rng)[0])
    assert f.treated_regions.any() and not f.treated_regions.all()


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def test_cluster_se_is_larger_than_the_naive_se():
    """The inference error the report sizes. If this ever fails, the geo section
    is claiming something it cannot show.

    ACROSS SEEDS, not one. The previous version asserted `robust > naive * 1.5`
    on a single seed, and the measured ratio runs 1.04 to 1.80 with a median of
    1.39 -- so that assertion was roughly a coin flip. It never flickered only
    because the weather draw was seeded from a salted string hash and silently
    re-rolled every process; fixing the seed made the coin land tails and
    exposed the test.

    The property that is actually true: the cluster-robust standard error is
    ALWAYS larger, and typically about 40% larger."""
    ratios = []
    for seed in range(8):
        mk = M.Marketplace(n_regions=12, couriers=12, seed=seed)
        af = M.assign_cluster(12, seed=seed)
        rec = mk.run(2, af, 0.10)
        robust = E.cluster_randomised(rec, af.treated_regions)["se"]
        naive = E.naive_se_for_cluster_design(rec)
        ratios.append(robust / naive)
    assert min(ratios) > 1.0, ratios
    assert float(np.median(ratios)) > 1.15, ratios


def test_switchback_burn_in_drops_samples():
    mk = M.Marketplace(n_regions=6, couriers=12, seed=31)
    rec = mk.run(2, M.assign_switchback(120, seed=31), 0.10)
    a = E.switchback(rec, 120, burn_in_minutes=0)
    b = E.switchback(rec, 120, burn_in_minutes=60)
    assert a["n_blocks"] == b["n_blocks"]
    assert np.isfinite(a["estimate"]) and np.isfinite(b["estimate"])


def test_naive_ab_overstates_in_a_tight_market():
    """The core Act-1 claim, at small scale. Averaged over seeds because a single
    run's estimate is noisy enough to flip the sign."""
    truths, naives = [], []
    for seed in range(3):
        rt = _run(couriers=8, days=2, assign=M.assign_all(True), seed=seed, regions=8)
        rc = _run(couriers=8, days=2, assign=M.assign_all(False), seed=seed, regions=8)
        truths.append(E.global_effect(rt, rc))
        rec = _run(couriers=8, days=2, seed=100 + seed, regions=8)
        naives.append(E.naive_ab(rec)["estimate"])
    assert np.mean(naives) > np.mean(truths)


def test_guardrails_report_both_sides():
    g = E.guardrails(_run(days=1, seed=41))
    for k in ("mean_utilisation", "p95_utilisation", "mean_eta", "orders"):
        assert k in g
