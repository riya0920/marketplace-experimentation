"""Tests for the second tranche: adjusted estimators, pair matching, MDE."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import designs as DS  # noqa: E402
from src import estimators as E  # noqa: E402
from src import market as M  # noqa: E402


def _switchback_rec(block=120, days=2, seed=1, regions=6, couriers=12, lift=0.10):
    mk = M.Marketplace(n_regions=regions, couriers=couriers, seed=seed)
    return mk.run(days, M.assign_switchback(block, seed=seed), lift)


# --------------------------------------------------------------------------
# adjusted switchback estimators
# --------------------------------------------------------------------------
def test_adjusted_estimator_returns_a_finite_estimate():
    rec = _switchback_rec()
    out = DS.switchback_adjusted(rec, 120, burn_in_minutes=20, seed=1)
    assert np.isfinite(out["estimate"])
    assert out["n_buckets_used"] > 0


def test_paired_estimator_uses_only_arm_splitting_pairs():
    rec = _switchback_rec()
    out = DS.switchback_paired(rec, 120, burn_in_minutes=20)
    assert np.isfinite(out["estimate"])
    # roughly half of adjacent block pairs split arms under a fair coin
    n_blocks = E.switchback(rec, 120, 20)["n_blocks"]
    assert 0 < out["n_pairs"] < n_blocks


def test_adjustment_reduces_variance_across_replications():
    """The claim the section makes. Adjustment attacks VARIANCE, not bias, so
    the test is on the spread across independent runs."""
    unadj, adj = [], []
    for rep in range(8):
        rec = _switchback_rec(block=120, days=2, seed=200 + rep)
        a = E.switchback(rec, 120, 20, seed=rep)
        b = DS.switchback_adjusted(rec, 120, 20, seed=rep)
        if np.isfinite(a["estimate"]) and np.isfinite(b["estimate"]):
            unadj.append(a["estimate"])
            adj.append(b["estimate"])
    assert len(adj) >= 5
    assert np.std(adj, ddof=1) < np.std(unadj, ddof=1)


def test_burn_in_does_not_change_the_block_count():
    """Burn-in drops MINUTES inside blocks, not blocks -- if it dropped blocks
    the comparison across burn-in values would be confounded."""
    rec = _switchback_rec()
    a = DS.switchback_adjusted(rec, 120, burn_in_minutes=0, seed=1)
    b = DS.switchback_adjusted(rec, 120, burn_in_minutes=20, seed=1)
    assert a["n_blocks"] == b["n_blocks"]


def test_adjusted_returns_nan_when_no_bucket_sees_both_arms():
    """Degenerate input must produce NaN, not a number computed from nothing."""
    mk = M.Marketplace(n_regions=4, couriers=12, seed=3)
    rec = mk.run(1, M.assign_all(True), 0.10)      # only ever treated
    out = DS.switchback_adjusted(rec, 120, seed=1)
    assert np.isnan(out["estimate"])


# --------------------------------------------------------------------------
# pair-matched assignment
# --------------------------------------------------------------------------
def test_pair_matching_balances_the_pre_period_covariate():
    rng = np.random.default_rng(0)
    pre = rng.lognormal(3.0, 0.8, 24)
    ind_imb, pm_imb = [], []
    for seed in range(40):
        ind = rng.random(24) < 0.5
        if ind.all() or (~ind).all():
            continue
        ind_imb.append(DS.imbalance(pre, ind))
        pm_imb.append(DS.imbalance(pre, DS.pair_matched_assignment(pre, seed=seed)))
    assert np.mean(pm_imb) < np.mean(ind_imb)


def test_pair_matching_splits_the_arms_roughly_evenly():
    pre = np.arange(1, 25, dtype=float)
    t = DS.pair_matched_assignment(pre, seed=1)
    assert abs(t.sum() - (~t).sum()) <= 1


def test_pair_matching_handles_an_odd_number_of_regions():
    pre = np.arange(1, 14, dtype=float)
    t = DS.pair_matched_assignment(pre, seed=2)
    assert len(t) == 13
    assert t.any() and not t.all()


def test_pair_matching_puts_one_of_each_pair_in_each_arm():
    """Sorted neighbours are the pairs, so consecutive ranks must not both land
    in the same arm -- that is the entire mechanism."""
    pre = np.arange(1, 21, dtype=float)
    t = DS.pair_matched_assignment(pre, seed=3)
    order = np.argsort(pre)
    for i in range(0, 20, 2):
        a, b = order[i], order[i + 1]
        assert t[a] != t[b]


def test_imbalance_is_zero_for_identical_arms():
    pre = np.array([1.0, 1.0, 2.0, 2.0])
    t = np.array([True, False, True, False])
    assert DS.imbalance(pre, t) == pytest.approx(0.0)


def test_imbalance_grows_with_separation():
    pre = np.array([1.0, 2.0, 10.0, 11.0])
    balanced = np.array([True, False, True, False])
    skewed = np.array([True, True, False, False])
    assert DS.imbalance(pre, skewed) > DS.imbalance(pre, balanced)


# --------------------------------------------------------------------------
# minimum detectable effect
# --------------------------------------------------------------------------
def test_mde_scales_with_standard_error():
    assert DS.mde(0.04) == pytest.approx(2 * DS.mde(0.02))


def test_mde_is_larger_with_few_degrees_of_freedom():
    """Using the normal at 10 df is how a 12-cluster test gets reported as
    adequately powered."""
    assert DS.mde(0.02, dof=10) > DS.mde(0.02, dof=1000)


def test_mde_matches_the_textbook_constant_at_large_dof():
    # (1.96 + 0.8416) * se
    assert DS.mde(0.01, dof=10_000) == pytest.approx(0.028016, abs=1e-4)


def test_mde_answers_the_powered_question():
    """The point of the number: it converts 'we have 12 geos' into a verdict."""
    true_effect = 0.033
    assert DS.mde(0.021, dof=10) > true_effect      # 12 clusters: cannot detect
    assert DS.mde(0.004, dof=200) < true_effect     # a much larger design: can
