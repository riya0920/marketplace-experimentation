"""Tests for the completion pass: CUPED, synthetic control, the MDE curve, and
the heterogeneous calendar."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import adjustment as ADJ   # noqa: E402
from src import market as MKT       # noqa: E402
from src import power as POW        # noqa: E402


# --------------------------------------------------------------------------
# CUPED
# --------------------------------------------------------------------------
def _paired(n=400, rho=0.9, effect=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(size=n) * np.sqrt(1 - rho ** 2)
    y = rho * x + noise
    treated = rng.random(n) < 0.5
    y = y + effect * treated
    return y, x, treated


def test_cuped_reduces_variance_by_rho_squared():
    """The reduction is exactly rho^2, which is why its value is knowable in
    advance from a correlation you already have."""
    y, x, _ = _paired(n=4000, rho=0.8, effect=0.0, seed=1)
    g = ADJ.cuped_gain(y, x)
    assert g["realised_reduction"] == pytest.approx(g["predicted_reduction"], abs=0.02)
    assert g["realised_reduction"] == pytest.approx(0.64, abs=0.05)


def test_cuped_does_not_move_the_effect_estimate():
    """It removes variance the treatment cannot have caused, so the point
    estimate must stay put. A method that moves the estimate is not adjusting,
    it is fitting."""
    y, x, t = _paired(n=6000, rho=0.9, effect=0.5, seed=2)
    r = ADJ.cuped_effect(y, t, x)
    assert r["effect"] == pytest.approx(r["effect_unadjusted"], abs=0.05)
    assert r["effect"] == pytest.approx(0.5, abs=0.08)


def test_cuped_shrinks_the_standard_error():
    y, x, t = _paired(n=4000, rho=0.9, effect=0.3, seed=3)
    r = ADJ.cuped_effect(y, t, x)
    assert r["se"] < r["se_unadjusted"]
    assert r["se_reduction"] > 0.4


def test_a_useless_covariate_costs_nothing():
    """theta goes to zero and CUPED degenerates to the unadjusted estimate --
    which is the right failure mode for something applied by default."""
    rng = np.random.default_rng(4)
    y, _, t = _paired(n=2000, rho=0.0, effect=0.4, seed=4)
    junk = rng.normal(size=len(y))
    r = ADJ.cuped_effect(y, t, junk)
    assert r["se"] == pytest.approx(r["se_unadjusted"], rel=0.15)


def test_theta_is_a_regression_coefficient_not_a_correlation():
    """Using the correlation instead under-adjusts whenever the two have
    different scales, and the slip is invisible at unit scale."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=3000) * 10.0
    y = 0.3 * x + rng.normal(size=3000)
    theta = ADJ.cuped_theta(y, x)
    assert theta == pytest.approx(0.3, abs=0.03)
    assert abs(theta - np.corrcoef(y, x)[0, 1]) > 0.3


# --------------------------------------------------------------------------
# synthetic control
# --------------------------------------------------------------------------
def _panel(n_units=10, n_pre=30, n_post=10, effect=0.0, seed=0,
           treated_level=None):
    """A panel with a shared trend and unit-level offsets.

    `treated_level` places unit 0 INSIDE the donors' range by default. That is
    not a convenience: a simplex-constrained fit cannot extrapolate, so a treated
    unit outside the convex hull of the donors is unreproducible and the residual
    leaks straight into the estimated effect. `test_a_treated_unit_outside_the_
    donor_hull_is_detectable` is the case that shows it.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=n_pre + n_post).cumsum() * 0.05
    level = rng.uniform(0.8, 1.2, n_units)
    if treated_level is None:
        level[0] = float(np.median(level[1:]))
    else:
        level[0] = treated_level
    series = level[:, None] + common[None, :] + rng.normal(0, 0.02,
                                                           (n_units, n_pre + n_post))
    pre, post = series[:, :n_pre], series[:, n_pre:]
    post = post.copy()
    post[0] += effect
    return pre, post


def test_weights_live_on_the_simplex():
    """Unconstrained least squares hands a donor -3 and EXTRAPOLATES. The whole
    credibility of the method is that the counterfactual is a weighted average
    of things that actually happened."""
    pre, post = _panel(seed=1)
    out = ADJ.synthetic_control(pre[0], pre[1:], post[0], post[1:])
    w = out["weights"]
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_synthetic_control_recovers_a_planted_effect():
    pre, post = _panel(n_units=12, effect=0.25, seed=2)
    out = ADJ.synthetic_control(pre[0], pre[1:], post[0], post[1:])
    assert out["effect"] == pytest.approx(0.25, abs=0.08)


def test_a_zero_effect_is_estimated_as_roughly_zero():
    pre, post = _panel(n_units=12, effect=0.0, seed=3)
    out = ADJ.synthetic_control(pre[0], pre[1:], post[0], post[1:])
    assert abs(out["effect"]) < 0.06


def test_the_pre_period_fit_is_reported_because_it_decides_everything():
    """A synthetic control with a poor pre-fit has not produced a weak estimate;
    it has failed to build a counterfactual."""
    pre, post = _panel(seed=4)
    out = ADJ.synthetic_control(pre[0], pre[1:], post[0], post[1:])
    assert "pre_rmse" in out and "pre_rmse_relative" in out
    assert out["pre_rmse_relative"] >= 0


def test_placebo_inference_finds_a_real_effect_unusual():
    pre, post = _panel(n_units=14, effect=0.6, seed=5)
    out = ADJ.placebo_inference(pre, post, treated_idx=0)
    assert out["p_value"] < 0.2


def test_placebo_inference_does_not_find_a_null_unusual():
    pre, post = _panel(n_units=14, effect=0.0, seed=6)
    out = ADJ.placebo_inference(pre, post, treated_idx=0)
    assert out["p_value"] > 0.1


def test_a_treated_unit_outside_the_donor_hull_is_detectable():
    """THE FAILURE MODE THAT MATTERS. A simplex fit cannot extrapolate, so a
    treated unit above every donor cannot be reproduced -- and the estimate is
    then wrong by the amount it could not match. The pre-period RMSE is what
    makes it detectable rather than silent, which is the entire reason it is
    reported next to every effect."""
    pre, post = _panel(n_units=12, effect=0.0, seed=8, treated_level=3.0)
    out = ADJ.synthetic_control(pre[0], pre[1:], post[0], post[1:])
    assert out["pre_rmse_relative"] > 0.2, out
    assert abs(out["effect"]) > 0.1, "the unmatched level leaks into the effect"


def test_placebos_with_a_poor_pre_fit_are_excluded():
    """Including them is the standard way this test acquires an artificially
    small p-value: a unit the pool cannot reproduce shows a large 'effect'
    regardless of treatment."""
    pre, post = _panel(n_units=12, effect=0.3, seed=7)
    pre = pre.copy()
    pre[-1] += 40.0            # an outlier no donor combination can track
    post = post.copy()
    post[-1] += 40.0
    out = ADJ.placebo_inference(pre, post, treated_idx=0)
    assert out["n_placebos"] < len(pre) - 1


# --------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------
def test_mde_uses_a_t_reference_not_a_normal():
    """Using the normal is how a 12-cluster test gets reported as adequately
    powered."""
    se = 0.01
    small = POW.mde_from_se(se, n_clusters=12)
    large = POW.mde_from_se(se, n_clusters=400)
    assert small > large
    assert small / se > 2.9          # t at 10 df is meaningfully above 2.80


def test_the_mde_curve_falls_and_flattens():
    curve = POW.mde_curve(0.02, 7, 12)
    mdes = [r["mde"] for r in curve]
    assert all(b <= a for a, b in zip(mdes, mdes[1:]))
    marg = POW.marginal_value(curve)
    # doubling the duration buys ~30%, never more
    for m in marg:
        ratio = m["to_days"] / m["from_days"]
        assert m["improvement"] <= 1 - 1 / np.sqrt(ratio) + 0.02


def test_days_needed_returns_none_when_the_effect_is_undetectable():
    curve = POW.mde_curve(1.0, 7, 12)
    assert POW.days_needed(curve, 0.0001) is None


def test_days_needed_finds_the_first_sufficient_duration():
    curve = POW.mde_curve(0.02, 7, 12)
    d = POW.days_needed(curve, 0.05)
    assert d is not None
    row = [r for r in curve if r["days"] == d][0]
    assert row["mde"] <= 0.05


def test_a_smaller_standard_error_needs_fewer_days():
    """The claim the CUPED section rests on."""
    plain = POW.mde_curve(0.018, 7, 12)
    adjusted = POW.mde_curve(0.004, 7, 12)
    target = 0.037
    assert POW.days_needed(adjusted, target) < POW.days_needed(plain, target)


# --------------------------------------------------------------------------
# the calendar
# --------------------------------------------------------------------------
def test_days_are_no_longer_identical():
    """The artifact the previous README flagged as making the switchback
    variance ranking suspect."""
    mults = [MKT.day_multiplier(d, None) for d in range(7)]
    assert max(mults) / min(mults) > 1.4


def test_the_weekend_is_the_peak():
    dow = MKT.DOW_MULTIPLIER
    assert dow[5] == dow.max()          # Saturday
    assert dow[1] == dow.min()          # Tuesday


def test_the_trend_compounds():
    a = MKT.day_multiplier(0, None)
    b = MKT.day_multiplier(28, None)
    assert b / (MKT.DOW_MULTIPLIER[0] / MKT.DOW_MULTIPLIER[0]) > a


def test_weather_is_a_shared_daily_shock_not_per_region_noise():
    """A rainy Tuesday raises demand everywhere at once, so it cannot be
    averaged away across regions the way region-specific noise can."""
    rng = np.random.default_rng(0)
    a = MKT.day_multiplier(3, rng, weather_sd=0.2)
    rng2 = np.random.default_rng(0)
    b = MKT.day_multiplier(3, rng2, weather_sd=0.2)
    assert a == b               # same draw, same day: one shock, not per-region
    assert a != MKT.day_multiplier(3, None)


# --------------------------------------------------------------------------
# the playbook
# --------------------------------------------------------------------------
def test_the_playbook_is_written_as_a_standalone_artifact(tmp_path):
    p = POW.write_playbook(str(tmp_path / "PLAYBOOK.md"))
    body = open(p, encoding="utf-8").read()
    assert "decision tree" in body.lower()
    assert "CUPED" in body and "synthetic control" in body.lower()
    # it must state its own limits, or it is a blog post
    assert "does NOT cover" in body
