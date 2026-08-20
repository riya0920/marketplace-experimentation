"""The analysis upgrades the first pass identified and did not build.

Three things, each of which the first report named as the obvious next step:

TIME-OF-DAY ADJUSTMENT FOR SWITCHBACKS
--------------------------------------
The first pass found that MORE BLOCKS DID NOT BUY PRECISION -- 30-minute blocks
had 232 of them and higher variance than daily blocks with 5. The cause is that
switchback variance is driven by heterogeneity BETWEEN blocks, not their count:
a block at 03:00 and one at 18:30 are different worlds, so an unadjusted
difference in means is dominated by which arm drew the dinner rush.

The fix is to remove the time-of-day mean before differencing. Two estimators
here do it two ways -- a within-hour estimator and a paired-adjacent-blocks
estimator -- because they fail differently and it is worth seeing which.

PAIR-MATCHED GEO ASSIGNMENT
---------------------------
Randomising 12 regions independently gives you whatever imbalance the coin
flips hand you, and with 12 units that imbalance is usually large. Pairing
regions on a PRE-PERIOD covariate and randomising within pairs removes the
imbalance you can see before you start.

MINIMUM DETECTABLE EFFECT
-------------------------
"We have 12 geos" deserves a number, not a shrug. The MDE is what the design can
actually resolve at 80% power, and computing it before the test is what
distinguishes "underpowered" as a finding from "underpowered" as an excuse.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# switchback estimators that control for time of day
# --------------------------------------------------------------------------
def switchback_adjusted(rec: dict, block_minutes: int, burn_in_minutes: int = 0,
                        n_boot: int = 400, seed: int = 0,
                        bucket_minutes: int = 60) -> dict:
    """Difference in means AFTER removing the time-of-day mean.

    Each block is assigned to a time-of-day bucket (hour by default). The
    outcome is centred within its bucket, so the estimate is a difference of
    RESIDUALS rather than of raw rates. Blocks in buckets that saw only one arm
    contribute nothing -- they carry no within-bucket contrast, and including
    them would smuggle the between-bucket variation straight back in.
    """
    m = ~rec["warmup"]
    minute = rec["minute"][m]
    block = minute // block_minutes
    since = minute - block * block_minutes
    keep = since >= burn_in_minutes

    b = block[keep]
    tod = (minute[keep] % 1440) // bucket_minutes
    ex = rec["exposures"][m][keep].astype(float)
    orl = rec["orders"][m][keep].astype(float)
    tex = rec["t_exposures"][m][keep].astype(float)

    rows = []
    for blk in np.unique(b):
        s = b == blk
        e, o = ex[s].sum(), orl[s].sum()
        if e <= 0:
            continue
        rows.append((float(o / e), bool(tex[s].sum() > 0.5 * e), float(e),
                     int(np.bincount(tod[s]).argmax())))
    if not rows:
        return dict(estimate=np.nan, se=np.nan, n_blocks=0, n_buckets_used=0)

    rate = np.array([r[0] for r in rows])
    treat = np.array([r[1] for r in rows])
    wts = np.array([r[2] for r in rows])
    bucket = np.array([r[3] for r in rows])

    # centre within bucket, and drop buckets with only one arm present
    resid = rate.copy()
    usable = np.zeros(len(rows), dtype=bool)
    for bk in np.unique(bucket):
        s = bucket == bk
        if treat[s].sum() == 0 or (~treat[s]).sum() == 0:
            continue
        resid[s] = rate[s] - np.average(rate[s], weights=wts[s])
        usable |= s
    if usable.sum() == 0 or treat[usable].sum() == 0 or (~treat[usable]).sum() == 0:
        return dict(estimate=np.nan, se=np.nan, n_blocks=len(rows), n_buckets_used=0)

    r_, t_, w_ = resid[usable], treat[usable], wts[usable]
    est = (np.average(r_[t_], weights=w_[t_])
           - np.average(r_[~t_], weights=w_[~t_]))

    rng = np.random.default_rng(seed)
    idx_t, idx_c = np.flatnonzero(t_), np.flatnonzero(~t_)
    boots = []
    for _ in range(n_boot):
        st = rng.choice(idx_t, len(idx_t), replace=True)
        sc = rng.choice(idx_c, len(idx_c), replace=True)
        boots.append(np.average(r_[st], weights=w_[st])
                     - np.average(r_[sc], weights=w_[sc]))
    return dict(estimate=float(est), se=float(np.std(boots, ddof=1)),
                n_blocks=len(rows), n_buckets_used=int(len(np.unique(bucket[usable]))))


def switchback_paired(rec: dict, block_minutes: int, burn_in_minutes: int = 0,
                      seed: int = 0) -> dict:
    """Difference within ADJACENT block pairs that happened to split arms.

    Two consecutive blocks are as close to the same market conditions as this
    design gets, so differencing them removes time-of-day almost exactly. The
    price is sample: only pairs that split treated/control contribute, which is
    about half of them.

    This is the estimator with the strongest claim to being unbiased by
    construction and the weakest claim to being efficient, and reporting it
    alongside the adjusted one is how you see that trade.
    """
    m = ~rec["warmup"]
    minute = rec["minute"][m]
    block = minute // block_minutes
    since = minute - block * block_minutes
    keep = since >= burn_in_minutes

    b = block[keep]
    ex = rec["exposures"][m][keep].astype(float)
    orl = rec["orders"][m][keep].astype(float)
    tex = rec["t_exposures"][m][keep].astype(float)

    info = {}
    for blk in np.unique(b):
        s = b == blk
        e, o = ex[s].sum(), orl[s].sum()
        if e > 0:
            info[int(blk)] = (o / e, tex[s].sum() > 0.5 * e, e)

    diffs, weights = [], []
    blocks = sorted(info)
    for a, c in zip(blocks[:-1], blocks[1:]):
        if c != a + 1:
            continue
        ra, ta, wa = info[a]
        rc, tc, wc = info[c]
        if ta == tc:
            continue                    # same arm: no contrast
        d = (ra - rc) if ta else (rc - ra)
        diffs.append(d)
        weights.append(min(wa, wc))
    if not diffs:
        return dict(estimate=np.nan, se=np.nan, n_pairs=0)
    diffs = np.array(diffs)
    weights = np.array(weights, float)
    est = float(np.average(diffs, weights=weights))
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    return dict(estimate=est, se=se, n_pairs=len(diffs))


# --------------------------------------------------------------------------
# pair-matched geo assignment
# --------------------------------------------------------------------------
def pair_matched_assignment(pre_period_metric: np.ndarray, seed: int = 0):
    """Sort regions by a pre-period covariate, pair neighbours, randomise within.

    With 12 regions, independent coin flips routinely produce an arm that is
    systematically busier than the other -- and there is no way to tell that
    imbalance apart from a treatment effect after the fact. Pairing on something
    measured BEFORE the test removes the imbalance you could have seen coming.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(pre_period_metric)
    treated = np.zeros(len(pre_period_metric), dtype=bool)
    for i in range(0, len(order) - 1, 2):
        a, b = order[i], order[i + 1]
        if rng.random() < 0.5:
            treated[a] = True
        else:
            treated[b] = True
    if len(order) % 2 == 1:             # odd region out: plain coin flip
        treated[order[-1]] = rng.random() < 0.5
    return treated


def imbalance(pre_period_metric: np.ndarray, treated: np.ndarray) -> float:
    """Standardised difference in the pre-period covariate between arms.

    This is the diagnostic to look at BEFORE unblinding anything. A large value
    means the arms were different before the treatment existed, and no amount of
    post-hoc adjustment fully rescues that.
    """
    a, b = pre_period_metric[treated], pre_period_metric[~treated]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float(abs(a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


# --------------------------------------------------------------------------
# minimum detectable effect
# --------------------------------------------------------------------------
def mde(se: float, alpha: float = 0.05, power: float = 0.80,
        dof: int | None = None) -> float:
    """Smallest effect this design can detect at the given power.

    MDE = (z_alpha/2 + z_power) * SE, using a t reference when the degrees of
    freedom are small -- which for a geo test they always are, and ignoring that
    is how a 12-cluster design gets reported as adequately powered.
    """
    from scipy import stats
    if dof is not None and dof < 30:
        crit = stats.t.ppf(1 - alpha / 2, dof)
        pw = stats.t.ppf(power, dof)
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
        pw = stats.norm.ppf(power)
    return float((crit + pw) * se)
