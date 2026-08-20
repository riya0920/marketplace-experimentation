"""Estimators, and the inference that goes with each design.

The recurring mistake is not choosing the wrong design, it is choosing the right
design and then analysing it as though it were a simple A/B test. Each function
here pairs an estimate with the standard error appropriate to how randomisation
actually happened.
"""
from __future__ import annotations

import numpy as np


def global_effect(rec_treated: dict, rec_control: dict, warmup: bool = True) -> float:
    """GROUND TRUTH: conversion under all-treated minus conversion under
    all-control, in otherwise identical worlds.

    This is the estimand that matters -- what happens if we SHIP it -- and it is
    the thing a marketplace experiment is trying to approximate. It is knowable
    here only because this is a simulator.
    """
    def cr(rec):
        m = ~rec["warmup"] if warmup else np.ones_like(rec["warmup"])
        return rec["orders"][m].sum() / max(rec["exposures"][m].sum(), 1)
    return cr(rec_treated) - cr(rec_control)


def naive_ab(rec: dict) -> dict:
    """Treated conversion minus control conversion, within one run.

    Two-proportion SE. It is the SE that is right for the randomisation (users
    were randomised independently) and still gives a confidently wrong answer,
    because the bias is in the ESTIMATE, not in the variance. Tight intervals
    around a biased point estimate are the failure mode.
    """
    m = ~rec["warmup"]
    te, to = rec["t_exposures"][m].sum(), rec["t_orders"][m].sum()
    ce, co = rec["c_exposures"][m].sum(), rec["c_orders"][m].sum()
    pt, pc = to / max(te, 1), co / max(ce, 1)
    se = np.sqrt(pt * (1 - pt) / max(te, 1) + pc * (1 - pc) / max(ce, 1))
    return dict(estimate=pt - pc, se=se, n_treated=int(te), n_control=int(ce))


def switchback(rec: dict, block_minutes: int, burn_in_minutes: int = 0,
               n_boot: int = 400, seed: int = 0) -> dict:
    """Block-level difference in means, with block bootstrap inference.

    TWO THINGS THAT ARE EASY TO GET WRONG AND ARE DONE HERE:

    burn-in -- the first minutes of a block are contaminated by the previous
    block. Couriers are still delivering orders placed under the other arm, so
    utilisation (and therefore conversion) reflects a mixture. Those minutes are
    dropped.

    block bootstrap -- observations within a block are anything but independent;
    they share a courier pool and a congestion state. Resampling minutes would
    understate the SE badly. Whole blocks are resampled instead.
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

    ub = np.unique(b)
    rates, treated_flags, weights = [], [], []
    for blk in ub:
        s = b == blk
        e, o = ex[s].sum(), orl[s].sum()
        if e <= 0:
            continue
        rates.append(o / e)
        # a switchback block is entirely one arm; >50% treated exposures
        # identifies which
        treated_flags.append(tex[s].sum() > 0.5 * e)
        weights.append(e)
    rates = np.array(rates)
    tf = np.array(treated_flags)
    wts = np.array(weights)
    if tf.sum() == 0 or (~tf).sum() == 0:
        return dict(estimate=np.nan, se=np.nan, n_blocks=len(rates))

    est = (np.average(rates[tf], weights=wts[tf])
           - np.average(rates[~tf], weights=wts[~tf]))

    rng = np.random.default_rng(seed)
    boots = []
    idx_t, idx_c = np.flatnonzero(tf), np.flatnonzero(~tf)
    for _ in range(n_boot):
        st = rng.choice(idx_t, len(idx_t), replace=True)
        sc = rng.choice(idx_c, len(idx_c), replace=True)
        boots.append(np.average(rates[st], weights=wts[st])
                     - np.average(rates[sc], weights=wts[sc]))
    return dict(estimate=est, se=float(np.std(boots, ddof=1)),
                n_blocks=int(len(rates)), n_treated_blocks=int(tf.sum()))


def cluster_randomised(rec: dict, treated_regions: np.ndarray) -> dict:
    """Region-level difference in means with CLUSTER-ROBUST standard errors.

    The effective sample size is the number of REGIONS, not the number of
    customers. Computing a two-proportion SE over hundreds of thousands of
    customers here would understate the true SE by an order of magnitude and
    produce significance that is entirely manufactured -- the single most common
    way a geo test is misread.

    With few clusters even the robust SE is optimistic, so the small-sample
    correction and the t reference distribution (df = clusters - 2) are applied.
    """
    m = ~rec["warmup"]
    region = rec["region"][m]
    ex, orl = rec["exposures"][m], rec["orders"][m]

    rates, treat, wts = [], [], []
    for r in np.unique(region):
        s = region == r
        e = ex[s].sum()
        if e <= 0:
            continue
        rates.append(orl[s].sum() / e)
        treat.append(bool(treated_regions[r]))
        wts.append(e)
    rates, treat, wts = np.array(rates), np.array(treat), np.array(wts, float)
    nt, nc = int(treat.sum()), int((~treat).sum())
    if nt < 2 or nc < 2:
        return dict(estimate=np.nan, se=np.nan, n_clusters=len(rates))

    est = rates[treat].mean() - rates[~treat].mean()
    # cluster-robust: variance of CLUSTER means, not of customers
    vt = rates[treat].var(ddof=1) / nt
    vc = rates[~treat].var(ddof=1) / nc
    se = float(np.sqrt(vt + vc))
    dof = max(nt + nc - 2, 1)
    correction = np.sqrt((nt + nc) / max(nt + nc - 2, 1))   # small-cluster inflation
    return dict(estimate=est, se=se * correction, n_clusters=int(nt + nc),
                n_treated_clusters=nt, dof=dof)


def naive_se_for_cluster_design(rec: dict) -> float:
    """What you get if you analyse a geo test as if users were randomised.

    Reported alongside the correct SE so the size of the error is visible rather
    than asserted.
    """
    m = ~rec["warmup"]
    te, to = rec["t_exposures"][m].sum(), rec["t_orders"][m].sum()
    ce, co = rec["c_exposures"][m].sum(), rec["c_orders"][m].sum()
    pt, pc = to / max(te, 1), co / max(ce, 1)
    return float(np.sqrt(pt * (1 - pt) / max(te, 1) + pc * (1 - pc) / max(ce, 1)))


def guardrails(rec: dict) -> dict:
    """Supply-side metrics. A marketplace experiment that reports only the
    demand-side metric is half an analysis."""
    m = ~rec["warmup"]
    w = rec["exposures"][m].astype(float)
    w = w if w.sum() > 0 else np.ones_like(w)
    return dict(mean_utilisation=float(np.average(rec["utilisation"][m], weights=w)),
                p95_utilisation=float(np.percentile(rec["utilisation"][m], 95)),
                mean_eta=float(np.average(rec["eta"][m], weights=w)),
                p95_eta=float(np.percentile(rec["eta"][m], 95)),
                orders=int(rec["orders"][m].sum()),
                exposures=int(rec["exposures"][m].sum()))
