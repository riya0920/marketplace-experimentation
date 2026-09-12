"""MDE as a CURVE over test length, and the playbook that reads off it.

TWO GAPS
--------
  "MDE is computed at one duration. The useful artifact is an MDE curve over test
   length, which is what actually answers 'how long do we need to run this'."
  "No experiment-design playbook document. The spec asks for a decision tree
   artifact; the guidance exists as report prose, not as a standalone
   deliverable."

WHY A CURVE AND NOT A NUMBER
----------------------------
A single MDE answers "can we detect X in five days", which is a yes/no about a
decision nobody made. The question a PM brings is "how long", and the answer is a
curve -- specifically, one that flattens. Standard error falls as 1/sqrt(n), so
doubling the test length improves the MDE by only 30%, and past some point
another week buys almost nothing.

That flattening is the useful shape: it is where "run it longer" stops being an
option and the honest answer becomes "change the design, change the metric, or
do not run this test".
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mde_from_se(se: float, n_clusters: int, alpha: float = 0.05,
                power: float = 0.80) -> float:
    """Minimum detectable effect, with a t reference on clusters-2 df.

    The t reference rather than the normal is what stops a 12-cluster test from
    being reported as adequately powered. At 10 degrees of freedom the two-sided
    5% critical value is 2.23 rather than 1.96 -- a 14% difference in the MDE,
    which is exactly the margin that decides whether a test looks feasible.
    """
    df = max(n_clusters - 2, 1)
    t_alpha = stats.t.ppf(1 - alpha / 2, df)
    t_beta = stats.t.ppf(power, df)
    return float((t_alpha + t_beta) * se)


def mde_curve(se_at_reference: float, reference_days: float, n_clusters: int,
              days=(1, 2, 3, 5, 7, 10, 14, 21, 28, 42, 56),
              alpha: float = 0.05, power: float = 0.80) -> list[dict]:
    """Scale a measured standard error by 1/sqrt(days) and read off the MDE.

    Scaling an OBSERVED se rather than assuming a variance is the honest version:
    the observed one already contains whatever clustering, autocorrelation and
    heterogeneity this marketplace has, and a formula-derived variance would
    assume them away.

    THE SCALING ASSUMES INDEPENDENT DAYS, and that is optimistic. Demand is
    autocorrelated -- a rainy week is a rainy week -- so the true se falls more
    slowly than 1/sqrt(days) and every duration here is a LOWER BOUND on how long
    the test needs. Stating the direction of the error matters more than
    correcting it: a planner who knows the estimate is optimistic will pad it.
    """
    out = []
    for d in days:
        se = se_at_reference * np.sqrt(reference_days / d)
        out.append(dict(days=d, se=float(se),
                        mde=mde_from_se(se, n_clusters, alpha, power)))
    return out


def days_needed(curve: list[dict], target_effect: float) -> float | None:
    """The first duration on the curve whose MDE is below the target effect."""
    for row in curve:
        if row["mde"] <= abs(target_effect):
            return row["days"]
    return None


def marginal_value(curve: list[dict]) -> list[dict]:
    """How much each extra week buys. This is where the curve flattens."""
    out = []
    for a, b in zip(curve[:-1], curve[1:]):
        out.append(dict(from_days=a["days"], to_days=b["days"],
                        mde_before=a["mde"], mde_after=b["mde"],
                        improvement=1 - b["mde"] / a["mde"] if a["mde"] else 0.0))
    return out


# --------------------------------------------------------------------------
# the playbook
# --------------------------------------------------------------------------
PLAYBOOK = """# Marketplace experiment design: a decision tree

Every branch below is a conclusion this project MEASURED rather than an opinion.
The section it came from is named so the number can be checked.

---

## 1. Does the treatment change the SHARED RESOURCE?

In a delivery marketplace the shared resource is courier supply. A treatment that
raises conversion consumes couriers, which raises ETAs for everyone -- including
the control group.

**If yes -> a user-level A/B is BIASED, not merely noisy.** The control group is
contaminated by the treatment's effect on supply, so SUTVA fails and the
estimate is wrong in a direction that depends on market tightness.

  * Measured: the naive A/B bias GROWS with tightness. (Act 1)
  * The bias is not a small correction; in a tight market it can exceed the
    effect being measured.

**If no** -- the treatment touches only the individual, e.g. a UI change with no
supply consequence -- a user-level A/B is fine and is the most powerful design
available. Use it.

---

## 2. If the resource IS shared: switchback or geo?

### Switchback -- randomise TIME

Use when the effect acts fast and washes out fast, and when you have one market
rather than many.

  * Blocks must be long enough that carryover has decayed. Burn-in the first
    minutes of every block and discard them; the burn-in length is a property of
    the treatment, not of the design.
  * **More blocks do NOT buy precision.** Measured: 30-minute blocks had 232 of
    them and HIGHER variance than daily blocks with 5, because switchback
    variance is driven by heterogeneity BETWEEN blocks (time of day), not by
    their count. (Act 2)
  * **Adjust for time of day.** Measured: 41-82% variance reduction across
    block sizes, and it is the single largest win in this project. Short blocks
    are only worth their count if the analysis controls for the thing that makes
    blocks differ.

### Geo -- randomise REGIONS

Use when the effect is slow, sticky, or spills across time (loyalty, pricing,
supply incentives), and when you have enough comparable markets.

  * **Cluster-robust standard errors, always.** Measured: naive SEs manufacture
    significance by treating correlated users inside a region as independent.
    (Act 2 continued)
  * **Pair-match on the pre-period.** Measured: ~5x lower pre-period imbalance
    and ~30% lower variance at 24 clusters. Independent coin flips hand you
    whatever imbalance they hand you. (Second pass)
  * **Use a t reference on clusters-2 degrees of freedom.** Using the normal is
    how a 12-cluster test gets reported as adequately powered.

---

## 3. Before running ANYTHING: are you powered?

Compute the MDE curve, not a single MDE.

  * **Measured: none of the geo designs here can resolve the true effect at 5
    days** -- MDE +0.037 to +0.065 against a true +0.033, at every cluster count
    and both assignment schemes. That verdict is available BEFORE running rather
    than after seeing a null and calling it evidence of no effect.
  * The curve FLATTENS: standard error falls as 1/sqrt(days), so doubling the
    duration improves the MDE by only 30%. Find where another week stops buying
    anything and treat that as the boundary of the design.

**If the MDE never reaches the effect at any plausible duration**, the options in
order of preference are:

  1. **Reduce variance, not time.** CUPED on a pre-period covariate is free and
     its benefit is knowable in advance -- the variance reduction is exactly
     rho^2. Do this first, always.
  2. **Change the unit.** Fewer, larger clusters have lower variance and fewer
     degrees of freedom; the trade is not obvious and has to be computed.
  3. **Change the metric.** A metric closer to the mechanism is less noisy: if
     the treatment works through ETA, measure ETA.
  4. **Synthetic control.** The answer when you cannot add units at all --
     it changes what a unit IS. Requires a long pre-period and inference by
     placebo permutation, and it FAILS QUIETLY when the pre-period fit is poor.
  5. **Do not run the test.** A design that is unbiased and too noisy to
     conclude anything has not helped anyone ship.

---

## 4. Whatever you run: the guardrails

  * A two-sided market has two sides. Measure the courier side as well as the
    customer side; a treatment that raises conversion by starving couriers is
    not a win.
  * Check pre-period imbalance BEFORE unblinding. A large value means the arms
    differed before the treatment did, and no post-hoc adjustment fully rescues
    that.
  * Report the MDE next to the result. A null with an MDE larger than the effect
    you cared about is not evidence of no effect.

---

## What this playbook does NOT cover

  * Sequential testing and always-valid inference. Everything above assumes a
    fixed horizon decided in advance, and peeking invalidates all of it.
  * Multiple comparisons across metrics.
  * Heterogeneous treatment effects -- every design here estimates one average.
  * Cross-region demand spillover. Couriers now move between regions under
    SE-3's repositioning policy (see the mobility experiment), so regions are no
    longer fully independent -- but that repairs the congestion-feedback
    interference this project already measures rather than modelling true demand
    spillover between regions, which the simulator still does not represent.
"""


def write_playbook(path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(PLAYBOOK)
    return path
