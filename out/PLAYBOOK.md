# Marketplace experiment design: a decision tree

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
  * Network effects between REGIONS. This simulator's regions are independent,
    so geo spillover cannot be represented, and in a real metro adjacent regions
    share couriers.
