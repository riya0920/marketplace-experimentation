# DATA-3 — Marketplace Experimentation: Interference & Switchbacks

**This is not deployable.** It is the first ~20% of the spec: a marketplace
simulator with an honest interference channel, the naive A/B's bias measured
against known truth, and two mitigations scored on bias *and* variance. Missing
80% at the bottom.

```bash
python run_experiments.py    # ~25min
python -m pytest tests -q    # 15 tests
```

## The simulator is a lab, not a claim

Nothing here is calibrated to a real company. It earns its keep by making the
**truth knowable**, so estimators can be scored instead of argued about. Its
stylised facts are checked every run and reported before any result:

| check | |
|---|---|
| delivery time rises with utilisation | PASS |
| …and is convex as utilisation → 1 | PASS |
| conversion falls with quoted delivery time | PASS |
| demand has lunch and dinner peaks | PASS |

The causal chain that makes interference real, and a test asserts each link:

```
treatment raises conversion → more orders → more couriers busy
  → utilisation rises → quoted delivery time rises
    → conversion falls for EVERYONE in the region, including control
```

The conclusions that transfer are about **estimators**, not about DoorDash.

## Act 1 — the naive A/B is biased, and the bias grows with tightness

True effect = conversion(everyone treated) − conversion(nobody treated). That's
the estimand that matters, because it's what shipping does.

| market | couriers | utilisation | truth | naive estimate | **overstated by** |
|---|---|---|---|---|---|
| slack | 26 | 25.5% | +0.0470 | +0.0451 | −4.0% |
| normal | 17 | 35.5% | +0.0394 | +0.0488 | **+23.9%** |
| tight | 12 | 43.1% | +0.0330 | +0.0438 | **+32.5%** |
| very tight | 9 | 50.1% | +0.0268 | +0.0379 | **+41.2%** |

The control group is not a set of unaffected bystanders — it's a group the
treatment actively made worse off by taking the couriers it was competing for.
Measuring (treated − control) counts the treatment's benefit *and* the damage it
did to its own baseline. SUTVA is violated by construction, and in a marketplace
it is always violated; the only question is by how much.

Note that the reported SE (0.0077) is small and **correct** for how randomisation
happened. The bias is in the point estimate, not the variance, so more users
cannot fix it — they just tighten the interval around the wrong number.

> Truth here is averaged over 4 all-treated/all-control run pairs with common
> random numbers. A single pair estimates truth with roughly the same SE as the
> estimator being scored against it, and the first version of this table was
> non-monotone in tightness for exactly that reason.

## Act 2 — the mitigations, and a result I didn't expect

Every design's bias is reported **against its own Monte Carlo error** (12
replications), because a bias smaller than twice its MC SE is not a measurement.

| block | burn-in | n blocks | bias | MC SE | verdict |
|---|---|---|---|---|---|
| 30 min | 0 | 232 | +0.0029 | 0.0066 | not distinguishable from 0 |
| 30 min | 20 | 232 | +0.0007 | 0.0066 | not distinguishable from 0 |
| 120 min | 0 | 58 | +0.0018 | 0.0090 | not distinguishable from 0 |
| 120 min | 20 | 58 | −0.0001 | 0.0090 | not distinguishable from 0 |
| 480 min | 0 | 15 | −0.0076 | 0.0064 | not distinguishable from 0 |
| 1440 min | 0 | 5 | +0.0001 | 0.0026 | not distinguishable from 0 |

**No switchback granularity shows resolvable bias** — including 30-minute blocks
whose carryover window (~34 min service time) is longer than the block itself.
The naive design's bias, by contrast, is many times its own MC error. The
mitigation works, and the interesting question moves from bias to **variance**.

And there the naive intuition breaks:

| block | n blocks | sd across reps |
|---|---|---|
| 30 min | 232 | 0.0227 |
| 120 min | 58 | **0.0312** |
| 480 min | 15 | 0.0220 |
| 1440 min | 5 | **0.0091** |

**More blocks did not buy precision.** The daily design has the fewest blocks and
the lowest variance. Switchback variance is driven by heterogeneity *between*
blocks, not by their count: a 30-minute block at 03:00 and one at 18:30 are
different worlds, so an unadjusted difference in means is dominated by which arm
drew the dinner rush.

The practitioner consequence: **short blocks are only worth their extra count if
the analysis controls for time-of-day** — paired adjacent blocks, or block-level
covariate adjustment. Running fine-grained switchbacks and then taking a raw
difference in means throws away the precision the design was chosen for. That
adjustment is not implemented here; it's the first thing I'd add.

Burn-in moves every row toward truth (30-min: +0.0029 → +0.0007; 120-min:
+0.0018 → −0.0001) — the direction carryover predicts, consistent across
granularities, but no individual move clears its MC error. So the honest
statement is *"directionally consistent with carryover, not resolved at 12
reps"*, not *"burn-in removes X of bias"*.

## Act 2 continued — geo, and the inference error that manufactures significance

| clusters | bias | MC SE | cluster-robust SE | naive SE if misanalysed | **understatement** |
|---|---|---|---|---|---|
| 12 | +0.0029 | 0.0061 | 0.0209 | 0.0081 | **2.6×** |
| 24 | +0.0040 | 0.0033 | 0.0139 | 0.0058 | 2.4× |
| 40 | +0.0017 | 0.0017 | 0.0108 | 0.0044 | 2.4× |

The effective sample size is the number of **regions**, not customers — customers
inside a region share a courier pool and are nothing like independent. Analysing
a geo test with a two-proportion SE understates by 2.4–2.6× and manufactures
significance out of nothing. With few clusters even the robust SE is optimistic,
so a t reference with df = clusters − 2 is used. At 12 clusters that's 10 df, and
the honest answer to "we have 12 geos" is often that the effect isn't detectable
at this size.

## The comparison

True effect +0.0330 (tight market):

| design | bias | reported SE |
|---|---|---|
| naive user-level A/B | **+0.0110** | 0.0078 |
| switchback 1440 min | +0.0001 | 0.0064 |
| cluster / geo (12) | +0.0029 | 0.0209 |

The geo design pays the expected price — 2.7× the naive SE, textbook
bias-variance. **The switchback does not**, and that's flagged rather than
claimed: the winning row is the *daily* blocks, and this simulator's days are
statistically identical, so daily blocks are unusually stable here in a way a
real week (day-of-week, weather) would not be. The finer granularities *do* carry
2–3× the variance, which is what a real deployment would pay. Reading the daily
row as "switchbacks are free" would be reading the lab as the world.

## Two-sided guardrails

| | control | treated | delta |
|---|---|---|---|
| orders | 7,778 | 8,323 | **+7.0%** |
| mean utilisation | 0.431 | 0.462 | +3.1 pts |
| p95 delivery time | 44.0 min | 57.0 min | **+13.0 min** |

A demand-side-only readout calls this a clean win. **Ship? Not on this evidence.**
The supply side is absorbing the whole gain, and the two things that decide it —
whether couriers churn at sustained utilisation, and whether customers who waited
longer come back — are long-run quantities absent from a 5-day experiment. The
honest recommendation is a staged rollout with supply-side and repeat-rate
guardrails monitored past the experiment window.

## The other 80% — what is NOT here

- **No experiment-design playbook document.** The spec asks for a decision tree
  artifact; the guidance exists as report prose, not as a standalone deliverable.
- **No time-of-day adjustment for switchbacks** — which the variance result above
  says is the single most valuable missing piece.
- **No pair-matched geo assignment, no covariate adjustment, no synthetic
  control.** These are named as the answer to "12 clusters is underpowered" and
  none is implemented.
- **No power analysis / MDE curves.** The report says when an effect isn't
  detectable but never computes the minimum detectable effect for a given design
  and duration.
- **Couriers do not reposition, accept, decline, or go offline.** They are a
  capacity pool with a service-time distribution. That is the biggest fidelity
  gap and it matters most for the courier-incentive case the report reasons about
  qualitatively — carryover there is driven by repositioning, which this
  simulator cannot represent at all.
- **No restaurant/merchant side**, so it is a two-sided model of a three-sided
  market. No prep times, no merchant capacity.
- **Single metric.** Conversion only; no basket size, no retention, no
  cannibalisation between regions.
- **Days are statistically identical** — no day-of-week, no weather, no trend,
  which is precisely the artifact flagged in the comparison section.

**Which conclusion is most sensitive to what the simulator gets wrong:** the
switchback variance ranking. It depends on between-block heterogeneity, and this
simulator has exactly one source of it (time-of-day) where a real market has
several. The bias results are far more robust — they depend only on the
interference channel existing, which is tested directly.
