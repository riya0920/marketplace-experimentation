# DATA-3 — Marketplace Experimentation

**Complete against the spec.** An agent-based marketplace where a treatment
consumes shared supply, naive-A/B bias measured against known truth by market
tightness, switchback granularity with time-of-day adjustment, cluster-robust geo
inference, pair matching, **CUPED**, **synthetic control with placebo inference**,
an **MDE curve**, a **heterogeneous calendar**, and the **decision-tree playbook**
the spec asked for as a standalone artifact.

```bash
python run_experiments.py    # ~6min  the original three acts
python run_complete.py       # ~30s   CUPED, synthetic control, the MDE curve
python -m pytest tests -q    # 53 tests
cat out/PLAYBOOK.md          #        the deliverable
```

## Days are no longer statistically identical

The previous README flagged this as the artifact that made its own
switchback-variance ranking suspect: *"no day-of-week, no weather, no trend,
which is precisely the artifact flagged in the comparison section."*

- **Day of week** runs 0.86 (Tuesday) to 1.42 (Saturday) — a **65% swing**, larger
  than any treatment effect anyone would test. A five-day test that ran
  Tuesday-to-Thursday and one that caught a weekend are not the same test.
- **Weather** is a per-**day** shock shared across every region, which is what
  makes it a confound rather than noise: a rainy Tuesday raises demand everywhere
  at once, so it cannot be averaged away across regions the way region-specific
  noise can. It is drawn from one seed so every design being compared gets the
  *same* weather — otherwise a design comparison is partly a comparison of the
  weather it drew.
- **Trend** is 0.4%/day. With an assignment unbalanced across days, a trend is
  attributed to the treatment, and no amount of within-day randomisation fixes it.

## CUPED — the cheapest variance reduction there is

| | value |
|---|---|
| pre/post correlation across regions | **+0.8905** |
| variance reduction predicted (ρ²) | 0.7931 |
| variance reduction **realised** | **0.7931** |
| effect, unadjusted | +0.02682 (se **0.01808**) |
| effect, CUPED | +0.02455 (se **0.00409**) |
| **standard error cut by** | **77.4%** |

CUPED subtracts `θ · (pre-period metric − its mean)` from the outcome, with θ
chosen to minimise variance. It costs nothing, it cannot bias the estimate
provided the covariate is pre-treatment, and **the variance reduction is exactly
ρ²** — so its value is knowable in advance from a correlation you already have.
That is why it belongs before any conversation about running longer.

**The one correctness condition is not checkable from the numbers.** A
post-treatment covariate makes CUPED silently *biased* rather than merely useless,
because it adjusts away part of the effect. Nothing in the data reveals that, so
the obligation sits with whoever chooses the column.

Predicted and realised reduction agree to four decimals, which is the check worth
doing: if they diverge, the covariate is not behaving the way the theory assumed.

> θ is a **regression coefficient**, not a correlation. Using the correlation
> under-adjusts whenever the two have different scales, and the slip is invisible
> at unit scale — a test pins it.

## Synthetic control — when you cannot add more units

```
treated region        : 2
donors in the pool    : 7, with weight > 0.01: 7
pre-period fit (RMSE) : 0.04609  (9.0% of the level)
estimated effect      : +0.03792
placebo permutation   : p over 11 usable placebos
```

Instead of adjusting a unit by its own past, it builds a **weighted combination of
control units** that tracks the treated unit's pre-period. It changes what a unit
*is* rather than adding more of them, which is why it is the answer to "12
clusters is underpowered".

- **The weights are constrained to a simplex** — non-negative, summing to one —
  and that is not decoration. Unconstrained least squares hands a donor a weight
  of −3, which *extrapolates*; the credibility of the method rests entirely on the
  counterfactual being a weighted average of things that actually happened.
- **Inference is a permutation test**, because there is no standard error. Each
  control unit is refitted pretending it was treated, and the question is whether
  the real effect is unusual among those placebos.
- **Placebos with a poor pre-fit are excluded.** A unit the donor pool cannot
  reproduce shows a large "effect" regardless of treatment, and including them is
  the standard way this test acquires an artificially small p-value.
- **The pre-period fit is reported next to every estimate**, because it decides
  whether the estimate means anything. A synthetic control with a bad pre-fit has
  not produced a weak result — it has failed to build a counterfactual. A test
  plants a treated unit outside the donor hull and asserts the RMSE catches it.

## The MDE curve — "how long do we need to run this"

| days | MDE | MDE with CUPED |
|---|---|---|
| 1 | 0.14864 | **0.03364** |
| 5 | 0.06647 | 0.01504 |
| 7 | 0.05618 | 0.01271 |
| 14 | 0.03972 | 0.00899 |
| **21** | **0.03243** | 0.00734 |
| 56 | 0.01986 | 0.00450 |

True effect on the conversion rate: **0.03720**.

**Days needed: 21 unadjusted, 1 with CUPED.** The cheapest variance reduction
there is buys three weeks of calendar.

**The curve flattens, and that is the whole reason to draw it.** Standard error
falls as 1/√days, so *doubling* the test improves the MDE by only 30% — every step
in the table buys 13–18%. Past some point another week buys almost nothing, and
that point is where "run it longer" stops being an option: the honest answer
becomes change the design, change the metric, or do not run it.

A single MDE answers "can we detect X in five days", which is a yes/no about a
decision nobody made. The curve answers the question that was asked.

**The scaling assumes independent days and is therefore optimistic.** Demand is
autocorrelated — a rainy week is a rainy week — so the true standard error falls
more slowly than 1/√days and every duration here is a **lower bound**. Stating the
direction matters more than correcting it: a planner who knows the estimate is
optimistic will pad it.

## The playbook, as an artifact

`out/PLAYBOOK.md` is a standalone decision tree: does the treatment change the
shared resource → switchback or geo → are you powered → what to do when you are
not. **Every branch is a conclusion this project measured**, with the section
named so the number can be checked. That is the difference between a playbook and
a blog post: a reader can disagree with a branch by disputing a specific number
rather than a preference.

It also states what it does not cover — sequential testing, multiple comparisons,
heterogeneous effects, and network effects *between* regions, which this simulator
cannot represent because its regions are independent and a real metro's are not.

## Bugs this pass caught

- **The analysis reconstructed the assignment by re-seeding a generator** instead
  of reading it off the assignment object. The two draws did not match, so the
  "treated" group in the analysis was not the group the simulator treated — and
  the estimated effect came out **negative** against a positive planted lift. An
  assignment that has to be guessed by the analysis is one that will eventually be
  guessed wrong.
- **The synthetic-control fixture placed the treated unit outside the donor
  hull**, so a simplex fit could not reproduce it and the unmatched level leaked
  straight into the "effect" (0.42 against a planted 0.25). That is now its own
  test rather than a fixture accident.

## What is deliberately not here

- **No sequential testing or always-valid inference.** Everything assumes a fixed
  horizon decided in advance, and peeking invalidates all of it.
- **Couriers still do not reposition, accept or decline** in *this* simulator —
  SE-3 models all three, and the two are not joined. That remains the biggest
  fidelity gap and it matters most for the courier-incentive case.
- **No merchant side**, so it is a two-sided model of a three-sided market. No
  prep times, no merchant capacity.
- **Single metric.** Conversion only; no basket size, no retention, no
  cannibalisation between regions.
- **Regions are independent**, so geo spillover cannot be represented at all. In a
  real metro adjacent regions share couriers, and that is exactly the failure a
  geo design is supposed to be protected against.
