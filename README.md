# DATA-3 — Marketplace Experimentation

**Complete against the spec.** An agent-based marketplace where a treatment
consumes shared supply, naive-A/B bias measured against known truth by market
tightness, switchback granularity with time-of-day adjustment, cluster-robust geo
inference, pair matching, **CUPED**, **synthetic control with placebo inference**,
an **MDE curve**, a **heterogeneous calendar**, **couriers that move between
regions under SE-3's policy**, and the **decision-tree playbook** the spec asked
for as a standalone artifact.

```bash
python run_experiments.py    # ~6min  the original three acts
python run_complete.py       # ~12min CUPED, synthetic control, MDE curve, mobility
python -m pytest tests -q    # 67 tests
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
| pre/post correlation across regions | **+0.9421** |
| variance reduction predicted (ρ²) | 0.8876 |
| variance reduction **realised** | **0.8876** |
| effect, unadjusted | +0.03401 (se **0.01593**) |
| effect, CUPED | +0.01600 (se **0.00443**) |
| **standard error cut by** | **72.2%** |

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

## Couriers that move — the fidelity gap, and the sign I got wrong

This project named the same gap twice: *"couriers still do not reposition in this
simulator"* and *"regions are independent, so geo spillover cannot be represented
at all."* A geo design's entire claim is that randomising by region removes
interference, and a simulator whose couriers cannot move can never show that claim
failing.

Couriers now move, under **SE-3's repositioning policy imported by file path**
rather than reimplemented. A rule written *here* to demonstrate spillover would be
a rule written to demonstrate spillover; SE-3's was built for a different question
and measured against an oracle placement.

| market | couriers | utilisation | static bias | mobile bias | paired diff | t | crossing |
|---|---|---|---|---|---|---|---|
| slack | 14 | 0.258 | −0.0223 | −0.0170 | +0.0053 | 0.78 | 46% |
| tight | 5 | 0.448 | −0.0478 | −0.0327 | **+0.0151** | **1.53** | 45% |

True lift 0.0800. 40 seeds per row, **paired on seed** — both arms see the same
weather, the same assignment and the same demand draws, and only the couriers'
ability to move differs.

**Nearly half the couriers end in a region of the opposite arm, and in the slack
market the estimate does not care** (t = 0.78). They cross in response to *region
heterogeneity*, drawn from a seed that has nothing to do with the assignment, so
the crossing is uncorrelated with the arms and averages out of the contrast.

**Crossing is necessary for interference and it is not sufficient.** A spillover
audit that measured the crossing rate and stopped would have condemned a design
that is, here, fine.

### And the sign is the opposite of the one I went looking for

The worry that motivated this section is that mobile couriers leak treatment into
control and **inflate** the estimate. In the tight market the paired difference is
**+0.0151** — mobility moves the estimate *up toward the true lift*, from −0.0478
of bias to −0.0327.

**Mobility does not add spillover here. It partially repairs the interference this
project already measures.** The static bias is congestion feedback — treatment
raises demand, demand raises utilisation, utilisation lengthens ETAs and suppresses
the very conversion being measured — and couriers moving toward busy regions
relieve exactly that congestion. The mechanism people build geo designs to defend
against is, in this market, working the other way.

At t = 1.53 on 40 paired seeds that is **a hint, not a finding**, and the report
says so. A 50-seed run at three days reads +0.0125 at t = 1.98 — same sign, same
size, still short.

### Why the effect is small, and the limit that puts on all of it

The treatment reaches the repositioning policy through exactly one channel: it
raises conversion → utilisation → surge → a region's attractiveness. Measured
inside a run, **treated regions carry surge 1.042 against control's 1.040**, and
the policy moves nobody below a 1.25 ratio.

Surge activates above 0.55 utilisation; this simulator reaches **0.448**. Pushing
further does not produce a tighter market so much as a broken one: at 4 couriers
the mean quoted ETA is **237 minutes**, at 3 it is 303. A four-hour delivery is not
a tight marketplace, it is a saturated queue.

**So the claim is bounded.** Courier mobility does not create geo spillover bias
anywhere this simulator can credibly go. The mechanism is not absent — it is
*unreachable*, because congestion chokes demand before utilisation rises enough for
supply to chase treatment. *"Mobility does not cause geo bias"* is the quotable
version and it is missing the only sentence that makes it true.

### Three bugs, all kept

- **A ratio of two small counts moved the fleet.** The first version passed a raw
  trailing demand estimate to the policy; at 00:30, still warming up, it read
  `[0.5, 0, 0, 1.1, 0, 1.3]` orders per half-hour. SE-3's policy decides on a
  **ratio**, and 1.1 against 0.5 clears its 1.25 threshold on a difference of six
  tenths of an order. Over a day of half-hourly decisions it ratcheted: **87 of 168
  couriers ended in one region.** Shrinking the estimate toward its pooled mean
  breaks the ratchet — the most crowded region holds 12% of the fleet with it and
  38% without. The repair went into the *estimate*, not into SE-3's policy: a ratio
  is the right shape for the decision, and editing another project's tuned rule to
  fix this project's input is the wrong place for it.
- **And the boundary of that fix is now a test too.** Shrinkage does *not*
  neutralise a single degenerate estimate — that vector shrunk by its own mean
  still spans a 3.69 ratio and still moves couriers. It defuses the *repetition*.
  The docstring claimed the stronger thing until a test disagreed.
- **The weather was seeded with `hash(("weather", days))`.** Python salts string
  hashing per process, so the comment promising that *"the same seed gives the same
  weather to every design being compared"* was **true within one run and false
  across two**: comparisons inside a report were sound, and nobody could reproduce
  the report. Caught when this sweep printed different t-statistics on a second run
  with every seed unchanged.
- **Fixing it exposed a test that had been a coin flip.**
  `test_cluster_se_is_larger_than_the_naive_se` asserted `robust > naive * 1.5` on
  a single seed; the measured ratio runs **1.04 to 1.80, median 1.39**. It never
  flickered only because the weather was silently re-rolling every process. It now
  asserts across eight seeds what is actually true: the cluster-robust standard
  error is *always* larger, and typically about 40% larger.

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
- **Couriers reposition; they still do not accept or decline.** SE-3 models the
  acceptance side and it is not wired in, so a courier here never turns work down
  — which is the channel a courier-incentive treatment would act on most directly.
- **The market cannot be made tight enough to test the mechanism that matters.**
  Congestion chokes demand before utilisation reaches the level where supply
  chases treatment, so the spillover result above is bounded by the simulator
  rather than by the finding.
- **No merchant side**, so it is a two-sided model of a three-sided market. No
  prep times, no merchant capacity.
- **Single metric.** Conversion only; no basket size, no retention, no
  cannibalisation between regions.
- **Regions share couriers now, but nothing else.** No demand substitution across
  a boundary: a customer who gives up in one region does not order from the next
  one, so the only spillover channel is supply.
