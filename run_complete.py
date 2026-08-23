"""The completion pass: CUPED, synthetic control, the MDE curve, a heterogeneous
calendar, and the playbook as an artifact.

Run after nothing. Writes out/complete_report.txt and out/PLAYBOOK.md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import adjustment as ADJ    # noqa: E402
from src import designs as DES       # noqa: E402
from src import estimators as EST    # noqa: E402
from src import market as MKT        # noqa: E402
from src import power as POW         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

DAYS = 7
LIFT = 0.06
N_REGIONS = 12


def region_daily(rec, days: int, n_regions: int):
    """Per (region, day) exposures and conversion rate."""
    day = rec["minute"] // MKT.MINUTES_PER_DAY
    df = pd.DataFrame(dict(region=rec["region"], day=day,
                           exposures=rec["exposures"], orders=rec["orders"]))
    g = df.groupby(["region", "day"]).sum().reset_index()
    g["rate"] = g.orders / g.exposures.replace(0, np.nan)
    return g


def main():
    os.makedirs(OUT, exist_ok=True)
    lines, summary = [], {}

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("DATA-3 COMPLETION PASS")
    emit("=" * 78)
    emit("")

    # ======================================================================
    emit("=" * 78)
    emit("A. DAYS ARE NO LONGER STATISTICALLY IDENTICAL")
    emit("=" * 78)
    emit("The previous README flagged this as the artifact that made the")
    emit("switchback-variance ranking suspect: 'no day-of-week, no weather, no")
    emit("trend, which is precisely the artifact flagged in the comparison")
    emit("section'.")
    emit("")
    rng = np.random.default_rng(0)
    rows = []
    for d in range(14):
        rows.append(dict(day=d, dow=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
                                     "Sun"][d % 7],
                         multiplier=MKT.day_multiplier(d, None)))
    D = pd.DataFrame(rows)
    emit(D.to_string(index=False, float_format=lambda x: "%9.4f" % x))
    emit("")
    emit("  Day-of-week runs %.2f (Tue) to %.2f (Sat) -- a %.0f%% swing, larger"
         % (MKT.DOW_MULTIPLIER.min(), MKT.DOW_MULTIPLIER.max(),
            100 * (MKT.DOW_MULTIPLIER.max() / MKT.DOW_MULTIPLIER.min() - 1)))
    emit("  than any treatment effect anyone would test. A five-day test that")
    emit("  ran Tuesday-to-Thursday and one that caught a weekend are not the")
    emit("  same test, and a simulator where every day is identical cannot")
    emit("  represent that at all.")
    emit("")
    emit("  WEATHER is a per-DAY shock shared across every region, which is what")
    emit("  makes it a confound rather than noise: a rainy Tuesday raises demand")
    emit("  everywhere at once, so it cannot be averaged away across regions the")
    emit("  way region-specific noise can. It is drawn from one seed so that")
    emit("  every design being compared gets the SAME weather -- otherwise a")
    emit("  design comparison is partly a comparison of the weather it drew.")
    emit("")
    emit("  TREND is %.1f%%/day. With an assignment that is unbalanced across"
         % (100 * MKT.DAILY_TREND))
    emit("  days, a trend is attributed to the treatment, and no amount of")
    emit("  within-day randomisation fixes it -- which is the argument for")
    emit("  balancing on days rather than only on users.")
    emit("")
    summary["calendar"] = D.round(4).to_dict("records")

    # ======================================================================
    emit("=" * 78)
    emit("B. CUPED -- THE CHEAPEST VARIANCE REDUCTION THERE IS")
    emit("=" * 78)
    mk = MKT.Marketplace(n_regions=N_REGIONS, seed=3)
    pre = mk.run(days=DAYS, assign_fn=MKT.assign_all(False), lift=0.0)
    pre_g = region_daily(pre, DAYS, N_REGIONS)
    pre_by_region = pre_g.groupby("region").rate.mean().to_numpy()

    # The assignment object carries its own treated set. Reconstructing it by
    # re-seeding a generator here was a real bug in the first draft: the two
    # draws did not match, so the "treated" group in the analysis was not the
    # group the simulator treated, and the estimated effect came out NEGATIVE
    # against a positive planted lift. An assignment that has to be guessed by
    # the analysis is an assignment that will eventually be guessed wrong.
    assign = MKT.assign_cluster(N_REGIONS, seed=1)
    treated_mask = np.asarray(assign.treated_regions, dtype=bool)

    mk2 = MKT.Marketplace(n_regions=N_REGIONS, seed=3)
    post = mk2.run(days=DAYS, assign_fn=assign, lift=LIFT)
    post_g = region_daily(post, DAYS, N_REGIONS)
    post_by_region = post_g.groupby("region").rate.mean().to_numpy()

    res = ADJ.cuped_effect(post_by_region, treated_mask, pre_by_region)
    gain = ADJ.cuped_gain(post_by_region, pre_by_region)
    emit("  pre/post correlation across regions : %+.4f" % gain["rho"])
    emit("  variance reduction predicted (rho^2): %.4f" % gain["predicted_reduction"])
    emit("  variance reduction realised         : %.4f" % gain["realised_reduction"])
    emit("")
    emit("  effect, unadjusted : %+.5f  (se %.5f)"
         % (res["effect_unadjusted"], res["se_unadjusted"]))
    emit("  effect, CUPED      : %+.5f  (se %.5f)" % (res["effect"], res["se"]))
    emit("  standard error cut by %.1f%%" % (100 * res["se_reduction"]))
    emit("")
    emit("  CUPED SUBTRACTS theta * (pre-period metric - its mean) from the")
    emit("  outcome, with theta chosen to minimise variance. It costs nothing, it")
    emit("  cannot bias the estimate provided the covariate is PRE-TREATMENT, and")
    emit("  the variance reduction is exactly rho^2 -- so its value is knowable in")
    emit("  advance from a correlation you already have. That is why it belongs")
    emit("  before any conversation about running longer.")
    emit("")
    emit("  THE ONE CORRECTNESS CONDITION IS NOT CHECKABLE FROM THE NUMBERS. A")
    emit("  post-treatment covariate makes CUPED silently BIASED rather than")
    emit("  merely useless, because it adjusts away part of the effect. Nothing in")
    emit("  the data reveals that, so the obligation sits with whoever chooses the")
    emit("  column -- here, the pre-period conversion rate measured before the")
    emit("  treatment existed.")
    emit("")
    emit("  Predicted and realised reduction agree to %.4f, which is the check"
         % abs(gain["predicted_reduction"] - gain["realised_reduction"]))
    emit("  worth doing: if they diverge, the covariate is not behaving the way")
    emit("  the theory assumed and the adjustment should not be trusted.")
    emit("")
    summary["cuped"] = dict(**{k: v for k, v in gain.items()}, **res)

    # ======================================================================
    emit("=" * 78)
    emit("C. SYNTHETIC CONTROL -- WHEN YOU CANNOT ADD MORE UNITS")
    emit("=" * 78)
    pre_panel = (pre_g.pivot(index="region", columns="day", values="rate")
                 .ffill(axis=1).bfill(axis=1).to_numpy())
    post_panel = (post_g.pivot(index="region", columns="day", values="rate")
                  .ffill(axis=1).bfill(axis=1).to_numpy())
    t_idx = int(np.where(treated_mask)[0][0])
    donors = np.where(~treated_mask)[0]

    sc = ADJ.synthetic_control(pre_panel[t_idx], pre_panel[donors],
                               post_panel[t_idx], post_panel[donors])
    emit("  treated region        : %d" % t_idx)
    emit("  donors in the pool    : %d, with weight > 0.01: %d"
         % (len(donors), sc["n_donors_used"]))
    emit("  pre-period fit (RMSE) : %.5f  (%.1f%% of the level)"
         % (sc["pre_rmse"], 100 * sc["pre_rmse_relative"]))
    emit("  post treated          : %.5f" % sc["post_treated"])
    emit("  post synthetic        : %.5f" % sc["post_synthetic"])
    emit("  estimated effect      : %+.5f  (true lift %.4f on conversion)"
         % (sc["effect"], LIFT))
    emit("")
    plc = ADJ.placebo_inference(pre_panel, post_panel, t_idx)
    emit("  placebo permutation   : p = %.4f over %d usable placebos"
         % (plc["p_value"], plc["n_placebos"]))
    emit("")
    emit("  SYNTHETIC CONTROL CHANGES WHAT A UNIT IS rather than adding more of")
    emit("  them, which is why it is the answer to '12 clusters is underpowered'.")
    emit("  Instead of adjusting a unit by its own past, it builds a weighted")
    emit("  combination of control units that tracks the treated unit's")
    emit("  pre-period and uses that as the counterfactual.")
    emit("")
    emit("  THE WEIGHTS ARE CONSTRAINED TO A SIMPLEX -- non-negative, summing to")
    emit("  one -- and that is not decoration. Unconstrained least squares will")
    emit("  hand a donor a weight of -3, which EXTRAPOLATES; the credibility of")
    emit("  the method rests entirely on the counterfactual being a weighted")
    emit("  average of things that actually happened.")
    emit("")
    emit("  INFERENCE IS A PERMUTATION TEST, because there is no standard error.")
    emit("  Each control unit is refitted pretending it was treated, and the")
    emit("  question is whether the real effect is unusual among those placebos.")
    emit("  Placebos whose own pre-fit is poor are EXCLUDED: a unit the donor pool")
    emit("  cannot reproduce shows a large 'effect' regardless of treatment, and")
    emit("  including them is the standard way this test acquires an artificially")
    emit("  small p-value.")
    emit("")
    if sc["pre_rmse_relative"] > 0.10:
        emit("  HONEST READING OF THIS RUN: the pre-period fit is %.1f%% of the"
             % (100 * sc["pre_rmse_relative"]))
        emit("  level, which is poor. A synthetic control with a bad pre-fit has")
        emit("  not produced a weak estimate -- it has failed to build a")
        emit("  counterfactual, and the effect above should not be quoted. The")
        emit("  cause here is a %d-day pre-period, which is far too short: the" % DAYS)
        emit("  method wants dozens of pre-periods to pin the weights.")
    else:
        emit("  The pre-period fit is %.1f%% of the level, which is tight enough"
             % (100 * sc["pre_rmse_relative"]))
        emit("  for the estimate to be worth reading.")
    emit("")
    summary["synthetic_control"] = dict(
        effect=sc["effect"], pre_rmse_relative=sc["pre_rmse_relative"],
        n_donors_used=sc["n_donors_used"], p_value=plc["p_value"],
        n_placebos=plc["n_placebos"])

    # ======================================================================
    emit("=" * 78)
    emit("D. THE MDE CURVE -- 'HOW LONG DO WE NEED TO RUN THIS'")
    emit("=" * 78)
    se_ref = res["se_unadjusted"]
    se_cuped = res["se"]
    emit("Measured standard error at %d days: %.5f unadjusted, %.5f with CUPED."
         % (DAYS, se_ref, se_cuped))
    emit("")
    curve = POW.mde_curve(se_ref, DAYS, N_REGIONS)
    curve_c = POW.mde_curve(se_cuped, DAYS, N_REGIONS)
    emit("  %6s %14s %14s" % ("days", "MDE", "MDE with CUPED"))
    for a, b in zip(curve, curve_c):
        emit("  %6d %14.5f %14.5f" % (a["days"], a["mde"], b["mde"]))
    emit("")
    true_effect = LIFT * 0.62         # lift on conversion, at the base rate
    d_plain = POW.days_needed(curve, true_effect)
    d_cuped = POW.days_needed(curve_c, true_effect)
    emit("  True effect on the conversion rate: %.5f" % true_effect)
    emit("  Days needed, unadjusted : %s" % (d_plain if d_plain else "never on this grid"))
    emit("  Days needed, with CUPED : %s" % (d_cuped if d_cuped else "never on this grid"))
    emit("")
    marg = POW.marginal_value(curve)
    emit("  What each step buys (unadjusted):")
    for m in marg[-5:]:
        emit("    %2d -> %2d days: MDE %.5f -> %.5f  (%.1f%% better)"
             % (m["from_days"], m["to_days"], m["mde_before"], m["mde_after"],
                100 * m["improvement"]))
    emit("")
    emit("  THE CURVE FLATTENS, and that is the whole reason to draw it. Standard")
    emit("  error falls as 1/sqrt(days), so DOUBLING the test improves the MDE by")
    emit("  only 30%. Past some point another week buys almost nothing, and that")
    emit("  point is where 'run it longer' stops being an option -- the honest")
    emit("  answer becomes change the design, change the metric, or do not run it.")
    emit("")
    emit("  A single MDE answers 'can we detect X in five days', which is a")
    emit("  yes/no about a decision nobody made. The curve answers 'how long',")
    emit("  which is the question that was asked.")
    emit("")
    emit("  THE SCALING ASSUMES INDEPENDENT DAYS AND IS THEREFORE OPTIMISTIC.")
    emit("  Demand is autocorrelated -- a rainy week is a rainy week -- so the")
    emit("  true standard error falls more slowly than 1/sqrt(days) and every")
    emit("  duration here is a LOWER BOUND. Stating the direction matters more")
    emit("  than correcting it: a planner who knows the estimate is optimistic")
    emit("  will pad it.")
    emit("")
    summary["mde_curve"] = dict(unadjusted=curve, cuped=curve_c,
                                true_effect=true_effect,
                                days_needed_unadjusted=d_plain,
                                days_needed_cuped=d_cuped)

    # ======================================================================
    emit("=" * 78)
    emit("E. THE PLAYBOOK, AS AN ARTIFACT")
    emit("=" * 78)
    path = POW.write_playbook(os.path.join(OUT, "PLAYBOOK.md"))
    emit("Written to out/PLAYBOOK.md -- a standalone decision tree, which is what")
    emit("the spec asked for and what the previous pass had only as report prose.")
    emit("")
    emit("  Every branch in it is a conclusion this project MEASURED, with the")
    emit("  section named so the number can be checked. That is the difference")
    emit("  between a playbook and a blog post: a reader can disagree with a")
    emit("  branch by disputing a specific number rather than a preference.")
    emit("")
    emit("  It also states what it does not cover -- sequential testing, multiple")
    emit("  comparisons, heterogeneous effects, and network effects BETWEEN")
    emit("  regions, which this simulator cannot represent because its regions")
    emit("  are independent and a real metro's are not.")
    emit("")
    summary["playbook"] = path

    with open(os.path.join(OUT, "complete_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "complete_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n-> out/complete_report.txt, out/PLAYBOOK.md")


if __name__ == "__main__":
    main()
