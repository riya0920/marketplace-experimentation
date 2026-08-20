"""Act 1: the problem, measured. Act 2: the fixes, priced.

Everything is scored against a KNOWN truth, which is the only reason any of it
is a result rather than an opinion.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import estimators as E  # noqa: E402
from src import market as M  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

LIFT = 0.10          # the treatment's true effect on individual conversion
DAYS = 5
N_REGIONS = 12
# couriers per region: fewer couriers = tighter market = more interference
TIGHTNESS = [("slack", 26), ("normal", 17), ("tight", 12), ("very_tight", 9)]
REPS = 6            # act 1: the bias is large, few reps resolve it
REPS_DESIGN = 12    # acts 2-3: the designs are noisier and need more


TRUTH_PAIRS = 4


def truth_for(couriers, seed=0, pairs=TRUTH_PAIRS):
    """Run the same world all-treated and all-control, and AVERAGE over pairs.

    Averaging is not fussiness. A single pair estimates the truth with roughly
    the same standard error as the naive estimator being scored against it, so
    `bias_pct` computed from one pair is mostly noise -- the first version of
    this table was non-monotone in market tightness for exactly that reason.
    Paired seeds (same world, different arm) are common random numbers, which
    removes what it can before averaging removes the rest.
    """
    effs = []
    rt = rc = None
    for i in range(pairs):
        a = M.Marketplace(n_regions=N_REGIONS, couriers=couriers, seed=seed + i)
        b = M.Marketplace(n_regions=N_REGIONS, couriers=couriers, seed=seed + i)
        rt_i = a.run(DAYS, M.assign_all(True), LIFT)
        rc_i = b.run(DAYS, M.assign_all(False), LIFT)
        effs.append(E.global_effect(rt_i, rc_i))
        if rt is None:
            rt, rc = rt_i, rc_i
    return float(np.mean(effs)), rt, rc


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    lines, summary = [], {}

    def emit(s=""):
        print(s)
        lines.append(s)

    # ------------------------------------------------------------------
    emit("=" * 78)
    emit("0. SIMULATOR VALIDATION -- this is a LAB, not a claim about reality")
    emit("=" * 78)
    v = M.validate(M.Marketplace(seed=0))
    for k, ok in v.items():
        emit("  %-34s %s" % (k, "PASS" if ok else "FAIL"))
    emit("")
    emit("Nothing here is calibrated to a real company. The simulator earns its")
    emit("keep by making the TRUTH knowable, so estimators can be scored instead")
    emit("of argued about. Its stylised facts are checked above; its conclusions")
    emit("are about ESTIMATORS, which is what transfers -- not about DoorDash.")
    summary["validation"] = v

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("1. ACT ONE -- THE NAIVE A/B TEST IS BIASED, AND THE BIAS GROWS")
    emit("=" * 78)
    emit("True effect = conversion(everyone treated) - conversion(nobody treated).")
    emit("That is the estimand that matters, because it is what shipping does.")
    emit("")
    rows = []
    for label, couriers in TIGHTNESS:
        truth, rt, rc = truth_for(couriers, seed=101)
        ests = []
        for rep in range(REPS):
            mk = M.Marketplace(n_regions=N_REGIONS, couriers=couriers, seed=200 + rep)
            rec = mk.run(DAYS, M.assign_user_level(0.5), LIFT)
            ests.append(E.naive_ab(rec))
        est = float(np.mean([e["estimate"] for e in ests]))
        rows.append(dict(
            market=label, couriers=couriers,
            mean_utilisation=E.guardrails(rc)["mean_utilisation"],
            truth=truth, naive_estimate=est, bias=est - truth,
            bias_pct=100 * (est - truth) / truth if truth else np.nan,
            reported_se=float(np.mean([e["se"] for e in ests]))))
    A = pd.DataFrame(rows).set_index("market")
    emit(A.to_string(float_format=lambda x: "%10.5f" % x))
    emit("")
    emit("THE MONEY FIGURE IS THE bias_pct COLUMN AGAINST MARKET TIGHTNESS.")
    for r in A.itertuples():
        emit("  %-11s utilisation %5.1f%%   truth %+.5f   naive %+.5f   OVERSTATED by %5.1f%%"
             % (r.Index, 100 * r.mean_utilisation, r.truth, r.naive_estimate, r.bias_pct))
    emit("")
    emit("Mechanism, stated as a chain because that is how you explain it to a PM:")
    emit("  treatment raises conversion -> more orders -> more couriers busy ->")
    emit("  utilisation up -> quoted delivery time up -> conversion falls for")
    emit("  EVERYONE in the region, including control.")
    emit("")
    emit("So the control group is not a group of unaffected bystanders. It is a")
    emit("group the treatment actively made worse off, by taking the couriers it")
    emit("was competing for. Measuring (treated - control) therefore counts the")
    emit("treatment's benefit AND the damage it did to its own baseline. SUTVA --")
    emit("the assumption that one unit's treatment does not affect another unit's")
    emit("outcome -- is violated by construction, and in a marketplace it is")
    emit("ALWAYS violated; the only question is by how much.")
    emit("")
    emit("Note the reported_se column: it is small and it is CORRECT for how")
    emit("randomisation happened. The bias lives in the point estimate, not in the")
    emit("variance, so no amount of extra sample fixes it -- more users make the")
    emit("confidence interval tighter around the wrong number.")
    summary["act1_naive_bias"] = A.reset_index().round(6).to_dict("records")

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("2. ACT TWO -- SWITCHBACK, AND WHAT GRANULARITY COSTS")
    emit("=" * 78)
    truth, rt, rc = truth_for(12, seed=101)
    emit("Market: tight (12 couriers/region). True effect %+.5f" % truth)
    emit("")
    rows = []
    for block in (30, 120, 480, 1440):
        # simulate ONCE per (block, rep); burn-in is an ANALYSIS choice applied
        # to the same data, so re-simulating for it would waste half the budget
        # and, worse, would confound the burn-in comparison with simulation noise
        recs = []
        for rep in range(REPS_DESIGN):
            mk = M.Marketplace(n_regions=N_REGIONS, couriers=12, seed=300 + rep)
            recs.append(mk.run(DAYS, M.assign_switchback(block, seed=50 + rep), LIFT))
        for burn in (0, 20):
            ests, ses, nb = [], [], 0
            for rep, rec in enumerate(recs):
                r = E.switchback(rec, block, burn_in_minutes=burn, seed=rep)
                if np.isfinite(r["estimate"]):
                    ests.append(r["estimate"])
                    ses.append(r["se"])
                    nb = r["n_blocks"]
            sd = float(np.std(ests, ddof=1))
            rows.append(dict(block_minutes=block, burn_in=burn, n_blocks=nb,
                             estimate=float(np.mean(ests)),
                             bias=float(np.mean(ests)) - truth,
                             bias_pct=100 * (float(np.mean(ests)) - truth) / truth,
                             mc_se_of_bias=sd / np.sqrt(len(ests)),
                             sd_across_reps=sd,
                             mean_bootstrap_se=float(np.mean(ses))))
    S = pd.DataFrame(rows).set_index(["block_minutes", "burn_in"])
    emit(S.to_string(float_format=lambda x: "%11.5f" % x))
    emit("")
    emit("READ THE mc_se_of_bias COLUMN FIRST. It is the Monte Carlo standard")
    emit("error of the bias estimate itself -- how precisely %d replications pin"
         % REPS_DESIGN)
    emit("down each design's bias. A bias smaller than about twice its own MC SE")
    emit("is not distinguishable from zero here, and saying which rows clear that")
    emit("bar is the difference between a measurement and a decoration:")
    for (blk, burn), r in S.iterrows():
        verdict = ("RESOLVED" if abs(r.bias) > 2 * r.mc_se_of_bias
                   else "not distinguishable from zero")
        emit("  block %4d min, burn-in %2d:  bias %+.5f +/- %.5f   %s"
             % (blk, burn, r.bias, r.mc_se_of_bias, verdict))
    emit("")
    emit("THE HEADLINE, AND IT IS NOT THE ONE I EXPECTED TO WRITE: at this")
    emit("replication count NO switchback granularity shows a bias distinguishable")
    emit("from zero, including the 30-minute blocks whose carryover window is")
    emit("longer than the block itself. The naive A/B's bias, by contrast, is many")
    emit("times its own MC error. So the mitigation works, and the interesting")
    emit("question moves from BIAS to VARIANCE.")
    emit("")
    emit("THE VARIANCE TRADE, which is where the granularity decision actually is:")
    for (blk, burn), r in S.iterrows():
        if burn == 0:
            emit("  %4d-minute blocks: %3d blocks, sd across reps %.5f"
                 % (blk, r.n_blocks, r.sd_across_reps))
    emit("")
    emit("  Read that column and the naive intuition breaks. MORE BLOCKS DID NOT")
    emit("  BUY PRECISION -- the 1,440-minute design has the fewest blocks and the")
    emit("  LOWEST variance, and the 120-minute design has four times as many")
    emit("  blocks and the highest.")
    emit("")
    emit("  The reason is that switchback variance is driven by heterogeneity")
    emit("  BETWEEN blocks, not by their count. A 30-minute block at 03:00 and one")
    emit("  at 18:30 are different worlds -- different demand, different congestion")
    emit("  -- so an unadjusted difference in means across arms is dominated by")
    emit("  which arm happened to draw the dinner rush. A daily block averages the")
    emit("  whole demand curve and is therefore stable.")
    emit("")
    emit("  THE PRACTITIONER CONSEQUENCE: short blocks are only worth their extra")
    emit("  count if the analysis CONTROLS for time-of-day -- paired adjacent")
    emit("  blocks, or block-level covariate adjustment on the time index. Running")
    emit("  fine-grained switchbacks and then taking a raw difference in means")
    emit("  throws away the precision the design was chosen for. That adjustment is")
    emit("  not implemented here; it is the first thing I would add.")
    emit("")
    emit("  The low variance of the daily design is also partly an artifact worth")
    emit("  naming: this simulator's days are statistically identical, so daily")
    emit("  blocks look more reliable than a real week of blocks would, where")
    emit("  day-of-week effects and weather make each day genuinely different.")
    emit("")
    emit("  BURN-IN moves every row toward truth -- 30-min bias %+.5f -> %+.5f,"
         % (S.loc[(30, 0), "bias"], S.loc[(30, 20), "bias"]))
    emit("  120-min %+.5f -> %+.5f. The direction is exactly what carryover"
         % (S.loc[(120, 0), "bias"], S.loc[(120, 20), "bias"]))
    emit("  predicts and it is consistent across granularities, but none of the")
    emit("  individual moves clears its own MC error, so the honest statement is")
    emit("  'directionally consistent with carryover, not resolved at 12 reps'")
    emit("  rather than 'burn-in removes X of bias'.")
    emit("")
    emit("  THE FRAMEWORK: choose the block just longer than the carryover window")
    emit("  (here ~34 minutes of service time), add burn-in of about that window,")
    emit("  and then ADJUST FOR TIME-OF-DAY or the extra blocks are wasted. For a")
    emit("  COURIER INCENTIVE test the carryover window is not the service time at")
    emit("  all -- it is how long a courier's repositioning decision persists,")
    emit("  which is hours. There, 15-minute blocks would be almost pure")
    emit("  contamination and 24-hour blocks the honest floor.")
    summary["act2_switchback"] = S.reset_index().round(6).to_dict("records")

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("3. ACT TWO CONTINUED -- CLUSTER / GEO RANDOMISATION")
    emit("=" * 78)
    rows = []
    for n_reg in (12, 24, 40):
        ests, ses, naive_ses = [], [], []
        for rep in range(REPS_DESIGN):
            mk = M.Marketplace(n_regions=n_reg, couriers=12, seed=400 + rep)
            af = M.assign_cluster(n_reg, seed=70 + rep)
            rec = mk.run(DAYS, af, LIFT)
            r = E.cluster_randomised(rec, af.treated_regions)
            if np.isfinite(r["estimate"]):
                ests.append(r["estimate"])
                ses.append(r["se"])
                naive_ses.append(E.naive_se_for_cluster_design(rec))
        rows.append(dict(n_clusters=n_reg,
                         estimate=float(np.mean(ests)),
                         bias=float(np.mean(ests)) - truth,
                         bias_pct=100 * (float(np.mean(ests)) - truth) / truth,
                         mc_se_of_bias=float(np.std(ests, ddof=1)) / np.sqrt(len(ests)),
                         sd_across_reps=float(np.std(ests, ddof=1)),
                         cluster_robust_se=float(np.mean(ses)),
                         naive_se_if_misanalysed=float(np.mean(naive_ses)),
                         se_understatement_x=float(np.mean(ses) / np.mean(naive_ses))))
    C = pd.DataFrame(rows).set_index("n_clusters")
    emit(C.to_string(float_format=lambda x: "%12.5f" % x))
    emit("")
    emit("Bias by cluster count, against its own Monte Carlo error:")
    for n_reg, r in C.iterrows():
        verdict = ("RESOLVED" if abs(r.bias) > 2 * r.mc_se_of_bias
                   else "not distinguishable from zero")
        emit("  %2d clusters:  bias %+.5f +/- %.5f   %s"
             % (n_reg, r.bias, r.mc_se_of_bias, verdict))
    emit("")
    emit("THE INFERENCE ERROR, SIZED. `naive_se_if_misanalysed` is what you get")
    emit("by analysing a geo test as though customers were randomised. The")
    emit("correct cluster-robust SE is %.0fx to %.0fx larger."
         % (C.se_understatement_x.min(), C.se_understatement_x.max()))
    emit("The effective sample size is the number of REGIONS, not the number of")
    emit("customers -- customers inside a region share a courier pool and are")
    emit("nothing like independent. Reporting the naive SE on a geo test")
    emit("manufactures significance out of nothing, and it is the single most")
    emit("common way these tests are misread.")
    emit("")
    emit("With few clusters even the robust SE is optimistic, so the t reference")
    emit("distribution with df = clusters - 2 is used rather than the normal.")
    emit("At 12 clusters that is 10 degrees of freedom, and the honest answer to")
    emit("'we have 12 geos' is often that the effect is not detectable at this")
    emit("size -- the options being pair-matched assignment on pre-period")
    emit("covariates, covariate adjustment, a longer run, or synthetic control.")

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("4. THE COMPARISON TABLE -- BIAS AND VARIANCE, PRICED")
    emit("=" * 78)
    best_sb = S.reset_index()
    best_sb = best_sb.loc[best_sb.bias.abs().idxmin()]
    naive_row = A.loc["tight"]
    clus = C.loc[12]
    comp = pd.DataFrame([
        dict(design="naive user-level A/B", estimate=naive_row.naive_estimate,
             bias=naive_row.bias, bias_pct=naive_row.bias_pct,
             mc_se_of_bias=np.nan,
             sd_across_reps=np.nan, reported_se=naive_row.reported_se),
        dict(design="switchback %dmin burn-in %d" % (best_sb.block_minutes,
                                                     best_sb.burn_in),
             estimate=best_sb.estimate,
             bias=best_sb.bias, bias_pct=best_sb.bias_pct,
             mc_se_of_bias=best_sb.mc_se_of_bias,
             sd_across_reps=best_sb.sd_across_reps,
             reported_se=best_sb.mean_bootstrap_se),
        dict(design="cluster / geo (12)", estimate=clus.estimate,
             bias=clus.bias, bias_pct=clus.bias_pct,
             mc_se_of_bias=clus.mc_se_of_bias,
             sd_across_reps=clus.sd_across_reps,
             reported_se=clus.cluster_robust_se),
    ]).set_index("design")
    emit("True effect %+.5f  (tight market, 12 couriers/region)" % truth)
    emit(comp.to_string(float_format=lambda x: "%11.5f" % x))
    emit("")
    emit("")
    emit("The naive design's bias is the only one in this table that is resolved;")
    emit("both mitigations remove it. What they charge for that is NOT uniform, and")
    emit("the reported_se column says so:")
    emit("")
    for d, r in comp.iterrows():
        emit("  %-30s bias %+8.5f   reported SE %.5f" % (d, r.bias, r.reported_se))
    emit("")
    emit("  The GEO design pays the expected price: %.1fx the naive SE, because its"
         % (comp.iloc[2].reported_se / comp.iloc[0].reported_se))
    emit("  effective sample size is 12 regions rather than ~100,000 customers.")
    emit("  That is the textbook bias-variance trade and it shows up exactly where")
    emit("  it should.")
    emit("")
    emit("  The SWITCHBACK does NOT, and I am flagging that rather than claiming")
    emit("  it: at %.5f its SE is no worse than the naive design's %.5f. Before"
         % (comp.iloc[1].reported_se, comp.iloc[0].reported_se))
    emit("  reading that as free lunch, note WHICH switchback won -- the daily")
    emit("  blocks -- and that this simulator's days are statistically identical.")
    emit("  Daily blocks are therefore unusually stable here in a way a real week")
    emit("  of blocks, with day-of-week and weather effects, would not be. The")
    emit("  finer granularities in section 2 DO carry 2-3x the variance, which is")
    emit("  the price a real deployment would pay. Treating the daily row as")
    emit("  evidence that switchbacks are free would be reading the lab as the")
    emit("  world.")
    emit("")
    emit("  The general shape survives that caveat: the choice is between a tight")
    emit("  interval around the wrong number and a wider one around the right")
    emit("  number, and in a marketplace only the second is worth shipping on.")
    emit("")
    emit("WHEN EACH ONE WINS:")
    emit("  naive A/B   -- only when the treatment cannot touch a shared resource.")
    emit("                 In a marketplace that is a short list: UI copy, an")
    emit("                 email, anything that does not change order volume. The")
    emit("                 moment it changes demand, this design is measuring its")
    emit("                 own shadow.")
    emit("  switchback  -- treatment affects the shared resource, effects decay")
    emit("                 within a known window, and the market is one pool.")
    emit("                 Cheapest unbiased option when it applies.")
    emit("  cluster/geo -- effects persist longer than any workable block")
    emit("                 (incentives, supply-side changes, anything with memory)")
    emit("                 or the market genuinely partitions geographically. Pay")
    emit("                 for it in power, and do not pretend otherwise in the SE.")
    summary["comparison"] = comp.reset_index().round(6).to_dict("records")

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("5. TWO-SIDED GUARDRAILS -- 'DEMAND UP, SHIP IT?'")
    emit("=" * 78)
    g_t, g_c = E.guardrails(rt), E.guardrails(rc)
    emit("%-24s %12s %12s %10s" % ("", "control", "treated", "delta"))
    for k in ("mean_utilisation", "p95_utilisation", "mean_eta", "p95_eta", "orders"):
        emit("%-24s %12.4f %12.4f %+10.4f" % (k, g_c[k], g_t[k], g_t[k] - g_c[k]))
    emit("")
    emit("The demand-side metric is up: %d orders vs %d, %+.1f%%."
         % (g_t["orders"], g_c["orders"],
            100 * (g_t["orders"] / g_c["orders"] - 1)))
    emit("Courier utilisation is up %+.1f points and p95 delivery time is up %+.1f"
         % (100 * (g_t["mean_utilisation"] - g_c["mean_utilisation"]),
            g_t["p95_eta"] - g_c["p95_eta"]))
    emit("minutes. A demand-side-only readout would call this a clean win.")
    emit("")
    emit("SHIP? Not on this evidence alone. The supply side is absorbing the whole")
    emit("gain, and the two things this simulator CANNOT see are exactly the two")
    emit("that decide it: whether couriers churn when utilisation sits at this")
    emit("level for weeks, and whether customers who waited longer come back. Both")
    emit("are long-run quantities and neither is in a %d-day experiment. The" % DAYS)
    emit("honest recommendation is a staged rollout with courier-supply and")
    emit("repeat-rate guardrails monitored past the experiment window, not a")
    emit("ship decision from a conversion delta.")
    summary["guardrails"] = dict(control=g_c, treated=g_t)

    emit("")
    emit("(%.0fs)" % (time.time() - t0))
    with open(os.path.join(OUT, "experiments_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "experiments_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n-> out/experiments_report.txt")


if __name__ == "__main__":
    main()
