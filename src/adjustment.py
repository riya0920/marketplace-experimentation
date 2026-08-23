"""CUPED and synthetic control -- the two covariate methods the report named.

TWO GAPS
--------
  "Covariate adjustment is only via pair matching -- no regression adjustment on
   pre-period outcomes (CUPED), which is the cheaper and usually larger win."
  "No synthetic control, which is the remaining answer to '12 clusters is
   underpowered' and the only one not now implemented."

THEY ARE THE SAME IDEA AT TWO STRENGTHS
---------------------------------------
Both use PRE-PERIOD data to remove variance the treatment cannot have caused.

  CUPED subtracts theta * (pre-period metric - its mean) from the outcome, with
  theta chosen to minimise variance. It is a regression adjustment, it costs
  nothing, it cannot bias the estimate if the covariate is pre-treatment, and the
  variance reduction is exactly rho^2 -- so its value is knowable in advance from
  a correlation you already have.

  SYNTHETIC CONTROL goes further: instead of adjusting each unit by its own past,
  it builds a WEIGHTED COMBINATION of control units that tracks the treated
  unit's pre-period, and uses that combination as the counterfactual. It can work
  with a single treated unit, which is why it is the answer to "12 clusters is
  underpowered" -- it changes what a unit IS rather than adding more of them.

WHY CUPED IS FIRST AND SYNTHETIC CONTROL IS SECOND
--------------------------------------------------
CUPED is nearly free and its benefit is predictable. Synthetic control requires a
long pre-period, a donor pool that plausibly spans the treated unit, and
inference by placebo permutation rather than a standard error -- and it fails
quietly when the pre-period fit is poor, which is the failure mode that gets it
published anyway. So the pre-period fit quality is reported next to every
estimate here, because a synthetic control with a bad pre-fit is not a weak
result, it is not a result.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# CUPED
# --------------------------------------------------------------------------
def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """The variance-minimising coefficient: cov(y, x) / var(x).

    Note this is a REGRESSION of the outcome on the covariate, not a correlation.
    Using the correlation instead is a common slip and it under-adjusts whenever
    the two have different scales.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = float(np.var(x, ddof=1))
    return float(np.cov(y, x, ddof=1)[0, 1] / v) if v > 0 else 0.0


def cuped_adjust(y: np.ndarray, x: np.ndarray,
                 theta: float | None = None) -> np.ndarray:
    """y_adjusted = y - theta * (x - mean(x)).

    THE COVARIATE MUST BE PRE-TREATMENT. That is the entire correctness
    condition, and it is not checkable from the numbers -- a post-treatment
    covariate makes CUPED silently biased rather than merely useless, because it
    adjusts away part of the effect. This function cannot verify it, so the
    caller carries the obligation and the report states which column was used.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    t = cuped_theta(y, x) if theta is None else theta
    return y - t * (x - x.mean())


def cuped_gain(y: np.ndarray, x: np.ndarray) -> dict:
    """Variance before and after, plus the rho^2 the theory predicts.

    Reporting both is the point: if the realised reduction does not match rho^2
    the covariate is not doing what was assumed, and that is worth seeing before
    the result is trusted.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    rho = float(np.corrcoef(y, x)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    adj = cuped_adjust(y, x)
    v0, v1 = float(np.var(y, ddof=1)), float(np.var(adj, ddof=1))
    return dict(rho=rho, predicted_reduction=rho ** 2,
                realised_reduction=1 - v1 / v0 if v0 > 0 else 0.0,
                var_before=v0, var_after=v1)


def cuped_effect(y: np.ndarray, treated: np.ndarray, x: np.ndarray) -> dict:
    """Difference in means on the CUPED-adjusted outcome, with its standard error."""
    adj = cuped_adjust(y, x)
    t, c = adj[treated.astype(bool)], adj[~treated.astype(bool)]
    raw_t, raw_c = np.asarray(y, float)[treated.astype(bool)], np.asarray(y, float)[~treated.astype(bool)]
    se = float(np.sqrt(np.var(t, ddof=1) / len(t) + np.var(c, ddof=1) / len(c)))
    se_raw = float(np.sqrt(np.var(raw_t, ddof=1) / len(raw_t)
                           + np.var(raw_c, ddof=1) / len(raw_c)))
    return dict(effect=float(t.mean() - c.mean()), se=se,
                effect_unadjusted=float(raw_t.mean() - raw_c.mean()),
                se_unadjusted=se_raw,
                se_reduction=1 - se / se_raw if se_raw > 0 else 0.0)


# --------------------------------------------------------------------------
# synthetic control
# --------------------------------------------------------------------------
def _simplex_fit(target: np.ndarray, donors: np.ndarray, iters: int = 4000,
                 seed: int = 0) -> np.ndarray:
    """Non-negative weights summing to one, by projected gradient descent.

    The simplex constraint is not decoration. Unconstrained least squares will
    happily hand a donor a weight of -3, which extrapolates rather than
    interpolates -- and the whole credibility of a synthetic control rests on the
    counterfactual being a WEIGHTED AVERAGE of things that actually happened.
    """
    n = donors.shape[0]
    w = np.full(n, 1.0 / n)
    lr = 0.5 / max(np.mean(donors ** 2), 1e-9)
    for _ in range(iters):
        resid = donors.T @ w - target
        grad = donors @ resid
        w = w - lr * grad / max(len(target), 1)
        w = np.clip(w, 0.0, None)
        ssum = w.sum()
        w = w / ssum if ssum > 0 else np.full(n, 1.0 / n)
    return w


def synthetic_control(pre_treated: np.ndarray, pre_donors: np.ndarray,
                      post_treated: np.ndarray, post_donors: np.ndarray) -> dict:
    """Build a weighted control that tracks the treated unit's pre-period.

    Returns the effect AND the pre-period fit, because the second decides whether
    the first means anything. A synthetic control whose pre-fit is poor has not
    produced a weak estimate; it has failed to build a counterfactual, and that
    distinction is what separates the method from curve-fitting.
    """
    w = _simplex_fit(pre_treated, pre_donors)
    pre_hat = pre_donors.T @ w
    post_hat = post_donors.T @ w
    pre_rmse = float(np.sqrt(np.mean((pre_treated - pre_hat) ** 2)))
    scale = float(np.mean(np.abs(pre_treated))) or 1.0
    return dict(weights=w, effect=float(np.mean(post_treated - post_hat)),
                pre_rmse=pre_rmse, pre_rmse_relative=pre_rmse / scale,
                n_donors_used=int((w > 0.01).sum()),
                post_treated=float(np.mean(post_treated)),
                post_synthetic=float(np.mean(post_hat)))


def placebo_inference(pre: np.ndarray, post: np.ndarray,
                      treated_idx: int) -> dict:
    """Run the same fit pretending each control unit was treated.

    Synthetic control has no standard error, so inference is a permutation test:
    if the treated unit's effect is not unusual among the placebos, there is
    nothing to report. Placebos with a poor PRE-fit are excluded, because a unit
    the donor pool cannot reproduce will show a large 'effect' regardless of
    treatment -- including them is the standard way this test gets an
    artificially small p-value.
    """
    n_units = pre.shape[0]
    real = None
    placebos = []
    for i in range(n_units):
        donors = np.delete(np.arange(n_units), i)
        out = synthetic_control(pre[i], pre[donors], post[i], post[donors])
        if i == treated_idx:
            real = out
        elif out["pre_rmse_relative"] < 0.25:
            placebos.append(abs(out["effect"]))
    if real is None or not placebos:
        return dict(effect=None, p_value=float("nan"), n_placebos=len(placebos))
    bigger = sum(1 for p in placebos if p >= abs(real["effect"]))
    return dict(effect=real["effect"], pre_rmse_relative=real["pre_rmse_relative"],
                p_value=(bigger + 1) / (len(placebos) + 1),
                n_placebos=len(placebos))
