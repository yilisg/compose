"""Covariance estimators.

Three families offered in the app:

1. **Plain sample**. Fine when T >> N, bad otherwise. Included mostly as a
   reference point / sanity check.
2. **Shrinkage** (Ledoit-Wolf or OAS). Shrinks the sample Σ toward a
   structured target (scaled identity by default). Ledoit-Wolf derives the
   shrinkage intensity analytically and is the compose default. OAS
   (Chen et al. 2010) is a close cousin that tends to be slightly better
   when returns are approximately Gaussian and N is small.
3. **Stress-blended**. Splits history into "normal" and "stress" regimes
   (SPX in drawdown ≥ `dd_threshold`, default 10%) and blends the two
   estimated covariances with a user-specified stress weight p:
       Σ_blend = (1 - p) · Σ_normal + p · Σ_stress
   A p of 0 recovers full-sample LW; a p of 1 assumes we are *in* a crisis
   today and uses the stress block only. Useful for pressure-testing
   whether a portfolio's risk estimate is robust to correlation spikes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

#: Number of trading months per year for annualization.
MONTHS = 12


@dataclass
class CovResult:
    cov: pd.DataFrame         # monthly covariance
    cov_annual: pd.DataFrame  # cov * 12
    method: str
    shrinkage: float | None   # shrinkage intensity if applicable
    n_obs: int
    mask: pd.Series | None = None  # bool series of rows used (stress mode)
    meta: dict | None = None       # estimator-specific extras (regime weights etc.)


def sample_cov(rets: pd.DataFrame) -> CovResult:
    cov = rets.cov()
    return CovResult(
        cov=cov, cov_annual=cov * MONTHS,
        method="sample", shrinkage=None, n_obs=len(rets),
    )


def ledoit_wolf_cov(rets: pd.DataFrame) -> CovResult:
    est = LedoitWolf().fit(rets.values)
    cov = pd.DataFrame(est.covariance_, index=rets.columns, columns=rets.columns)
    return CovResult(
        cov=cov, cov_annual=cov * MONTHS,
        method="ledoit_wolf", shrinkage=float(est.shrinkage_), n_obs=len(rets),
    )


def oas_cov(rets: pd.DataFrame) -> CovResult:
    est = OAS().fit(rets.values)
    cov = pd.DataFrame(est.covariance_, index=rets.columns, columns=rets.columns)
    return CovResult(
        cov=cov, cov_annual=cov * MONTHS,
        method="oas", shrinkage=float(est.shrinkage_), n_obs=len(rets),
    )


def _drawdown_mask(eq_returns: pd.Series, threshold: float) -> pd.Series:
    """Return a bool Series (True = 'stress') marking months where the
    cumulative equity drawdown is at or worse than `-threshold`."""
    cum = (1.0 + eq_returns).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return dd <= -abs(threshold)


def stress_blended_cov(
    rets: pd.DataFrame,
    equity_col: str,
    dd_threshold: float = 0.10,
    stress_weight: float = 0.30,
    base: str = "ledoit_wolf",
) -> CovResult:
    """Split `rets` into stress / non-stress rows via SPX drawdown and blend
    shrunk covariances. `stress_weight` in [0, 1] is p in the docstring
    formula. Falls back to full-sample LW when either regime has < 24 obs."""
    if equity_col not in rets.columns:
        raise ValueError(f"stress_blended_cov needs {equity_col!r} in returns")
    mask = _drawdown_mask(rets[equity_col], dd_threshold)

    rets_stress = rets.loc[mask]
    rets_normal = rets.loc[~mask]

    if len(rets_stress) < 24 or len(rets_normal) < 24:
        res = ledoit_wolf_cov(rets)
        res.method = f"stress_blended (fallback→LW; stress n={len(rets_stress)})"
        res.mask = mask
        return res

    base_fn = ledoit_wolf_cov if base == "ledoit_wolf" else oas_cov
    res_n = base_fn(rets_normal)
    res_s = base_fn(rets_stress)
    p = float(np.clip(stress_weight, 0.0, 1.0))
    blend = (1.0 - p) * res_n.cov.values + p * res_s.cov.values
    cov = pd.DataFrame(blend, index=rets.columns, columns=rets.columns)
    return CovResult(
        cov=cov, cov_annual=cov * MONTHS,
        method=f"stress_blended (p={p:.2f}, dd≥{int(dd_threshold*100)}%)",
        shrinkage=None, n_obs=len(rets), mask=mask,
    )


def ewma_cov(
    rets: pd.DataFrame,
    half_life_months: float = 12.0,
) -> CovResult:
    """Exponentially-weighted moving-average covariance with `half_life_months`.

    Weight on observation at lag k (0 = most recent) is `α^k` with
    `α = 0.5 ** (1 / half_life)`. The estimator is the demeaned weighted
    cross-product divided by the sum of weights; not corrected for
    bias because for monthly portfolios the bias is small. Useful when
    the user wants the recent regime to dominate without going all the
    way to a hard regime split."""
    half_life = max(1.0, float(half_life_months))
    alpha = 0.5 ** (1.0 / half_life)
    n = len(rets)
    # Most-recent observation has weight 1; older observations decay by α.
    # k = 0 is most recent.
    k = np.arange(n)[::-1]  # n-1 ... 0 reversed -> 0 = oldest, n-1 = newest
    # We want most-recent weight = 1, oldest = α^(n-1)
    weights = alpha ** (k)  # shape (n,)
    weights = weights / weights.sum()
    X = rets.values
    mu_w = (weights[:, None] * X).sum(axis=0)
    Xc = X - mu_w
    cov_mat = (Xc * weights[:, None]).T @ Xc
    cov = pd.DataFrame(cov_mat, index=rets.columns, columns=rets.columns)
    return CovResult(
        cov=cov, cov_annual=cov * MONTHS,
        method=f"ewma (half-life={half_life_months:.0f}m)",
        shrinkage=None, n_obs=n,
    )


def regime_blended_cov(
    rets: pd.DataFrame,
    regime: pd.Series,
    today_probs: dict[str, float] | None = None,
    base: str = "ledoit_wolf",
    min_obs: int = 24,
) -> CovResult:
    """Compute Σ per regime label (string-indexed), blend by today's
    regime probabilities (or equal weight if none provided).

    `regime` is a Series indexed by month with string labels (e.g. from
    `regime_label.label_from_z` applied per month, or simply
    'Stress'/'Normal' from the SPX-drawdown bucket).

    Falls back to full-sample LW if any regime has fewer than `min_obs`
    months."""
    base_fn = ledoit_wolf_cov if base == "ledoit_wolf" else oas_cov
    aligned = regime.reindex(rets.index).dropna()
    rets = rets.loc[aligned.index]
    counts = aligned.value_counts()
    if (counts < min_obs).any():
        res = ledoit_wolf_cov(rets)
        res.method = "regime_blended (fallback→LW)"
        return res
    cov_per: dict[str, pd.DataFrame] = {}
    for label in counts.index:
        sub = rets.loc[aligned == label]
        cov_per[label] = base_fn(sub).cov

    if today_probs is None:
        today_probs = {label: 1.0 / len(counts) for label in counts.index}
    # Fill any missing labels with zero weight; renormalize.
    pw = {label: float(today_probs.get(label, 0.0)) for label in counts.index}
    s = sum(pw.values())
    if s <= 0:
        pw = {label: 1.0 / len(counts) for label in counts.index}
    else:
        pw = {label: v / s for label, v in pw.items()}

    blend = sum(pw[label] * cov_per[label].values for label in counts.index)
    cov = pd.DataFrame(blend, index=rets.columns, columns=rets.columns)
    return CovResult(
        cov=cov, cov_annual=cov * MONTHS,
        method=f"regime_blended ({len(counts)} regimes)",
        shrinkage=None, n_obs=len(rets),
        meta={"regime_counts": counts.to_dict(), "weights": pw},
    )


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    s = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(s, s)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def condition_number(cov: pd.DataFrame) -> float:
    w = np.linalg.eigvalsh(cov.values)
    w = w[w > 0]
    if len(w) == 0:
        return float("inf")
    return float(w.max() / w.min())
