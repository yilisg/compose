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
