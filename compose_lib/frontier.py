"""Efficient frontier + Capital Market Line.

The frontier is traced by solving min-variance at a grid of target returns.
The CML is the tangent line from the risk-free rate to the tangent portfolio
(max-Sharpe point on the frontier): y = rf + (μ_T - rf)/σ_T · x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from compose_lib.optimize import (
    Constraints,
    Solution,
    min_variance,
    min_variance_at_target,
    max_sharpe as _max_sharpe,
)


MONTHS = 12


@dataclass
class Frontier:
    vols: np.ndarray            # annualized
    returns: np.ndarray         # annualized
    weights: list[pd.Series]
    tangent: Solution           # the max-Sharpe portfolio
    rf_annual: float


def trace_frontier(cov: pd.DataFrame, mu: pd.Series,
                   cons: Constraints, rf_annual: float = 0.0,
                   n_points: int = 40) -> Frontier:
    codes = list(cov.columns)
    mvp = min_variance(cov, mu, cons, rf_annual)
    mu_hi = float(mu.max())
    mu_lo = float(mu.values @ mvp.weights.reindex(codes).values)
    if mu_hi <= mu_lo:
        # Degenerate — μ is flat. Fall back to a trivial frontier at the MVP.
        return Frontier(
            vols=np.array([mvp.volatility]),
            returns=np.array([mvp.exp_return]),
            weights=[mvp.weights],
            tangent=mvp, rf_annual=rf_annual,
        )

    grid = np.linspace(mu_lo, mu_hi, n_points)
    vols, rets, ws = [], [], []
    for tgt in grid:
        try:
            s = min_variance_at_target(cov, mu, tgt, cons, rf_annual)
            vols.append(s.volatility)
            rets.append(s.exp_return)
            ws.append(s.weights)
        except Exception:
            continue

    tangent = _max_sharpe(cov, mu, cons, rf_annual, n_grid=max(60, n_points))
    return Frontier(
        vols=np.array(vols), returns=np.array(rets), weights=ws,
        tangent=tangent, rf_annual=rf_annual,
    )


def cml_points(frontier: Frontier, sigma_max: float | None = None,
               n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Capital Market Line: (σ, r) pairs from rf through the tangent. If
    `sigma_max` is None, extends 20% past the tangent portfolio's σ."""
    t = frontier.tangent
    rf = frontier.rf_annual
    if sigma_max is None:
        sigma_max = max(t.volatility * 1.2, float(frontier.vols.max()))
    x = np.linspace(0.0, sigma_max, n)
    slope = (t.exp_return - rf) / t.volatility if t.volatility > 0 else 0.0
    y = rf + slope * x
    return x, y


def asset_points(mu: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Per-asset (σ, μ, code) dataframe for plotting individual assets on
    the frontier chart."""
    sigmas = np.sqrt(np.diag(cov.values)) * np.sqrt(MONTHS)
    mus_ann = (1.0 + mu.values) ** MONTHS - 1.0
    return pd.DataFrame({"code": mu.index, "vol": sigmas, "ret": mus_ann})
