"""Walk-forward backtest with monthly rebalancing and pro-rated assets.

Each rebalance date `t`:
  1. Estimator window = last `lookback_months` of returns ending at t.
  2. Active asset set = columns with at least `min_obs_per_window` non-NaN
     observations in that window.
  3. Subset Σ and μ to the active set; subset constraints via
     `subset_constraints`.
  4. Run the chosen optimizer; record weights, vol, μ, active set.
  5. Roll forward to t+1 and realize the (active-only) return for the next
     month using the previous month's solved weights.

The unit of internal math stays monthly; annualized stats are computed
once at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from compose_lib.covariance import (
    ewma_cov,
    ledoit_wolf_cov,
    oas_cov,
    stress_blended_cov,
)
from compose_lib.expected_returns import (
    black_litterman_equilibrium,
    historical_mean,
    jorion_shrinkage,
)
from compose_lib.optimize import (
    Constraints,
    Solution,
    hrp,
    max_sharpe,
    min_variance,
    risk_parity,
    subset_constraints,
)


MONTHS = 12


@dataclass
class WalkForwardWindow:
    rebal_date: pd.Timestamp
    active_assets: list[str]
    weights: pd.Series
    realized_return: float
    monthly_vol_estimate: float
    method: str = ""


@dataclass
class WalkForwardResult:
    method: str
    cum_wealth: pd.Series           # monthly compounded wealth, indexed by date
    monthly_returns: pd.Series      # realized portfolio returns, monthly
    weights_history: pd.DataFrame   # rows=rebal dates, cols=asset codes
    windows: list[WalkForwardWindow] = field(default_factory=list)

    @property
    def sharpe(self) -> float:
        m = self.monthly_returns.mean() * MONTHS
        v = self.monthly_returns.std() * np.sqrt(MONTHS)
        return float(m / v) if v > 0 else float("nan")

    @property
    def ann_return(self) -> float:
        if self.monthly_returns.empty:
            return float("nan")
        # Geometric: total wealth ^ (12/n) - 1
        end = float(self.cum_wealth.iloc[-1])
        n = len(self.monthly_returns)
        return float(end ** (MONTHS / n) - 1.0) if n > 0 else float("nan")

    @property
    def ann_vol(self) -> float:
        return float(self.monthly_returns.std() * np.sqrt(MONTHS))

    @property
    def max_drawdown(self) -> float:
        peak = self.cum_wealth.cummax()
        dd = self.cum_wealth / peak - 1.0
        return float(dd.min())

    @property
    def turnover(self) -> float:
        """Average L1 turnover per rebalance (sum of |Δw| / 2)."""
        w = self.weights_history.fillna(0.0)
        if len(w) < 2:
            return 0.0
        diffs = w.diff().abs().sum(axis=1) / 2.0
        return float(diffs.iloc[1:].mean())


# ---------------------------------------------------------------------------
# Σ / μ factories used during walk-forward
# ---------------------------------------------------------------------------


def _make_cov(method: str, rets_window: pd.DataFrame, **kwargs):
    if method == "ledoit_wolf":
        return ledoit_wolf_cov(rets_window)
    if method == "oas":
        return oas_cov(rets_window)
    if method == "ewma":
        return ewma_cov(rets_window, half_life_months=kwargs.get("half_life", 12.0))
    if method == "stress_blended":
        eq_col = kwargs.get("equity_col", "us_eq")
        if eq_col not in rets_window.columns:
            return ledoit_wolf_cov(rets_window)
        return stress_blended_cov(
            rets_window, equity_col=eq_col,
            dd_threshold=kwargs.get("dd_threshold", 0.10),
            stress_weight=kwargs.get("stress_weight", 0.30),
        )
    return ledoit_wolf_cov(rets_window)


def _make_mu(method: str, rets_window: pd.DataFrame, cov: pd.DataFrame,
             **kwargs):
    if method == "historical":
        return historical_mean(rets_window).mu
    if method == "jorion":
        return jorion_shrinkage(rets_window).mu
    if method == "black_litterman":
        bm = kwargs.get("benchmark", {})
        delta = kwargs.get("bl_delta", 3.0)
        rf_m = kwargs.get("rf_monthly", 0.0)
        return black_litterman_equilibrium(
            cov, bm, risk_aversion=delta, risk_free_monthly=rf_m,
        ).mu
    return historical_mean(rets_window).mu


# ---------------------------------------------------------------------------
# Optimizer registry for the backtest
# ---------------------------------------------------------------------------


def _solve_for_method(
    method: str,
    cov: pd.DataFrame,
    mu: pd.Series,
    cons: Constraints,
    rf_annual: float,
) -> Solution:
    if method == "max_sharpe":
        return max_sharpe(cov, mu, cons, rf_annual=rf_annual)
    if method == "min_var":
        return min_variance(cov, mu, cons, rf_annual=rf_annual)
    if method == "risk_parity":
        return risk_parity(cov, mu, cons, rf_annual=rf_annual)
    if method == "hrp":
        return hrp(cov, mu, cons, rf_annual=rf_annual)
    raise ValueError(f"unsupported optimizer: {method}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def walk_forward(
    rets: pd.DataFrame,
    cons: Constraints,
    cov_method: str = "ledoit_wolf",
    mu_method: str = "historical",
    optimizer: str = "max_sharpe",
    lookback_months: int = 60,
    min_obs_per_window: int = 24,
    rf_annual: float = 0.03,
    cov_kwargs: dict | None = None,
    mu_kwargs: dict | None = None,
) -> WalkForwardResult:
    """Run a walk-forward backtest with monthly rebal.

    Pro-rating: at each window, only assets with `>= min_obs_per_window`
    valid observations in the look-back are included in the optimization;
    the constraints are subset accordingly. The realized portfolio return
    for the *next* month uses only the assets that were active.
    """
    cov_kwargs = cov_kwargs or {}
    mu_kwargs = mu_kwargs or {}
    rf_monthly = (1.0 + rf_annual) ** (1.0 / MONTHS) - 1.0

    if not isinstance(rets.index, pd.DatetimeIndex):
        rets = rets.copy()
        rets.index = pd.to_datetime(rets.index)
    rets = rets.sort_index()

    dates = rets.index
    if len(dates) < lookback_months + 2:
        raise ValueError(
            f"walk_forward needs at least {lookback_months + 2} months; "
            f"got {len(dates)}"
        )

    windows: list[WalkForwardWindow] = []
    monthly_pnl: list[tuple[pd.Timestamp, float]] = []
    weights_rows: dict[pd.Timestamp, pd.Series] = {}

    for i in range(lookback_months, len(dates) - 1):
        rebal_date = dates[i]
        next_date = dates[i + 1]
        window = rets.iloc[i - lookback_months: i]

        # Active asset universe in this window
        valid = window.notna().sum()
        active = [c for c in window.columns if valid[c] >= min_obs_per_window]
        if len(active) < 2:
            continue

        sub = window[active].dropna(how="any")
        if len(sub) < min_obs_per_window:
            continue

        try:
            cov_res = _make_cov(cov_method, sub, **cov_kwargs)
            sub_cons = subset_constraints(cons, active)
            # BL needs a benchmark dict over active assets
            mu_kw = dict(mu_kwargs)
            if mu_method == "black_litterman":
                mu_kw.setdefault("benchmark", sub_cons.benchmark or {})
                mu_kw.setdefault("rf_monthly", rf_monthly)
            mu_vec = _make_mu(mu_method, sub, cov_res.cov, **mu_kw)
            sol = _solve_for_method(optimizer, cov_res.cov, mu_vec,
                                    sub_cons, rf_annual)
        except Exception:
            continue

        # Realize next-month return for active assets only.
        next_row = rets.iloc[i + 1]
        # Use only assets that had a return next month *and* were active.
        realized_assets = [c for c in active if pd.notna(next_row.get(c))]
        if not realized_assets:
            continue
        w_next = sol.weights.reindex(realized_assets).fillna(0.0)
        # If we had to drop assets at realization, renormalize what's left.
        if w_next.sum() > 0:
            w_next = w_next / w_next.sum()
        port_ret = float((w_next.values * next_row[realized_assets].values).sum())

        windows.append(WalkForwardWindow(
            rebal_date=rebal_date,
            active_assets=active,
            weights=sol.weights.copy(),
            realized_return=port_ret,
            monthly_vol_estimate=float(sol.volatility / np.sqrt(MONTHS)),
            method=optimizer,
        ))
        monthly_pnl.append((next_date, port_ret))
        # Pad weights to full universe for the history frame (NaN for inactive).
        full_w = pd.Series(np.nan, index=rets.columns)
        for c, v in sol.weights.items():
            full_w[c] = v
        weights_rows[rebal_date] = full_w

    if not monthly_pnl:
        raise RuntimeError("No walk-forward windows succeeded.")

    pnl = pd.Series(dict(monthly_pnl)).sort_index()
    cum = (1.0 + pnl).cumprod()
    weights_df = pd.DataFrame(weights_rows).T.sort_index()
    return WalkForwardResult(
        method=f"{optimizer} | Σ={cov_method} | μ={mu_method}",
        cum_wealth=cum, monthly_returns=pnl,
        weights_history=weights_df, windows=windows,
    )


def compare_methods(
    rets: pd.DataFrame,
    cons: Constraints,
    methods: list[dict],
    lookback_months: int = 60,
    rf_annual: float = 0.03,
) -> dict[str, WalkForwardResult]:
    """Run `walk_forward` for each entry in `methods` and return a dict
    keyed by a human-readable label. Each method dict needs at minimum
    {'label', 'optimizer', 'cov_method', 'mu_method'} and may carry
    'cov_kwargs', 'mu_kwargs'."""
    out: dict[str, WalkForwardResult] = {}
    for spec in methods:
        try:
            res = walk_forward(
                rets, cons,
                cov_method=spec["cov_method"],
                mu_method=spec.get("mu_method", "historical"),
                optimizer=spec["optimizer"],
                lookback_months=lookback_months,
                rf_annual=rf_annual,
                cov_kwargs=spec.get("cov_kwargs"),
                mu_kwargs=spec.get("mu_kwargs"),
            )
            out[spec["label"]] = res
        except Exception as e:
            # Skip silently with a label that explains failure
            out[spec["label"] + " (FAILED)"] = None  # type: ignore
            print(f"walk-forward {spec['label']!r} failed: {e}")
    return out


def metric_grid(results: dict[str, WalkForwardResult]) -> pd.DataFrame:
    rows = []
    for label, res in results.items():
        if res is None:
            continue
        rows.append({
            "method": label,
            "ann return (%)": round(res.ann_return * 100, 2),
            "ann vol (%)": round(res.ann_vol * 100, 2),
            "Sharpe": round(res.sharpe, 2),
            "max DD (%)": round(res.max_drawdown * 100, 2),
            "avg turnover (%)": round(res.turnover * 100, 2),
            "windows": len(res.windows),
        })
    return pd.DataFrame(rows)
