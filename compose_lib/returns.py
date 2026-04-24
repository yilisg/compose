"""Prices → monthly total returns.

Conventions:
- Equities / bonds / commodity / REIT / gold ETFs: monthly log returns from
  month-end adjusted close. (Log returns are additive across time, which is
  convenient for backtest math, and for monthly horizons they are within
  ~1 bp of simple returns.)
- Cash (^IRX): the Yahoo series is the 13-week T-bill yield in percent.
  We convert to a daily return ≈ yield/252, then compound to a monthly
  return. No dividends involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from compose_lib.universe import BY_CODE


def _cash_monthly_return(irx: pd.Series) -> pd.Series:
    """^IRX (13wk T-bill yield, % annualized) -> monthly return."""
    yld = irx.astype(float) / 100.0
    daily = yld / 252.0
    # Compound daily returns to monthly
    return (1.0 + daily).resample("ME").prod() - 1.0


def compute_monthly_returns(prices: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """Given the wide price panel and a list of asset codes, return a wide
    frame of **monthly** returns for the requested codes. Cash is treated
    specially (see module docstring).
    """
    out = {}
    for code in codes:
        spec = BY_CODE[code]
        tkr = spec.ticker
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        if spec.group == "cash":
            r = _cash_monthly_return(s)
        else:
            monthly_close = s.resample("ME").last()
            r = np.log(monthly_close).diff()
        out[code] = r
    rets = pd.DataFrame(out)
    # Drop rows where any asset is missing — SAA estimators need balanced panels
    rets = rets.dropna(how="any")
    return rets


def returns_from_uploaded(df: pd.DataFrame, already_returns: bool) -> pd.DataFrame:
    """Accept an uploaded CSV/JSON (first col = date, rest numeric) and
    return monthly returns. If the file already contains returns at any
    frequency, resample to monthly by compounding."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if already_returns:
        # Compound to monthly: log(1+r) sums, so log1p and resample.sum, then expm1
        logr = np.log1p(df)
        monthly = logr.resample("ME").sum()
        return np.expm1(monthly).dropna(how="any")
    else:
        monthly_close = df.resample("ME").last()
        return np.log(monthly_close).diff().dropna(how="any")


def excess_returns(rets: pd.DataFrame, cash_code: str = "cash") -> pd.DataFrame:
    """Subtract the cash return row-wise to get excess returns. If cash is
    not in the universe, returns the input unchanged."""
    if cash_code not in rets.columns:
        return rets
    return rets.sub(rets[cash_code], axis=0).drop(columns=[cash_code])


def annualize_return(monthly_ret: float) -> float:
    return (1.0 + monthly_ret) ** 12 - 1.0


def annualize_vol(monthly_vol: float) -> float:
    return monthly_vol * np.sqrt(12.0)
