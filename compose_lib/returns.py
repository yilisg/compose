"""Prices → monthly total returns.

Conventions:
- Equities / bonds / commodity / REIT / gold ETFs: monthly log returns from
  month-end adjusted close. (Log returns are additive across time, which is
  convenient for backtest math, and for monthly horizons they are within
  ~1 bp of simple returns.)
- Cash (^IRX): the Yahoo series is the 13-week T-bill yield in percent.
  We convert to a daily return ≈ yield/252, then compound to a monthly
  return. No dividends involved.
- History extension: long-running Vanguard mutual-fund predecessors are
  spliced onto modern ETFs (e.g. VFINX→SPY) so Compose can study the
  pre-ETF era. HYG is back-spliced with the FRED total-return BAML HY
  series. See `splice_total_return` and `SPLICE_MAP`.
"""

from __future__ import annotations

import io
import urllib.request
import warnings

import numpy as np
import pandas as pd

from compose_lib.universe import BY_CODE


# Splice list: modern asset code -> ordered list of fallback price/return
# sources used for history before the modern series begins. Each entry is
# a (kind, identifier) pair where kind in {"yahoo", "fred"}.
#
# Order: most recent first. The series early in the list takes priority,
# older series fill the historical tail. Splice is done on monthly returns,
# rebased via the median ratio over an overlapping window.
SPLICE_MAP: dict[str, list[tuple[str, str]]] = {
    "us_eq":   [("yahoo", "SPY"),  ("yahoo", "VFINX")],          # 1980+
    "us_agg":  [("yahoo", "AGG"),  ("yahoo", "VBMFX")],          # 1986+
    "us_tsy":  [("yahoo", "IEF"),  ("yahoo", "VFITX")],          # 1991+
    "us_ig":   [("yahoo", "LQD"),  ("yahoo", "VWESX")],          # 1980+
    "us_hy":   [("yahoo", "HYG"),  ("fred",  "BAMLHYH0A0HYM2TRIV")],  # 1986+
    "intl_eq": [("yahoo", "EFA"),  ("yahoo", "VGTSX")],          # 1996+
    "em_eq":   [("yahoo", "EEM"),  ("yahoo", "VEIEX")],          # 1994+
}


def fetch_fred_total_return(series_id: str) -> pd.Series:
    """Pull a FRED CSV (e.g. BAMLHYH0A0HYM2TRIV) by HTTP. Returns a daily
    `pd.Series` of the index level. Network call — should be wrapped in
    cache when used inside the app."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    df = pd.read_csv(io.BytesIO(raw))
    # FRED CSV: first column is observation_date; series column matches id.
    date_col = df.columns[0]
    val_col = series_id if series_id in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.set_index(date_col)[val_col].dropna()
    s.name = series_id
    return s


def _cash_monthly_return(irx: pd.Series) -> pd.Series:
    """^IRX (13wk T-bill yield, % annualized) -> monthly return."""
    yld = irx.astype(float) / 100.0
    daily = yld / 252.0
    # Compound daily returns to monthly
    return (1.0 + daily).resample("ME").prod() - 1.0


def compute_monthly_returns(
    prices: pd.DataFrame,
    codes: list[str],
    extend_history: bool = False,
    start: str | None = "2000-01-01",
    rebase_overlap_months: int = 12,
) -> pd.DataFrame:
    """Given the wide price panel and a list of asset codes, return a wide
    frame of **monthly** returns for the requested codes. Cash is treated
    specially (see module docstring).

    Parameters
    ----------
    prices : wide DataFrame indexed by date with one column per ticker.
    codes  : asset codes (universe.BY_CODE keys).
    extend_history : if True, splice each code's modern ETF backward via
        the predecessors in `SPLICE_MAP` (Vanguard mutual funds for most
        equities/bonds; FRED BAML HY for HYG). The result is an *unbalanced*
        wide frame — older entries will have fewer columns. The caller is
        responsible for handling pro-rating per window.
    start : if not None, only keep rows on or after this ISO date. Used by
        the "long-term model" toggle. Set to None to keep all history.
    rebase_overlap_months : passed through to `splice_total_return`.
    """
    if extend_history:
        rets = compute_extended_monthly_returns(
            prices, codes, rebase_overlap_months=rebase_overlap_months,
        )
    else:
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

    if start is not None:
        rets = rets.loc[pd.Timestamp(start):]

    # Drop rows where any asset is missing — SAA estimators need balanced panels.
    # When `extend_history` is True the caller usually wants the unbalanced
    # frame for pro-rated walk-forward, so we only dropna in the legacy path.
    if not extend_history:
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


# ---------------------------------------------------------------------------
# Splice helpers
# ---------------------------------------------------------------------------


def _to_monthly_simple_return(prices: pd.Series) -> pd.Series:
    """Month-end resample → simple monthly return (`p_t / p_{t-1} − 1`).
    Splices use simple returns so they compound correctly when concatenated;
    the rest of the codebase uses log returns but they're within ~1 bp at
    monthly horizons and the splice is in the noise of the rebase anyway."""
    if prices.empty:
        return pd.Series(dtype=float)
    monthly = prices.resample("ME").last()
    return monthly.pct_change()


def _fetch_one(kind: str, identifier: str,
               yahoo_panel: pd.DataFrame | None) -> pd.Series:
    """Resolve one splice leg to a daily price series.

    `yahoo_panel` is the cached default-prices DataFrame; if a Yahoo ticker
    is in there we read it from the cache rather than re-downloading. FRED
    series are always fetched live (small files, no cache yet)."""
    if kind == "yahoo":
        if yahoo_panel is not None and identifier in yahoo_panel.columns:
            return yahoo_panel[identifier].dropna()
        # Fall back to live download
        from compose_lib.data_fetch import fetch_prices
        df = fetch_prices([identifier], start="1980-01-01")
        return df[identifier].dropna() if identifier in df.columns else pd.Series(dtype=float)
    if kind == "fred":
        return fetch_fred_total_return(identifier)
    raise ValueError(f"Unknown splice source kind: {kind}")


def splice_total_return(
    sources: list[tuple[str, str]],
    yahoo_panel: pd.DataFrame | None = None,
    rebase_overlap_months: int = 12,
    tracking_error_warn: float = 0.005,
) -> pd.Series:
    """Splice a chain of monthly total-return series into one long series.

    `sources` is the ordered list from `SPLICE_MAP[asset_code]` — most-recent
    series first, older predecessors after.

    Method
    ------
    For each consecutive pair (modern, older), we compute the median ratio
    of `(1 + r_modern) / (1 + r_older)` over the overlap window of size
    `rebase_overlap_months`. The older series is then scaled by that ratio
    so its compounded level matches the modern series at the splice point.

    The check at the end warns (does not raise) if the absolute MoM
    tracking error in the splice month exceeds `tracking_error_warn`
    (default 50 bps). The function returns a `pd.Series` of monthly simple
    returns covering the union of dates.
    """
    legs: list[pd.Series] = []
    for kind, identifier in sources:
        try:
            prices = _fetch_one(kind, identifier, yahoo_panel)
        except Exception as e:
            warnings.warn(f"splice_total_return: failed to fetch {kind}:{identifier} ({e})")
            continue
        ret = _to_monthly_simple_return(prices).dropna()
        if not ret.empty:
            legs.append(ret)
    if not legs:
        return pd.Series(dtype=float)
    if len(legs) == 1:
        return legs[0]

    # Walk from the modern series backward, rebasing each older leg.
    spliced = legs[0].copy()
    for older in legs[1:]:
        overlap = older.index.intersection(spliced.index)
        if len(overlap) >= rebase_overlap_months:
            window = overlap.sort_values()[-rebase_overlap_months:]
            ratio = ((1.0 + spliced.loc[window]) / (1.0 + older.loc[window])).median()
        elif len(overlap) >= 1:
            window = overlap
            ratio = ((1.0 + spliced.loc[window]) / (1.0 + older.loc[window])).median()
        else:
            ratio = 1.0
        scale = float(ratio) if np.isfinite(ratio) else 1.0
        rebased = (1.0 + older) * scale - 1.0

        # MoM tracking-error check on the overlap (after rebase)
        if len(overlap) >= 1:
            te = (spliced.loc[overlap] - rebased.loc[overlap]).abs()
            if (te > tracking_error_warn).any():
                worst = float(te.max())
                worst_dt = te.idxmax()
                warnings.warn(
                    f"splice_total_return: max |MoM TE| = {worst*100:.2f}% "
                    f"on {worst_dt:%Y-%m} exceeds {tracking_error_warn*100:.1f}% "
                    f"between {sources[0][1]} and one of its predecessors."
                )

        # Take the modern series where it exists; older where modern is NaN.
        early = rebased.loc[rebased.index.difference(spliced.index)]
        spliced = pd.concat([spliced, early]).sort_index()

    return spliced.sort_index()


def splice_tracking_error(sources: list[tuple[str, str]],
                          yahoo_panel: pd.DataFrame | None = None,
                          rebase_overlap_months: int = 12) -> dict:
    """Diagnostic: returns per-pair max |MoM TE| in the splice overlap.
    Returns a dict keyed by 'modern_vs_older' with floats (in absolute
    return units, not percent)."""
    out: dict = {}
    legs: list[tuple[str, pd.Series]] = []
    for kind, identifier in sources:
        try:
            prices = _fetch_one(kind, identifier, yahoo_panel)
        except Exception:
            continue
        ret = _to_monthly_simple_return(prices).dropna()
        if not ret.empty:
            legs.append((identifier, ret))
    for i in range(len(legs) - 1):
        modern_id, modern = legs[i]
        older_id, older = legs[i + 1]
        overlap = older.index.intersection(modern.index)
        if overlap.empty:
            out[f"{modern_id}_vs_{older_id}"] = float("nan")
            continue
        window = overlap.sort_values()[-rebase_overlap_months:]
        ratio = ((1.0 + modern.loc[window]) / (1.0 + older.loc[window])).median()
        scale = float(ratio) if np.isfinite(ratio) else 1.0
        rebased = (1.0 + older) * scale - 1.0
        te = (modern.loc[overlap] - rebased.loc[overlap]).abs().max()
        out[f"{modern_id}_vs_{older_id}"] = float(te)
    return out


def compute_extended_monthly_returns(
    prices: pd.DataFrame,
    codes: list[str],
    rebase_overlap_months: int = 12,
) -> pd.DataFrame:
    """Like `compute_monthly_returns` but for each code in `SPLICE_MAP`,
    extends history backward via Vanguard / FRED predecessor series. Codes
    not in `SPLICE_MAP` follow the regular path.
    """
    out: dict[str, pd.Series] = {}
    for code in codes:
        spec = BY_CODE[code]
        if spec.group == "cash":
            tkr = spec.ticker
            if tkr in prices.columns:
                from compose_lib.returns import _cash_monthly_return
                out[code] = _cash_monthly_return(prices[tkr].dropna())
            continue
        if code in SPLICE_MAP:
            try:
                out[code] = splice_total_return(
                    SPLICE_MAP[code], yahoo_panel=prices,
                    rebase_overlap_months=rebase_overlap_months,
                )
            except Exception as e:
                warnings.warn(f"splice failed for {code}: {e}; falling back to plain ETF")
                tkr = spec.ticker
                if tkr in prices.columns:
                    out[code] = _to_monthly_simple_return(prices[tkr].dropna())
            continue
        # Default path — modern ETF only, log returns to match the v1 convention.
        tkr = spec.ticker
        if tkr in prices.columns:
            monthly_close = prices[tkr].dropna().resample("ME").last()
            out[code] = np.log(monthly_close).diff()

    rets = pd.DataFrame(out)
    return rets
