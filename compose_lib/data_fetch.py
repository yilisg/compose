"""Yahoo Finance price fetch + cached parquet loader.

The cached panel is stored in `data/default_prices.parquet` and indexed by
date with one column per ticker. `^IRX` is the 13-week T-bill *yield* in
percent; it is converted to a monthly cash return in `returns.py`, not here,
so the parquet stays a pure price panel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from compose_lib.universe import ALL_ASSETS

PANEL_PATH = Path(__file__).parent.parent / "data" / "default_prices.parquet"


def fetch_prices(tickers: list[str], start: str = "2000-01-01") -> pd.DataFrame:
    """Download daily adjusted-close prices from Yahoo. Returns wide frame
    indexed by date with one column per ticker. Missing tickers are dropped
    with a warning."""
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )
    # Normalize: yfinance returns a single-ticker frame vs. a multi-ticker
    # MultiIndex frame depending on len(tickers). Unify both.
    if isinstance(raw.columns, pd.MultiIndex):
        close = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.sort_index().dropna(how="all")
    return close


def load_default_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"No cached panel at {PANEL_PATH}. Run `python refresh_prices.py`."
        )
    df = pd.read_parquet(PANEL_PATH)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def save_default_panel(df: pd.DataFrame) -> None:
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PANEL_PATH)


def all_tickers() -> list[str]:
    return [a.ticker for a in ALL_ASSETS]
