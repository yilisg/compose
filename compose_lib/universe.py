"""Asset universes — Tier 0 (starter, 3 assets) through Tier 3 (10 assets).

Each tier is strictly additive over the previous one, so Compose can slide
from a Bogleheads three-fund portfolio up to a diversified endowment-lite
universe without renaming any series.

Proxies are liquid US-listed ETFs where possible. Cash is proxied by the
13-week T-bill yield (^IRX) converted to a monthly return; all other series
are total-return ETFs where dividends are reinvested.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetSpec:
    code: str          # short ID used in the app (e.g. "us_eq")
    ticker: str        # Yahoo ticker (or "^IRX" for cash)
    name: str          # display name
    group: str         # "equity" | "rates" | "credit" | "real" | "cash"


#: Full roster — all ten assets across all tiers, deduplicated.
ALL_ASSETS: list[AssetSpec] = [
    # Tier 0
    AssetSpec("us_eq",   "SPY",  "US Equity (SPX)",         "equity"),
    AssetSpec("us_agg",  "AGG",  "US Aggregate Bonds",      "rates"),
    AssetSpec("cash",    "^IRX", "Cash (3M T-Bill)",        "cash"),
    # Tier 1
    AssetSpec("intl_eq", "EFA",  "Intl Developed Equity",   "equity"),
    AssetSpec("gold",    "GLD",  "Gold",                    "real"),
    # Tier 2
    AssetSpec("em_eq",   "EEM",  "Emerging Markets Equity", "equity"),
    AssetSpec("comdty",  "DBC",  "Broad Commodities",       "real"),
    # Tier 3 (replaces us_agg with split fixed-income sleeves, adds REITs)
    AssetSpec("us_tsy",  "IEF",  "US Treasuries (7-10y)",   "rates"),
    AssetSpec("us_ig",   "LQD",  "US IG Credit",            "credit"),
    AssetSpec("us_hy",   "HYG",  "US High Yield",           "credit"),
    AssetSpec("us_tips", "TIP",  "US TIPS",                 "rates"),
    AssetSpec("us_reit", "VNQ",  "US REITs",                "real"),
]


BY_CODE: dict[str, AssetSpec] = {a.code: a for a in ALL_ASSETS}


TIERS: dict[str, list[str]] = {
    "Tier 0 — Starter (3)":              ["us_eq", "us_agg", "cash"],
    "Tier 1 — Global 60/40 (5)":         ["us_eq", "us_agg", "cash", "intl_eq", "gold"],
    "Tier 2 — All-Weather starter (7)":  ["us_eq", "us_agg", "cash", "intl_eq", "gold",
                                          "em_eq", "comdty"],
    "Tier 3 — Diversified (10)":         ["us_eq", "intl_eq", "em_eq",
                                          "us_tsy", "us_ig", "us_hy", "us_tips",
                                          "us_reit", "gold", "cash"],
}


#: A simple 60/40 benchmark used for the tracking-error constraint. Falls
#: back to 100% US equity if AGG is not in the selected universe.
def default_benchmark(codes: list[str]) -> dict[str, float]:
    if "us_eq" in codes and "us_agg" in codes:
        return {"us_eq": 0.60, "us_agg": 0.40}
    if "us_eq" in codes and "us_tsy" in codes:
        return {"us_eq": 0.60, "us_tsy": 0.40}
    return {c: 1.0 / len(codes) for c in codes}


#: Group-level constraint presets that users typically want. Keys are group
#: names as they appear on AssetSpec.group.
def default_group_bounds(codes: list[str]) -> dict[str, tuple[float, float]]:
    """Return sensible (lower, upper) bounds per group for the given universe."""
    groups = {BY_CODE[c].group for c in codes}
    bounds: dict[str, tuple[float, float]] = {}
    if "equity" in groups:
        bounds["equity"] = (0.0, 0.90)
    if "rates" in groups:
        bounds["rates"] = (0.0, 0.90)
    if "credit" in groups:
        bounds["credit"] = (0.0, 0.50)
    if "real" in groups:
        bounds["real"] = (0.0, 0.30)
    if "cash" in groups:
        bounds["cash"] = (0.0, 0.30)
    return bounds


def tickers_for(codes: list[str]) -> list[str]:
    return [BY_CODE[c].ticker for c in codes]


def display_names(codes: list[str]) -> list[str]:
    return [BY_CODE[c].name for c in codes]
