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
    # us_tsy is included as a bond fallback for sources (e.g. tabula private
    # mode) where us_agg has no direct equivalent — keeps the 60/40 split
    # honest. With us_agg available, default_benchmark prefers us_agg.
    "Tier 1 — Global 60/40 (5)":         ["us_eq", "us_agg", "us_tsy", "cash", "intl_eq", "gold"],
    "Tier 2 — All-Weather starter (7)":  ["us_eq", "us_agg", "cash", "intl_eq", "gold",
                                          "em_eq", "comdty"],
    "Tier 3 — Diversified (10)":         ["us_eq", "intl_eq", "em_eq",
                                          "us_tsy", "us_ig", "us_hy", "us_tips",
                                          "us_reit", "gold", "cash"],
}


#: A simple 60/40 benchmark used for the tracking-error constraint. Picks
#: the first available equity proxy and the first available bond proxy from
#: the active universe. If no bond is available (e.g. tabula private mode
#: where us_agg has no equivalent), falls back to 100% equity rather than
#: spreading weight equal-weight across non-bond assets.
def default_benchmark(codes: list[str]) -> dict[str, float]:
    # Hard-coded preference order — keep us_eq as the canonical equity proxy
    # and us_agg / us_tsy as the canonical bonds. Other rates/credit codes
    # (us_ig, us_hy, us_tips) are acceptable fallbacks but not preferred.
    EQUITY_PRIORITY = ["us_eq", "intl_eq", "em_eq"]
    BOND_PRIORITY = ["us_agg", "us_tsy", "us_ig", "us_tips", "us_hy"]
    eq = next((c for c in EQUITY_PRIORITY if c in codes), None)
    bond = next((c for c in BOND_PRIORITY if c in codes), None)
    if eq and bond:
        return {eq: 0.60, bond: 0.40}
    if eq:
        return {eq: 1.0}
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
