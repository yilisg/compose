"""Regime labeling — vendored from rhyme/rhyme_lib/labeler.py.

Source: https://github.com/yilisg/rhyme  rhyme_lib/labeler.py
Pinned at commit 8b79fb3 ("Mode-specific regime labels, 3-clock Cycle tab,
expanded free-data panel"). See rhyme's CLAUDE.md for the full grid spec.

This module exposes `label_from_z(...)` for point-in-time regime labels
without running the full clustering pipeline. We vendor it (rather than
`pip install -e ../rhyme`) because Streamlit Cloud deploys break with
local editable installs.

If the upstream signature changes, update both this file and the pinned
commit hash above.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# Macro thresholds.
MACRO_THRESHOLD = 0.15
MACRO_RISK_MOD = 0.50

# Market thresholds.
MARKET_THRESHOLD = 0.25
MARKET_VOL_MOD = 0.70

Mode = Literal["macro", "market"]


def _macro_base(g: float, i: float) -> str:
    t = MACRO_THRESHOLD
    if g > t and i > t:
        return "Reflation"
    if g > t and abs(i) <= t:
        return "Expansion"
    if g > t and i < -t:
        return "Goldilocks"
    if abs(g) <= t and i > t:
        return "Inflationary"
    if abs(g) <= t and abs(i) <= t:
        return "Neutral"
    if abs(g) <= t and i < -t:
        return "Disinflation"
    if g < -t and i > t:
        return "Stagflation"
    if g < -t and abs(i) <= t:
        return "Slowdown"
    return "Deflationary bust"


def _macro_suffix(fin_z: float) -> str:
    if fin_z > MACRO_RISK_MOD:
        return " (risk-off)"
    if fin_z < -MACRO_RISK_MOD:
        return " (risk-on)"
    return ""


def _market_base(m: float, s: float) -> str:
    t = MARKET_THRESHOLD
    if m < -t and s > t:
        return "Melt-up"
    if m < -t and abs(s) <= t:
        return "Risk-on"
    if m < -t and s < -t:
        return "Recovery"
    if abs(m) <= t and s > t:
        return "Bullish"
    if abs(m) <= t and abs(s) <= t:
        return "Sideways"
    if abs(m) <= t and s < -t:
        return "Cautious"
    if m > t and s > t:
        return "Tightening peak"
    if m > t and abs(s) <= t:
        return "Risk-off"
    return "Crisis"


def _market_suffix(vix_z: float) -> str:
    if not np.isfinite(vix_z):
        return ""
    if vix_z > MARKET_VOL_MOD:
        return " (high vol)"
    if vix_z < -MARKET_VOL_MOD:
        return " (calm)"
    return ""


def label_from_z(
    growth_z: float = 0.0,
    inflation_z: float = 0.0,
    financial_z: float = 0.0,
    sentiment_z: float = 0.0,
    vix_z: float = float("nan"),
    mode: Mode = "macro",
) -> str:
    """Map theme z-scores to a regime label without running the full
    rhyme clustering pipeline. `mode='macro'` (default) uses growth ×
    inflation with monetary modifier; `mode='market'` uses monetary ×
    sentiment with VIX modifier."""
    if mode == "market":
        return _market_base(financial_z, sentiment_z) + _market_suffix(vix_z)
    return _macro_base(growth_z, inflation_z) + _macro_suffix(financial_z)


def regime_from_drawdown(
    eq_returns,
    dd_threshold: float = 0.10,
):
    """Lightweight fallback when no macro panel is available. Returns a
    string-labeled Series ('Stress' / 'Normal') indexed by `eq_returns`.

    `dd_threshold` is the running drawdown from peak that triggers Stress.
    """
    import pandas as pd
    cum = (1.0 + eq_returns).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return pd.Series(
        np.where(dd <= -abs(dd_threshold), "Stress", "Normal"),
        index=eq_returns.index, name="regime",
    )
