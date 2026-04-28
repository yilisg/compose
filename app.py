"""Compose — strategic asset allocation workbench.

Eight tabs:
  1. Overview     — headline: exp return / vol / Sharpe / TE, weights, risk contribution
  2. Universe     — asset table, history, correlation preview
  3. Views (μ)    — side-by-side μ estimates; manual CMAs editor; BL views editor
  4. Covariance   — Σ and correlation heatmaps for each estimator
  5. Optimize     — efficient frontier + Capital Market Line, solved portfolio starred
  6. Backtest     — walk-forward comparison of methods + stacked-area weights
  7. Stress       — re-solve under a stress Σ; weight drift
  8. Methodology  — plain-English explanation, renders METHODOLOGY.md

Every portfolio is a composition.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from compose_lib.backtest import compare_methods, metric_grid
from compose_lib.covariance import (
    cov_to_corr,
    condition_number,
    ewma_cov,
    ledoit_wolf_cov,
    oas_cov,
    regime_blended_cov,
    sample_cov,
    stress_blended_cov,
)
from compose_lib.data_fetch import load_default_panel
from compose_lib.diagnostics import (
    bootstrap_weights,
    risk_contributions,
    weight_ci,
)
from compose_lib.expected_returns import (
    black_litterman_equilibrium,
    blend_bl_with_views,
    historical_mean,
    jorion_shrinkage,
    manual_mu,
)
from compose_lib.frontier import asset_points, cml_points, trace_frontier
from compose_lib.optimize import (
    Constraints,
    hrp,
    max_return_at_te,
    max_return_at_vol,
    max_sharpe,
    min_cvar,
    min_variance,
    resampled,
    risk_parity,
)
from compose_lib.regime_label import regime_from_drawdown
from compose_lib.returns import compute_monthly_returns, returns_from_uploaded
from compose_lib.universe import (
    ALL_ASSETS,
    BY_CODE,
    TIERS,
    default_benchmark,
    default_group_bounds,
    display_names,
)


# Tabula → compose code mapping. Tabula stores series_ids per the
# watchlist; we map a few common ones onto compose's asset universe so
# users with a tabula panel can drive Compose end-to-end. Anything not
# in this mapping is dropped silently.
TABULA_SERIES_MAP: dict[str, str] = {
    # --- Yahoo tickers used by tabula (most common case) ---
    "SPY": "us_eq",
    "AGG": "us_agg",
    "EFA": "intl_eq",
    "EEM": "em_eq",
    "GLD": "gold",
    "DBC": "comdty",
    "IEF": "us_tsy",
    "LQD": "us_ig",
    "HYG": "us_hy",
    "TIP": "us_tips",
    "VNQ": "us_reit",
    "^IRX": "cash",
    # --- common alternative IDs ---
    "^GSPC": "us_eq",
    "GC=F": "gold",
}


TABULA_PANEL_PATH = Path("/Users/yili/Desktop/Claude/tabula/data/output/tabula_panel.parquet")


st.set_page_config(page_title="Compose", layout="wide", page_icon="🎼")


# ---------------------------------------------------------------------------
# Cache wrappers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_prices():
    return load_default_panel()


@st.cache_data(show_spinner=False)
def _cached_returns(
    panel_key: str, panel_bytes: bytes, codes_key: str,
    extend_history: bool = False, start: str | None = "2000-01-01",
):
    panel = pd.read_parquet(io.BytesIO(panel_bytes))
    panel.index = pd.to_datetime(panel.index)
    codes = codes_key.split(",")
    return compute_monthly_returns(
        panel, codes,
        extend_history=extend_history, start=start,
    )


@st.cache_data(show_spinner=False)
def _cached_tabula_panel(path_str: str):
    """Load the tabula long-format parquet and pivot to a wide price panel
    keyed by compose asset code. We read directly with `pd.read_parquet`
    rather than importing tabula, per the brief."""
    df = pd.read_parquet(path_str)
    if not {"series_id", "observation_date", "value"}.issubset(df.columns):
        raise ValueError(
            f"tabula parquet missing required columns; got {list(df.columns)}"
        )
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    # Map series_id -> compose code (drop unmapped)
    df["code"] = df["series_id"].map(TABULA_SERIES_MAP)
    df = df.dropna(subset=["code"])
    if df.empty:
        return pd.DataFrame()
    wide = (df.pivot_table(
        index="observation_date", columns="code",
        values="value", aggfunc="last",
    ).sort_index())
    # Translate compose codes back to the ticker names compute_monthly_returns expects.
    wide.columns = [BY_CODE[c].ticker for c in wide.columns]
    return wide


GOLD = "#FFD700"
CRIMSON = "#D10000"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


st.sidebar.title("Compose")
st.sidebar.caption("*Every portfolio is a composition.*")

source_mode = st.sidebar.selectbox(
    "Panel source",
    ["Public (Yahoo + cache)", "Private (tabula)", "Custom upload"],
    index=0,
    help=(
        "Public — built-in cached Yahoo panel (default). "
        "Private — read tabula's long-format parquet. "
        "Custom — upload your own CSV/parquet."
    ),
)

uploaded = None
upload_kind = "prices"
if source_mode == "Custom upload":
    uploaded = st.sidebar.file_uploader(
        "File (first column = date)", type=["csv", "json", "parquet"],
    )
    upload_kind = st.sidebar.radio(
        "Upload contains", ["prices", "returns"], index=0, horizontal=True,
        help="'prices' -> log returns to monthly. 'returns' -> compounded to monthly.",
    )

tier_name = st.sidebar.radio("Universe", list(TIERS.keys()), index=1)

long_term_model = st.sidebar.checkbox(
    "Long-term model (full history)",
    value=False,
    help=(
        "OFF (default): panel filtered to 2000-01-01+. "
        "ON: use the full available history. With Public source this also "
        "splices Vanguard mutual-fund / FRED predecessors onto the modern "
        "ETFs (SPY←VFINX from 1980, AGG←VBMFX from 1986, etc.) — fixes "
        "Tier 3's HYG-induced 2007 cliff."
    ),
)

with st.sidebar.expander("Expected returns (μ)", expanded=True):
    mu_method = st.radio(
        "μ method",
        ["historical", "jorion", "black_litterman", "manual"],
        format_func=lambda x: {
            "historical": "Historical mean",
            "jorion": "Jorion shrinkage",
            "black_litterman": "BL equilibrium",
            "manual": "Manual CMAs",
        }[x],
        index=2,
        help="Black-Litterman equilibrium is the default anchor. Historical "
             "mean is noisy and biased. Jorion shrinks historical toward the "
             "MVP return. Manual lets you paste published CMAs.",
    )
    bl_delta = st.slider(
        "BL risk aversion (δ)", 1.0, 6.0, 3.0, 0.5,
        help="Only used for Black-Litterman. δ = 3 is typical for a diversified mandate.",
    )

with st.sidebar.expander("Covariance (Σ)", expanded=True):
    cov_method = st.radio(
        "Σ method",
        ["ledoit_wolf", "oas", "ewma", "sample", "stress_blended"],
        format_func=lambda x: {
            "ledoit_wolf": "Ledoit-Wolf (default)",
            "oas": "OAS shrinkage",
            "ewma": "EWMA (recency-weighted)",
            "sample": "Sample",
            "stress_blended": "Stress-blended (SPX drawdown)",
        }[x],
        index=0,
    )
    if cov_method == "stress_blended":
        dd_threshold = st.slider(
            "Drawdown threshold (%)", 5, 30, 10, 1,
            help="Months where SPX cumulative drawdown is at least this bad are tagged 'stress'.",
        ) / 100.0
        stress_weight = st.slider(
            "Stress weight (p)", 0.0, 1.0, 0.30, 0.05,
            help="Σ_blend = (1-p)·Σ_normal + p·Σ_stress.",
        )
    else:
        dd_threshold = 0.10
        stress_weight = 0.30
    if cov_method == "ewma":
        ewma_half_life = st.slider(
            "EWMA half-life (months)", 6, 24, 12, 1,
            help="Weight on observation k months ago is 0.5^(k/HL).",
        )
    else:
        ewma_half_life = 12

    use_regime_cov = st.checkbox(
        "Regime-aware Σ (cluster + blend)",
        value=False,
        help=(
            "Estimate Σ separately per regime label and blend by today's "
            "regime probabilities. Without a macro panel we use a "
            "simple SPX-drawdown bucket; with rhyme labels available the "
            "labeling can be replaced by `label_from_z`."
        ),
    )

with st.sidebar.expander("Objective", expanded=True):
    obj = st.radio(
        "Objective",
        ["max_sharpe", "max_ret_te", "max_ret_vol", "min_var",
         "risk_parity", "hrp", "min_cvar", "resampled"],
        format_func=lambda x: {
            "max_sharpe":   "Max Sharpe (default)",
            "max_ret_te":   "Max Return s.t. TE",
            "max_ret_vol":  "Max Return s.t. Vol",
            "min_var":      "Min Variance",
            "risk_parity":  "Risk Parity (ERC)",
            "hrp":          "Hierarchical Risk Parity",
            "min_cvar":     "Min CVaR",
            "resampled":    "Resampled (Michaud)",
        }[x],
        index=0,
    )
    if obj == "max_ret_te":
        te_cap = st.slider("Tracking error cap (% annual)", 1.0, 15.0, 5.0, 0.5) / 100.0
    else:
        te_cap = 0.05
    if obj == "max_ret_vol":
        vol_cap = st.slider("Volatility cap (% annual)", 3.0, 25.0, 10.0, 0.5) / 100.0
    else:
        vol_cap = 0.10
    if obj == "min_cvar":
        cvar_alpha = st.slider("CVaR α", 0.80, 0.99, 0.95, 0.01)
    else:
        cvar_alpha = 0.95
    if obj == "resampled":
        n_sims = st.slider("Resampled sims", 50, 500, 200, 50)
    else:
        n_sims = 200

with st.sidebar.expander("Constraints", expanded=False):
    rf_annual = st.slider("Risk-free rate (%)", 0.0, 8.0, 3.0, 0.25) / 100.0
    allow_short = st.checkbox("Allow short positions", value=False)
    use_group_caps = st.checkbox("Apply group caps", value=True,
                                 help="Editable in the 'Group bounds' sub-section below.")
    uniform_lb = st.slider("Per-asset lower bound (%)", 0.0, 30.0, 0.0, 1.0) / 100.0
    uniform_ub = st.slider("Per-asset upper bound (%)", 10.0, 100.0, 100.0, 5.0) / 100.0
    use_turnover_cap = st.checkbox(
        "Turnover cap",
        value=False,
        help="Annualized L1 turnover ≤ cap (vs current weights = benchmark).",
    )
    if use_turnover_cap:
        turnover_cap_annual = st.slider(
            "Turnover cap (% annual)", 5.0, 400.0, 100.0, 5.0,
        ) / 100.0
    else:
        turnover_cap_annual = None

with st.sidebar.expander("Per-asset box bounds", expanded=False):
    st.caption("Override the uniform bounds for individual assets.")
    if "per_asset_box" not in st.session_state:
        st.session_state.per_asset_box = {}
    enable_per_asset = st.checkbox("Enable per-asset bounds", value=False)

with st.sidebar.expander("Group bounds (editable)", expanded=False):
    st.caption("Editable group caps. Applied when 'Apply group caps' is on.")
    if "group_bounds_edits" not in st.session_state:
        st.session_state.group_bounds_edits = {
            "equity": (0.0, 0.90),
            "rates": (0.0, 0.90),
            "credit": (0.0, 0.50),
            "real": (0.0, 0.30),
            "cash": (0.0, 0.30),
        }
    new_groups = {}
    for g, (lo, hi) in st.session_state.group_bounds_edits.items():
        col_lo, col_hi = st.columns(2)
        with col_lo:
            new_lo = st.number_input(
                f"{g} min %", min_value=0.0, max_value=100.0,
                value=float(lo * 100), step=5.0, key=f"glo_{g}",
            ) / 100.0
        with col_hi:
            new_hi = st.number_input(
                f"{g} max %", min_value=0.0, max_value=100.0,
                value=float(hi * 100), step=5.0, key=f"ghi_{g}",
            ) / 100.0
        new_groups[g] = (new_lo, new_hi)
    st.session_state.group_bounds_edits = new_groups


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


start_filter = None if long_term_model else "2000-01-01"

if source_mode == "Public (Yahoo + cache)":
    try:
        prices = _cached_prices()
    except FileNotFoundError as e:
        st.error(f"{e}\n\nRun `python refresh_prices.py` to build the cache.")
        st.stop()
    codes = TIERS[tier_name]
    buf = io.BytesIO()
    prices.to_parquet(buf)
    rets = _cached_returns(
        "default", buf.getvalue(), ",".join(codes),
        extend_history=long_term_model, start=start_filter,
    )
    panel_source = "default"
elif source_mode == "Private (tabula)":
    if not TABULA_PANEL_PATH.exists():
        st.error(
            f"Tabula parquet not found at `{TABULA_PANEL_PATH}`. "
            "Build it with tabula's pipeline first."
        )
        st.stop()
    try:
        prices = _cached_tabula_panel(str(TABULA_PANEL_PATH))
    except Exception as e:
        st.error(f"Failed to read tabula parquet: {e}")
        st.stop()
    if prices.empty:
        st.error(
            "No tabula series mapped to a compose asset. "
            "Add IDs to `TABULA_SERIES_MAP` in `app.py`."
        )
        st.stop()
    codes = TIERS[tier_name]
    # Filter codes to those whose ticker is present in the tabula panel.
    avail_codes = [c for c in codes if BY_CODE[c].ticker in prices.columns]
    if not avail_codes:
        st.error(
            "Tabula panel does not contain any tickers for this tier. "
            f"Wanted: {[BY_CODE[c].ticker for c in codes]}; "
            f"have: {list(prices.columns)}."
        )
        st.stop()
    if len(avail_codes) < len(codes):
        st.warning(
            "Tabula is missing some tier tickers — running with the "
            f"available subset: {avail_codes}"
        )
    codes = avail_codes
    buf = io.BytesIO()
    prices.to_parquet(buf)
    rets = _cached_returns(
        "tabula", buf.getvalue(), ",".join(codes),
        extend_history=long_term_model, start=start_filter,
    )
    panel_source = "tabula"
else:
    if uploaded is None:
        st.info(
            "Upload a CSV / JSON / parquet in the sidebar "
            "(first column = date, rest numeric)."
        )
        st.stop()
    if uploaded.name.endswith(".parquet"):
        raw = pd.read_parquet(uploaded)
    elif uploaded.name.endswith(".json"):
        raw = pd.read_json(uploaded)
    else:
        raw = pd.read_csv(uploaded)
    if raw.index.dtype == "object" or not isinstance(raw.index, pd.DatetimeIndex):
        raw = raw.set_index(raw.columns[0])
    rets = returns_from_uploaded(raw, already_returns=(upload_kind == "returns"))
    if start_filter is not None:
        rets = rets.loc[pd.Timestamp(start_filter):]
    codes = list(rets.columns)
    panel_source = uploaded.name

if rets.empty or rets.shape[0] < 24:
    st.error(f"Not enough history: {rets.shape[0]} monthly obs.")
    st.stop()

# When extending history we get an unbalanced frame; for the headline
# solver pipeline use the dropna'd common-history view. The Backtest tab
# operates on the unbalanced frame so it can pro-rate per window.
rets_full = rets
rets = rets_full.dropna(how="any")
if rets.shape[0] < 24:
    st.error(
        f"Common history too short: {rets.shape[0]} monthly obs after dropna. "
        "Try unchecking 'Long-term model' or shrinking the universe."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Estimate μ and Σ
# ---------------------------------------------------------------------------


cov_fns = {
    "sample": lambda r: sample_cov(r),
    "ledoit_wolf": lambda r: ledoit_wolf_cov(r),
    "oas": lambda r: oas_cov(r),
    "ewma": lambda r: ewma_cov(r, half_life_months=ewma_half_life),
    "stress_blended": lambda r: stress_blended_cov(
        r, equity_col=("us_eq" if "us_eq" in r.columns else r.columns[0]),
        dd_threshold=dd_threshold, stress_weight=stress_weight,
    ),
}

try:
    cov_res = cov_fns[cov_method](rets)
    if use_regime_cov:
        # Bucket history by SPX-drawdown regime and blend Σ.
        eq_col = "us_eq" if "us_eq" in rets.columns else rets.columns[0]
        regime = regime_from_drawdown(rets[eq_col], dd_threshold=dd_threshold)
        # Today's prob: just look at the last month's regime.
        today_label = regime.iloc[-1]
        # Mild blend: 70% today's regime, 30% the other(s).
        labels = sorted(set(regime.unique()))
        probs = {label: (0.7 if label == today_label else 0.3 / max(1, len(labels) - 1))
                 for label in labels}
        cov_res = regime_blended_cov(rets, regime, today_probs=probs)
except Exception as e:
    st.error(f"Covariance estimation failed: {e}")
    st.stop()

cov = cov_res.cov

if mu_method == "historical":
    mu_res = historical_mean(rets)
elif mu_method == "jorion":
    mu_res = jorion_shrinkage(rets)
elif mu_method == "black_litterman":
    bm_for_bl = default_benchmark(codes)
    mu_res = black_litterman_equilibrium(
        cov, bm_for_bl, risk_aversion=bl_delta,
        risk_free_monthly=(1.0 + rf_annual) ** (1.0 / 12) - 1.0,
    )
elif mu_method == "manual":
    if "manual_mu_edits" not in st.session_state:
        st.session_state.manual_mu_edits = {c: 0.05 for c in codes}
    for c in codes:
        st.session_state.manual_mu_edits.setdefault(c, 0.05)
    mu_res = manual_mu(st.session_state.manual_mu_edits, codes)
else:
    mu_res = historical_mean(rets)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


benchmark = default_benchmark(codes)
group_map = {c: BY_CODE[c].group for c in codes if c in BY_CODE}
if use_group_caps:
    # Filter the editable group bounds to groups present in this universe.
    groups_in_use = {BY_CODE[c].group for c in codes}
    group_bounds = {
        g: tuple(v) for g, v in st.session_state.group_bounds_edits.items()
        if g in groups_in_use
    }
else:
    group_bounds = {}

# Per-asset bounds: start uniform, then layer the per-asset overrides.
box_bounds = {c: (uniform_lb, uniform_ub) for c in codes}
if enable_per_asset:
    for c in codes:
        lo, hi = st.session_state.per_asset_box.get(c, (uniform_lb, uniform_ub))
        box_bounds[c] = (lo, hi)

cons = Constraints(
    box=box_bounds,
    group_bounds=group_bounds,
    group_map=group_map,
    turnover=turnover_cap_annual,
    current_weights=benchmark if turnover_cap_annual is not None else None,
    benchmark=benchmark,
    allow_short=allow_short,
)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def _solve() -> "Solution":  # noqa: F821
    if obj == "max_sharpe":
        return max_sharpe(cov, mu_res.mu, cons, rf_annual=rf_annual)
    if obj == "max_ret_te":
        return max_return_at_te(cov, mu_res.mu, te_cap, cons, rf_annual=rf_annual)
    if obj == "max_ret_vol":
        return max_return_at_vol(cov, mu_res.mu, vol_cap, cons, rf_annual=rf_annual)
    if obj == "min_var":
        return min_variance(cov, mu_res.mu, cons, rf_annual=rf_annual)
    if obj == "risk_parity":
        return risk_parity(cov, mu_res.mu, cons, rf_annual=rf_annual)
    if obj == "hrp":
        return hrp(cov, mu_res.mu, cons, rf_annual=rf_annual)
    if obj == "min_cvar":
        return min_cvar(rets, mu_res.mu, cov, cons, alpha=cvar_alpha, rf_annual=rf_annual)
    if obj == "resampled":
        return resampled(max_sharpe, mu_res.mu, cov, rets, cons,
                         n_sims=n_sims, rf_annual=rf_annual)
    raise ValueError(obj)


try:
    sol = _solve()
except Exception as e:
    st.error(f"Optimizer failed: {e}")
    st.stop()


# Frontier + CML (always compute for the Optimize tab, reuse the tangent)
try:
    frontier = trace_frontier(cov, mu_res.mu, cons, rf_annual=rf_annual, n_points=35)
except Exception as e:
    frontier = None
    frontier_err = str(e)


# Benchmark stats and TE
bm_w = pd.Series(benchmark).reindex(codes).fillna(0.0)
bm_vol_m = float(np.sqrt(bm_w.values @ cov.values @ bm_w.values))
bm_ret_m = float(bm_w.values @ mu_res.mu.values)
w_minus_b = sol.weights.reindex(codes).values - bm_w.values
te_ann = float(np.sqrt(w_minus_b @ cov.values @ w_minus_b) * np.sqrt(12))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


(tab_overview, tab_universe, tab_views, tab_cov, tab_opt,
 tab_backtest, tab_stress, tab_method) = st.tabs(
    ["Overview", "Universe", "Views (μ)", "Covariance",
     "Optimize", "Backtest", "Stress", "Methodology"]
)


# --- Overview --------------------------------------------------------------


with tab_overview:
    st.title("Compose")
    st.caption(f"{tier_name}  |  μ = {mu_method}  |  Σ = {cov_res.method}  |  objective = {obj}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected return", f"{sol.exp_return * 100:.2f}%")
    col2.metric("Volatility", f"{sol.volatility * 100:.2f}%")
    col3.metric("Sharpe ratio", f"{sol.sharpe:.2f}")
    col4.metric("TE vs 60/40", f"{te_ann * 100:.2f}%")

    # Weights bar chart
    st.subheader("Solved weights")
    w = sol.weights.reindex(codes)
    names = pd.Series({c: BY_CODE[c].name for c in codes})
    bar_df = pd.DataFrame({"asset": names.values, "weight": w.values, "code": w.index})
    bar_df = bar_df.sort_values("weight", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bar_df["weight"] * 100, y=bar_df["asset"],
        orientation="h",
        marker=dict(color=GOLD, line=dict(color=CRIMSON, width=1.5)),
        text=[f"{v*100:.1f}%" for v in bar_df["weight"]],
        textposition="outside",
    ))
    # Overlay benchmark for comparison
    bm_plot = [bm_w.get(c, 0.0) * 100 for c in bar_df["code"]]
    fig.add_trace(go.Scatter(
        x=bm_plot, y=bar_df["asset"], mode="markers",
        marker=dict(color="rgba(120,120,120,0.8)", symbol="line-ns-open", size=20,
                    line=dict(width=3)),
        name="60/40 benchmark",
    ))
    fig.update_layout(
        height=max(320, 40 * len(codes)),
        xaxis_title="Weight (%)", yaxis_title=None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk contributions
    st.subheader("Risk contributions (annualized)")
    rc = risk_contributions(sol.weights, cov).loc[codes]
    rc["name"] = [BY_CODE[c].name for c in rc.index]
    rc_display = rc[["name", "weight", "mrc", "rc", "pct"]].rename(columns={
        "weight": "weight",
        "mrc": "marginal RC",
        "rc": "RC",
        "pct": "% of σ",
    })
    rc_display["weight"] = (rc_display["weight"] * 100).round(2).astype(str) + "%"
    rc_display["marginal RC"] = (rc_display["marginal RC"] * 100).round(2).astype(str) + "%"
    rc_display["RC"] = (rc_display["RC"] * 100).round(2).astype(str) + "%"
    rc_display["% of σ"] = (rc_display["% of σ"] * 100).round(1).astype(str) + "%"
    st.dataframe(rc_display, use_container_width=True, hide_index=True)

    st.caption(
        f"Benchmark (60/40): {bm_w.round(2).to_dict()}. "
        f"Shrinkage intensity λ = {cov_res.shrinkage:.3f} (if applicable). "
        f"Risk-free rate = {rf_annual * 100:.2f}% annual."
    )


# --- Universe --------------------------------------------------------------


with tab_universe:
    st.subheader(f"Universe — {tier_name}")
    rows = []
    for c in codes:
        a = BY_CODE[c]
        s = rets[c]
        rows.append({
            "code": c,
            "ticker": a.ticker,
            "name": a.name,
            "group": a.group,
            "n_obs": int(s.notna().sum()),
            "start": s.dropna().index.min().date() if s.notna().any() else None,
            "monthly μ (%)": float(s.mean() * 100),
            "monthly σ (%)": float(s.std() * 100),
            "ann. μ (%)": float(((1 + s.mean()) ** 12 - 1) * 100),
            "ann. σ (%)": float(s.std() * np.sqrt(12) * 100),
        })
    st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)

    st.subheader("Correlation heatmap (full sample)")
    corr = rets.corr()
    fig = px.imshow(
        corr.values, x=corr.columns, y=corr.index,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=".2f", aspect="auto",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative growth of $1 (monthly)")
    wealth = (1 + rets).cumprod()
    fig = go.Figure()
    for c in codes:
        fig.add_trace(go.Scatter(
            x=wealth.index, y=wealth[c], mode="lines",
            name=BY_CODE[c].name,
        ))
    fig.update_layout(height=420, yaxis_type="log", yaxis_title="Growth of $1 (log)")
    st.plotly_chart(fig, use_container_width=True)


# --- Views -----------------------------------------------------------------


with tab_views:
    st.subheader("Expected return views — side by side")
    mu_hist = historical_mean(rets)
    mu_jor = jorion_shrinkage(rets)
    bm_for_bl = default_benchmark(codes)
    mu_bl = black_litterman_equilibrium(
        cov, bm_for_bl, risk_aversion=bl_delta,
        risk_free_monthly=(1.0 + rf_annual) ** (1.0 / 12) - 1.0,
    )
    compare = pd.DataFrame({
        "name": [BY_CODE[c].name for c in codes],
        "historical": (mu_hist.mu_annual * 100).round(2).values,
        "jorion": (mu_jor.mu_annual * 100).round(2).values,
        "BL equilibrium": (mu_bl.mu_annual * 100).round(2).values,
    }, index=codes)
    if mu_method == "manual":
        compare["manual"] = [
            st.session_state.manual_mu_edits.get(c, 0.05) * 100 for c in codes
        ]
    compare["**selected**"] = (mu_res.mu_annual * 100).round(2).values
    st.dataframe(compare, use_container_width=True)

    if mu_method == "manual":
        st.subheader("Edit manual CMAs (annualized %)")
        new_vals = {}
        cols = st.columns(min(4, len(codes)))
        for i, c in enumerate(codes):
            with cols[i % len(cols)]:
                new_vals[c] = st.number_input(
                    BY_CODE[c].name,
                    value=float(st.session_state.manual_mu_edits.get(c, 0.05) * 100),
                    step=0.1, key=f"mu_input_{c}",
                ) / 100.0
        if st.button("Apply manual CMAs"):
            st.session_state.manual_mu_edits = new_vals
            st.rerun()

    st.caption(
        f"μ method in effect: **{mu_method}**. "
        f"Jorion shrinkage intensity = {mu_jor.meta['shrinkage']:.3f} "
        f"(higher = more shrinkage toward MVP). "
        f"Black-Litterman δ = {bl_delta:.1f}."
    )

    # ---- Black-Litterman with Views ---------------------------------------
    st.subheader("Black-Litterman with Views")
    st.caption(
        "Express absolute or relative views and blend them with the BL "
        "equilibrium prior. Absolute view: 'asset A returns X% annually'. "
        "Relative view: 'asset A returns X% MORE than asset B annually'."
    )

    if "bl_views" not in st.session_state:
        st.session_state.bl_views = pd.DataFrame([
            {"asset": codes[0], "view_type": "absolute", "vs_asset": "",
             "expected_return_pct": 8.0, "confidence_pct": 50.0},
        ])
    asset_options = list(codes)
    edited_views = st.data_editor(
        st.session_state.bl_views,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "asset": st.column_config.SelectboxColumn(
                "Asset", options=asset_options, required=True,
            ),
            "view_type": st.column_config.SelectboxColumn(
                "View type", options=["absolute", "relative"], required=True,
            ),
            "vs_asset": st.column_config.SelectboxColumn(
                "Vs (relative only)", options=[""] + asset_options,
            ),
            "expected_return_pct": st.column_config.NumberColumn(
                "Expected return (% annual)", min_value=-50.0, max_value=50.0, step=0.5,
            ),
            "confidence_pct": st.column_config.NumberColumn(
                "Confidence (%)", min_value=1.0, max_value=99.0, step=5.0,
            ),
        },
        key="bl_views_editor",
    )
    st.session_state.bl_views = edited_views

    if st.button("Apply views to BL μ"):
        # Build P, q, Ω from the views table.
        views = edited_views.dropna(subset=["asset", "expected_return_pct"])
        views = views[views["asset"].isin(codes)]
        if views.empty:
            st.warning("No usable views.")
        else:
            n = len(codes)
            rows: list[np.ndarray] = []
            qs: list[float] = []
            confs: list[float] = []
            for _, row in views.iterrows():
                p = np.zeros(n)
                a_idx = codes.index(row["asset"])
                p[a_idx] = 1.0
                q_annual = float(row["expected_return_pct"]) / 100.0
                if row["view_type"] == "relative" and row.get("vs_asset") in codes:
                    b_idx = codes.index(row["vs_asset"])
                    p[b_idx] = -1.0
                # De-annualize to monthly
                q_monthly = (1.0 + q_annual) ** (1.0 / 12) - 1.0
                rows.append(p)
                qs.append(q_monthly)
                confs.append(max(0.01, min(0.99, float(row["confidence_pct"]) / 100.0)))
            P = np.vstack(rows)
            q = np.array(qs)
            # Map confidence ∈ (0, 1) to view-uncertainty Ω. High confidence
            # = low Ω. Take diag(τ · P Σ Pᵀ) then scale by (1-c)/c.
            tau = 0.05
            base_omega = np.diag(np.diag(P @ (tau * cov.values) @ P.T))
            scale = np.array([(1.0 - c) / c for c in confs])
            omega = base_omega * np.diag(scale)
            mu_views = blend_bl_with_views(mu_bl, cov, P, q, omega=omega, tau=tau)

            cmp = pd.DataFrame({
                "name": [BY_CODE[c].name for c in codes],
                "BL equilibrium (% annual)": (mu_bl.mu_annual * 100).round(2).values,
                "BL + views (% annual)": (mu_views.mu_annual * 100).round(2).values,
                "Δ (pp)": ((mu_views.mu_annual - mu_bl.mu_annual) * 100).round(2).values,
            }, index=codes)
            st.dataframe(cmp, use_container_width=True)
            st.caption(
                "Active μ shown vs equilibrium μ. To use BL+views as the "
                "main μ, set μ method = 'BL equilibrium' and rely on this "
                "tab for what-if analysis (views are not yet wired into the "
                "main solver to keep the headline solve stable)."
            )


# --- Covariance ------------------------------------------------------------


with tab_cov:
    st.subheader(f"Covariance — {cov_res.method}")
    c1, c2, c3 = st.columns(3)
    c1.metric("N obs", f"{cov_res.n_obs:,}")
    c2.metric("Shrinkage λ",
              f"{cov_res.shrinkage:.3f}" if cov_res.shrinkage is not None else "—")
    c3.metric("Condition number", f"{condition_number(cov):.1f}")

    st.subheader("Correlation matrix")
    corr = cov_to_corr(cov)
    fig = px.imshow(
        corr.values, x=corr.columns, y=corr.index,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=".2f", aspect="auto",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Annualized σ per asset")
    sigma_ann = np.sqrt(np.diag(cov.values)) * np.sqrt(12)
    sig_df = pd.DataFrame({
        "code": codes,
        "name": [BY_CODE[c].name for c in codes],
        "annual σ (%)": (sigma_ann * 100).round(2),
    })
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

    if cov_method == "stress_blended" and cov_res.mask is not None:
        n_stress = int(cov_res.mask.sum())
        n_normal = int((~cov_res.mask).sum())
        st.caption(
            f"Stress regime: {n_stress} months (SPX in ≥{int(dd_threshold*100)}% drawdown), "
            f"Normal regime: {n_normal} months, "
            f"Blend weight p = {stress_weight:.2f}."
        )


# --- Optimize (efficient frontier + CML) -----------------------------------


with tab_opt:
    st.subheader("Efficient frontier and Capital Market Line")
    if frontier is None:
        st.error(f"Frontier failed: {frontier_err}")
    else:
        ap = asset_points(mu_res.mu, cov)
        cml_x, cml_y = cml_points(frontier)

        fig = go.Figure()

        # Frontier
        fig.add_trace(go.Scatter(
            x=frontier.vols * 100, y=frontier.returns * 100,
            mode="lines+markers",
            line=dict(color="rgba(120,120,120,0.8)", width=2),
            marker=dict(size=4, color="rgba(120,120,120,0.8)"),
            name="Efficient frontier",
            hovertemplate="σ=%{x:.2f}%<br>μ=%{y:.2f}%<extra></extra>",
        ))

        # CML
        fig.add_trace(go.Scatter(
            x=cml_x * 100, y=cml_y * 100,
            mode="lines",
            line=dict(color=GOLD, width=2, dash="dash"),
            name=f"CML (slope = Sharpe {frontier.tangent.sharpe:.2f})",
            hovertemplate="σ=%{x:.2f}%<br>μ=%{y:.2f}%<extra>CML</extra>",
        ))

        # Risk-free anchor
        fig.add_trace(go.Scatter(
            x=[0], y=[rf_annual * 100],
            mode="markers+text",
            marker=dict(size=10, color="rgba(100,100,100,0.9)", symbol="circle"),
            text=["r_f"], textposition="bottom right",
            showlegend=False,
            hovertemplate=f"r_f = {rf_annual*100:.2f}%<extra></extra>",
        ))

        # Per-asset points
        fig.add_trace(go.Scatter(
            x=ap["vol"] * 100, y=ap["ret"] * 100,
            mode="markers+text",
            marker=dict(size=10, color="rgba(70,130,180,0.6)", symbol="circle",
                        line=dict(color="#333", width=1)),
            text=[BY_CODE[c].name for c in ap["code"]],
            textposition="top center",
            textfont=dict(size=10),
            name="Assets",
            hovertemplate="%{text}<br>σ=%{x:.2f}%<br>μ=%{y:.2f}%<extra></extra>",
        ))

        # Benchmark
        fig.add_trace(go.Scatter(
            x=[bm_vol_m * np.sqrt(12) * 100],
            y=[((1 + bm_ret_m) ** 12 - 1) * 100],
            mode="markers+text",
            marker=dict(size=14, color="rgba(60,60,60,0.8)", symbol="square",
                        line=dict(color="white", width=2)),
            text=["60/40"], textposition="top right",
            name="Benchmark",
            hovertemplate=f"60/40 benchmark<extra></extra>",
        ))

        # Tangent portfolio
        fig.add_trace(go.Scatter(
            x=[frontier.tangent.volatility * 100],
            y=[frontier.tangent.exp_return * 100],
            mode="markers+text",
            marker=dict(size=20, color=GOLD, symbol="star",
                        line=dict(color=CRIMSON, width=2)),
            text=["tangent"], textposition="bottom center",
            textfont=dict(color=CRIMSON, size=12),
            name=f"Tangent (Sharpe {frontier.tangent.sharpe:.2f})",
        ))

        # Solved portfolio (may differ from tangent if obj != max_sharpe)
        if obj != "max_sharpe":
            fig.add_trace(go.Scatter(
                x=[sol.volatility * 100],
                y=[sol.exp_return * 100],
                mode="markers+text",
                marker=dict(size=20, color=CRIMSON, symbol="diamond",
                            line=dict(color=GOLD, width=2)),
                text=[obj], textposition="top center",
                textfont=dict(color=GOLD, size=12),
                name=f"Solved ({obj})",
            ))

        fig.update_layout(
            height=560,
            xaxis_title="Volatility (% annual)",
            yaxis_title="Expected return (% annual)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Gold star = tangent portfolio (max Sharpe). Dashed gold line = CML "
            "(slope = tangent Sharpe). Grey line = efficient frontier. Blue dots = "
            "individual assets. Grey square = 60/40 benchmark."
            + ("" if obj == "max_sharpe"
               else "  Crimson diamond = your solved portfolio under the current objective.")
        )

        # Weights table across the frontier
        with st.expander("Frontier weights table", expanded=False):
            w_df = pd.DataFrame(
                [w.values for w in frontier.weights],
                columns=codes,
            )
            w_df.insert(0, "σ (%)", np.round(frontier.vols * 100, 2))
            w_df.insert(1, "μ (%)", np.round(frontier.returns * 100, 2))
            w_df = w_df.round(3)
            st.dataframe(w_df, use_container_width=True, hide_index=True)

    # Bootstrap CIs
    st.subheader("Bootstrap confidence intervals on weights")
    with st.expander("Run bootstrap (slow — 200 draws × full re-solve)", expanded=False):
        n_boot = st.slider("Bootstrap draws", 50, 500, 200, 50, key="n_boot")
        block = st.slider("Expected block length (months)", 1, 24, 6, 1, key="block_len")
        ci_pct = st.slider("Confidence (%)", 60, 99, 90, 1, key="ci_pct") / 100.0
        if st.button("Run bootstrap"):
            with st.spinner("Bootstrapping..."):
                # Use the same optimizer chosen in the sidebar
                opt_map = {
                    "max_sharpe":  lambda **kw: max_sharpe(**kw),
                    "max_ret_te":  lambda **kw: max_return_at_te(te_cap_annual=te_cap, **kw),
                    "max_ret_vol": lambda **kw: max_return_at_vol(vol_cap_annual=vol_cap, **kw),
                    "min_var":     lambda **kw: min_variance(**kw),
                    "risk_parity": lambda **kw: risk_parity(**kw),
                    "hrp":         lambda **kw: hrp(**kw),
                }
                boot_fn = opt_map.get(obj, lambda **kw: max_sharpe(**kw))
                boot_df = bootstrap_weights(
                    rets, boot_fn, cons,
                    rf_annual=rf_annual, n_boot=n_boot, block=block,
                )
                ci_df = weight_ci(boot_df, ci=ci_pct)
                ci_df["name"] = [BY_CODE[c].name for c in ci_df.index]
                ci_display = ci_df[["name"] + [c for c in ci_df.columns if c != "name"]].round(3)
                st.dataframe(ci_display, use_container_width=True)
                st.caption(
                    f"Stationary-block bootstrap with {n_boot} draws, "
                    f"expected block length = {block} months. Wider intervals = "
                    "more sensitivity to the sample path."
                )


# --- Backtest --------------------------------------------------------------


with tab_backtest:
    st.subheader("Walk-forward backtest")
    st.caption(
        "Rolling-window estimation, monthly rebal. Pro-rates per window: "
        "an asset enters the optimization only once it has the chosen "
        "minimum history. Methods compared head-to-head — pick what to "
        "include below. Use the unbalanced ('extended') panel via the "
        "long-term-model toggle in the sidebar to study pre-2007 history."
    )

    cb_lookback = st.slider(
        "Look-back (months)", 24, 120, 60, 6, key="bt_lookback",
        help="Window of monthly returns used to estimate Σ and μ at each rebal.",
    )
    cb_methods_all = {
        "LW + max-Sharpe":     {"label": "LW + max-Sharpe",
                                 "cov_method": "ledoit_wolf",
                                 "mu_method": "historical",
                                 "optimizer": "max_sharpe"},
        "OAS + max-Sharpe":    {"label": "OAS + max-Sharpe",
                                 "cov_method": "oas",
                                 "mu_method": "historical",
                                 "optimizer": "max_sharpe"},
        "Stress-blend + max-Sharpe": {
            "label": "Stress-blend + max-Sharpe",
            "cov_method": "stress_blended",
            "mu_method": "historical",
            "optimizer": "max_sharpe",
            "cov_kwargs": {"equity_col": "us_eq",
                           "dd_threshold": dd_threshold,
                           "stress_weight": stress_weight}
        },
        "HRP":                 {"label": "HRP",
                                 "cov_method": "ledoit_wolf",
                                 "mu_method": "historical",
                                 "optimizer": "hrp"},
        "Risk parity (ERC)":   {"label": "Risk parity (ERC)",
                                 "cov_method": "ledoit_wolf",
                                 "mu_method": "historical",
                                 "optimizer": "risk_parity"},
    }
    selected = st.multiselect(
        "Methods to compare",
        list(cb_methods_all.keys()),
        default=list(cb_methods_all.keys()),
    )

    run_bt = st.button("Run walk-forward")
    if run_bt:
        # Use the *unbalanced* extended frame so pro-rating can do its job.
        bt_panel = rets_full
        if bt_panel.shape[0] < cb_lookback + 12:
            st.error(
                f"Panel has {bt_panel.shape[0]} months; need at least "
                f"{cb_lookback + 12}. Try a shorter look-back or extend "
                "history via the sidebar toggle."
            )
        else:
            specs = [cb_methods_all[m] for m in selected]
            with st.spinner("Running walk-forward..."):
                results = compare_methods(
                    bt_panel, cons, specs,
                    lookback_months=cb_lookback, rf_annual=rf_annual,
                )
            grid = metric_grid(results)
            st.subheader("Method comparison")
            st.dataframe(grid, use_container_width=True, hide_index=True)

            # Two-column layout: cumulative wealth on left, stacked area on right
            colL, colR = st.columns([3, 2])
            with colL:
                st.subheader("Cumulative wealth ($1 → growth)")
                fig = go.Figure()
                for label, res in results.items():
                    if res is None:
                        continue
                    fig.add_trace(go.Scatter(
                        x=res.cum_wealth.index, y=res.cum_wealth.values,
                        mode="lines", name=label,
                    ))
                fig.update_layout(
                    height=480, yaxis_type="log",
                    yaxis_title="Wealth (log scale)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

            with colR:
                # Stacked-area weights for the headline method (first selected)
                live = [(label, res) for label, res in results.items()
                        if res is not None]
                if live:
                    headline_label, headline_res = live[0]
                    st.subheader(f"Weights evolution — {headline_label}")
                    wh = headline_res.weights_history.fillna(0.0)
                    # Asset color palette — keep stable
                    assets = list(wh.columns)
                    palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
                    fig = go.Figure()
                    for i, c in enumerate(assets):
                        name = BY_CODE[c].name if c in BY_CODE else c
                        fig.add_trace(go.Scatter(
                            x=wh.index, y=wh[c].values * 100,
                            mode="lines", stackgroup="one",
                            name=name,
                            line=dict(color=palette[i % len(palette)], width=0.5),
                            hovertemplate=f"{name}<br>%{{x|%Y-%m}}: %{{y:.1f}}%<extra></extra>",
                        ))
                    fig.update_layout(
                        height=480, yaxis_title="Weight (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Active-asset set per window", expanded=False):
                        rows = []
                        for win in headline_res.windows:
                            rows.append({
                                "rebal": win.rebal_date.date(),
                                "n_active": len(win.active_assets),
                                "active": ", ".join(win.active_assets),
                            })
                        st.dataframe(pd.DataFrame(rows),
                                     use_container_width=True, hide_index=True)
            st.caption(
                "Pro-rating: assets without enough history at a given rebal "
                "drop out and the constraints renormalize automatically. "
                "Group caps remain literal (a 90% equity cap stays 90% on "
                "the active universe)."
            )


# --- Stress ----------------------------------------------------------------


with tab_stress:
    st.subheader("Stress scenario — re-solve under a stress-blended Σ")
    st.caption(
        "Computes a stress Σ from SPX-drawdown months only, re-solves the "
        "same objective, and shows how the weights drift. Use this to see "
        "whether your portfolio's composition is robust to correlation spikes."
    )
    c1, c2 = st.columns(2)
    with c1:
        stress_dd = st.slider("Drawdown threshold (%)", 5, 30, 10, 1, key="s_dd") / 100.0
    with c2:
        stress_p = st.slider("Stress weight (p)", 0.0, 1.0, 0.70, 0.05, key="s_p",
                             help="p=1 -> crisis Σ only; p=0 -> full-sample LW")
    if "us_eq" not in codes and codes[0] != "us_eq":
        st.warning("Stress requires an equity column. Using the first column as a proxy.")
    eq_col = "us_eq" if "us_eq" in codes else codes[0]
    stress_cov = stress_blended_cov(
        rets, equity_col=eq_col, dd_threshold=stress_dd, stress_weight=stress_p,
    )

    # Re-solve under stress
    def _solve_with_cov(cv):
        if obj == "max_sharpe":
            return max_sharpe(cv.cov, mu_res.mu, cons, rf_annual=rf_annual)
        if obj == "max_ret_te":
            return max_return_at_te(cv.cov, mu_res.mu, te_cap, cons, rf_annual=rf_annual)
        if obj == "max_ret_vol":
            return max_return_at_vol(cv.cov, mu_res.mu, vol_cap, cons, rf_annual=rf_annual)
        if obj == "min_var":
            return min_variance(cv.cov, mu_res.mu, cons, rf_annual=rf_annual)
        if obj == "risk_parity":
            return risk_parity(cv.cov, mu_res.mu, cons, rf_annual=rf_annual)
        if obj == "hrp":
            return hrp(cv.cov, mu_res.mu, cons, rf_annual=rf_annual)
        if obj == "min_cvar":
            return min_cvar(rets, mu_res.mu, cv.cov, cons, alpha=cvar_alpha, rf_annual=rf_annual)
        return max_sharpe(cv.cov, mu_res.mu, cons, rf_annual=rf_annual)

    try:
        sol_stress = _solve_with_cov(stress_cov)
    except Exception as e:
        st.error(f"Stress solve failed: {e}")
        sol_stress = None

    if sol_stress is not None:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Stress exp. return", f"{sol_stress.exp_return*100:.2f}%",
                   delta=f"{(sol_stress.exp_return - sol.exp_return)*100:+.2f}pp")
        cc2.metric("Stress volatility", f"{sol_stress.volatility*100:.2f}%",
                   delta=f"{(sol_stress.volatility - sol.volatility)*100:+.2f}pp")
        cc3.metric("Stress Sharpe", f"{sol_stress.sharpe:.2f}",
                   delta=f"{sol_stress.sharpe - sol.sharpe:+.2f}")

        delta_df = pd.DataFrame({
            "asset": [BY_CODE[c].name for c in codes],
            "base weight (%)": (sol.weights.reindex(codes).values * 100).round(2),
            "stress weight (%)": (sol_stress.weights.reindex(codes).values * 100).round(2),
        })
        delta_df["Δ (pp)"] = (delta_df["stress weight (%)"] - delta_df["base weight (%)"]).round(2)
        st.subheader("Weight drift: base vs stress")
        st.dataframe(delta_df, use_container_width=True, hide_index=True)

        # Bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=delta_df["asset"], y=delta_df["base weight (%)"],
            name="Base", marker_color="rgba(70,130,180,0.7)",
        ))
        fig.add_trace(go.Bar(
            x=delta_df["asset"], y=delta_df["stress weight (%)"],
            name="Stress", marker_color=CRIMSON,
        ))
        fig.update_layout(
            height=420, barmode="group",
            yaxis_title="Weight (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        if stress_cov.mask is not None:
            n_s = int(stress_cov.mask.sum())
            st.caption(
                f"Stress regime: {n_s} months of history "
                f"(SPX ≥{int(stress_dd*100)}% drawdown, using {eq_col}). "
                f"Blend p = {stress_p:.2f}."
            )


# --- Methodology -----------------------------------------------------------


with tab_method:
    st.subheader("Methodology")
    methodology_path = Path(__file__).parent / "METHODOLOGY.md"
    if methodology_path.exists():
        st.markdown(methodology_path.read_text())
    else:
        st.info("METHODOLOGY.md not found.")
