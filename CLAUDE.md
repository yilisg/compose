# CLAUDE.md — context for future Claude Code sessions on this project

This file briefs a fresh Claude Code instance on **why Compose was built the
way it was**, the design decisions baked into the code, and the conventions
to follow when extending it. README.md is for users; this is for the next
Claude.

## Who this is for

The user (`yilisg`) is a portfolio-construction / asset-allocation
practitioner. They're fluent in quant-finance vocabulary — Sharpe, TE,
Ledoit-Wolf, regime-conditional Σ, efficient frontier — so frame
explanations at that level. They build personal Streamlit tools and
appreciate:

- Minimalist one-word project names with a short italic tagline (`rhyme`
  → "History doesn't repeat, but it rhymes."; `compose` → "Every
  portfolio is a composition.").
- The **language/music family** of metaphors. Sibling projects share an
  aesthetic.
- Reproducible defaults; parquet-cached data so apps work offline.
- An in-app **Methodology** tab rendering a `METHODOLOGY.md` from disk.

## Sibling project: rhyme

`/Users/yili/Desktop/Claude/rhyme/` — historical-analog / regime detection
on US macro + market panels. Compose deliberately mirrors rhyme's
structure (sidebar + tabs + Methodology + gold/crimson accents) so the
two feel like one tool suite. Repo: `https://github.com/yilisg/rhyme`.

A v2 stretch goal mentioned during planning was **importing rhyme's
regime labels into Compose** to drive regime-conditional Σ. As of
April 2026 the regime-aware Σ path is wired up: the SPX-drawdown
bucket is the default labeler, and `compose_lib/regime_label.py`
vendors rhyme's `label_from_z` (pinned to commit 8b79fb3) so the
upstream macro-grid labels can be substituted without `pip install
-e ../rhyme` (which breaks Streamlit Cloud deploys).

## Origin of the project

In April 2026 the user asked for an SAA (Strategic Asset Allocation) tool
modeled on rhyme. The brief:

- Specify a return target and constraints (TE, drawdown, min/max per
  asset class) → solve for optimal weights.
- Default objective: max risk-adjusted return.
- Alternate objective: max return s.t. TE / vol cap.
- Support different correlation matrices including robust shrinkage and
  regime-based ones (risk-on vs risk-off).
- Asset sets at increasing flexibility, starting with the canonical
  3-asset (SPX, AGG, Cash) starter.

After research and brainstorming, we agreed on:

- **Name**: `compose` (musical sibling to `rhyme`; "Every portfolio is
  a composition.").
- **Universes**: Tier 0 (3 assets) → Tier 3 (10 assets); skip the
  factor-tilt and alternatives tiers for v1.
- **Techniques**: include all from the research table — LW (default),
  OAS, BL equilibrium, Resampled (Michaud), HRP, Min-CVaR, bootstrap CIs.
- **Plots**: efficient frontier + CML (both required).
- **Skipped for v1**: rhyme regime-label import (use SPX-drawdown
  bucketing instead).
- **Defaults locked at scaffold time**: monthly returns; TE benchmark =
  60/40 SPX/AGG; light theme; public repo.

## Research summary (SAA best practices that informed the design)

**The core problem.** Choose weights `w` for `n` assets to optimize an
objective subject to constraints. Common objectives:

1. **Max Sharpe** (default in Compose): maximize `(μ'w − r_f) / σ_p`.
2. **Max return s.t. TE/vol cap**: information-ratio maximizing.
3. **Min variance**: defensive, doesn't trust μ.
4. **Risk parity (ERC)**: equalize risk contributions; μ-free.
5. **Min-CVaR**: tail-risk objective.

**Naive MVO is an error-maximizer.** Tiny perturbations in `μ` produce
corner solutions. `Σ` is also noisy when `T` is not >> `N`. Crisis
correlations spike — 60/40 stops diversifying when it matters most.

**The robust toolkit Compose implements:**

| Technique | Fixes | Where in code |
|---|---|---|
| Ledoit-Wolf shrinkage (default) | Σ noise | `covariance.ledoit_wolf_cov` |
| OAS shrinkage | Σ noise (small N, Gaussian) | `covariance.oas_cov` |
| Stress-blended Σ | Crisis correlation underestimate | `covariance.stress_blended_cov` |
| Black-Litterman equilibrium | μ noise; reverse-optimized | `expected_returns.black_litterman_equilibrium` |
| Jorion shrinkage | μ noise; shrinks toward MVP return | `expected_returns.jorion_shrinkage` |
| HRP | Avoids matrix inversion entirely | `optimize.hrp` |
| Min-CVaR (R-U LP) | Tail risk vs variance | `optimize.min_cvar` |
| Resampled (Michaud) | Frontier instability | `optimize.resampled` |
| Bootstrap CIs on weights | Honest uncertainty on solution | `diagnostics.bootstrap_weights` |

**Regime-aware Σ.** Three tiers of sophistication exist; we built tier 1:

1. **Naive stress overlay** (built): split history by SPX drawdown
   threshold (default ≥10%), shrink each block separately, blend with
   weight `p`.
2. **Macro-panel regime classification** (deferred): would import
   rhyme's regime labels and compute Σ per cluster.
3. **Markov-switching / DCC-GARCH** (skipped): elegant but fragile.

**Capital Market Assumptions.** Compose lets users paste annualized μ
manually (CMA path). The intent is to support published CMAs from
BlackRock / JPM / GMO / Research Affiliates — that's how serious SAA
practitioners actually do this. BL equilibrium is the second-best anchor
when no CMA is at hand.

## Architectural conventions

**Layout (mirrors rhyme):**

```
app.py                       # Streamlit UI, all tabs
refresh_prices.py            # Pull Yahoo, write data/default_prices.parquet
                             # (also fetches Vanguard predecessor tickers
                             #  used by the long-term splice)
compose_lib/
  universe.py                # Tier definitions + default benchmark/group bounds
  data_fetch.py              # Yahoo download + parquet load/save
  returns.py                 # Prices → monthly returns (cash special-cased);
                             # SPLICE_MAP + splice_total_return for long history
  covariance.py              # Σ estimators (LW, OAS, EWMA, stress-blended,
                             #  regime-blended)
  expected_returns.py        # μ estimators (incl. blend_bl_with_views)
  optimize.py                # Optimizers, constraints, frontier sub-problems;
                             # subset_constraints helper for pro-rating
  frontier.py                # Frontier + CML
  diagnostics.py             # Risk contributions, bootstrap CIs
  backtest.py                # Walk-forward driver + metric_grid
  regime_label.py            # Vendored rhyme `label_from_z` (commit 8b79fb3)
data/default_prices.parquet  # Cached panel; ships in the repo
.streamlit/config.toml       # Light theme default
METHODOLOGY.md               # Rendered in the Methodology tab
```

**UI patterns** (apply to any new feature):

- Title + italic caption at the top of the sidebar.
- Sidebar uses **`st.expander`** to group parameters; the "Expected
  returns", "Covariance", "Objective" expanders default to expanded.
- Sidebar **panel-source selector** (top): Public / Private / Custom
  upload. Mirrors what rhyme is moving to.
- Sidebar **Long-term-model checkbox** (default OFF). When OFF the
  panel is filtered to 2000-01-01+. When ON, full history is used and
  Vanguard / FRED predecessors are spliced onto the modern ETFs (see
  `SPLICE_MAP`). The unbalanced extended frame is exposed to the
  Backtest tab; the headline solver still uses the dropna'd common
  history to avoid changing default output.
- Main area is **8 tabs**: Overview, Universe, Views (μ), Covariance,
  Optimize, Backtest, Stress, Methodology — last tab is always
  Methodology.
- Overview: 4 `st.metric` tiles top, then weights bar chart, then risk
  contributions table.
- Views (μ): side-by-side comparison + Black-Litterman-with-views editor
  via `st.data_editor` (asset / view_type / vs_asset / expected_return /
  confidence). Builds (P, q, Ω) and calls `blend_bl_with_views`.
- Backtest: walk-forward comparison of LW / OAS / stress-blended / HRP /
  ERC by ann return / vol / Sharpe / max DD / turnover. Cumulative-wealth
  line chart on the left, stacked-area weights evolution on the right
  (consistent palette with Universe tab), active-asset set per window
  in an expander.
- **Color convention**: gold (`#FFD700`) + crimson (`#D10000`) for "the
  current / reference / solved" markers. Greys for history. Blue for
  per-asset points. Reuse the constants from `app.py`.
- Plotly charts; no matplotlib.

**Code patterns:**

- Prefer **dataclasses** for results (`Solution`, `CovResult`, `MuResult`,
  `Frontier`, `Constraints`).
- Cache expensive computations with `@st.cache_data` on the **bytes
  representation of the panel** plus a string key for parameter
  combinations — see `_cached_returns` in `app.py`.
- Convex problems use **cvxpy** (`min_variance`, `max_return_at_*`,
  `min_cvar`); non-convex use scipy or custom iteration (`risk_parity`)
  or recursion (`hrp`).
- Solver fallback: `_solve` in `optimize.py` tries Clarabel → SCS → ECOS.

**Conventions to preserve:**

- All quantities flow as **monthly returns / monthly Σ** internally;
  annualize only at the display layer (`× 12` for returns,
  `× sqrt(12)` for vol).
- Cash gets special treatment — `^IRX` is a yield in percent, not a
  total-return series. See `returns._cash_monthly_return`.
- `Constraints` accepts annualized TE, but converts to monthly internally
  before adding the cvxpy quad-form constraint.
- `Solution` carries `active_assets` (None on the headline solve;
  populated by walk-forward windows). `subset_constraints(cons,
  active)` is the canonical way to renormalize when an asset drops
  out — group caps stay literal, box / benchmark / current_weights
  are filtered and renormalized.
- Common history (legacy path) is enforced by `dropna(how="any")` in
  `compute_monthly_returns(extend_history=False)`. Tier 3's HYG-anchored
  cliff (April 2007) is what motivated the long-term splices: with
  `extend_history=True`, SPY history goes back to 1980 via VFINX,
  AGG to 1986 via VBMFX, etc., and the resulting frame is intentionally
  unbalanced for the Backtest tab to pro-rate over.
- HYG splice uses FRED `BAMLHYH0A0HYM2TRIV`. As of late 2023 FRED's
  ICE BAML total-return series got truncated to ~3 years of history
  (the splice falls back gracefully). If you need pre-2007 HY,
  vendor a longer total-return series elsewhere (Bloomberg, Robert
  Shiller's data set, etc.) and add a leg to `SPLICE_MAP["us_hy"]`.

## Default settings (and why)

| Setting | Default | Rationale |
|---|---|---|
| Panel source | Public (Yahoo + cache) | No setup required; private/custom available |
| Long-term model | OFF (panel from 2000-01-01) | Avoids surprising the new user with mutual-fund splices; flip ON for pre-2000 work |
| Universe | Tier 1 — Global 60/40 (5 assets) | Realistic out-of-the-box demo |
| μ method | Black-Litterman equilibrium | Most stable anchor; avoids the historical-mean trap |
| Σ method | Ledoit-Wolf | User explicitly asked for LW as default |
| Regime-aware Σ | OFF | Adds complexity; available as a toggle |
| EWMA half-life | 12 months | Half a year of memory each direction; reasonable middle-ground |
| Objective | Max Sharpe | User specified |
| BL δ | 3.0 | Standard for diversified mandates |
| Risk-free rate | 3.0% annual | Reasonable steady-state assumption |
| Stress dd | 10% | Captures the standard "correction" threshold |
| Stress p | 0.30 (Σ tab) / 0.70 (Stress tab) | Mild blend for the base solve; aggressive blend for stress test |
| Group caps | Equity ≤90%, Bonds ≤90%, Credit ≤50%, Real ≤30%, Cash ≤30% (editable) | Prevents corner solutions; now user-editable in sidebar |
| Turnover cap | OFF | Optional; vs benchmark when ON |
| Backtest look-back | 60 months | Standard 5-year rolling window |
| Theme | Light | User preference (committed in `.streamlit/config.toml`) |

## Smoke testing pattern

Before pushing UI changes, run a **programmatic Streamlit render** —
faster than spinning up a browser:

```bash
PYTHONPATH=. /Users/yili/Desktop/Claude/rhyme/.venv/bin/python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py', default_timeout=60)
at.run()
print('exceptions:', at.exception)
print('errors:', [e.value for e in at.error])
print('metrics:', [m.label + '=' + str(m.value) for m in at.metric])
"
```

Compose passes this test cleanly with default settings (zero
exceptions, zero errors). The `use_container_width` deprecation warnings
are non-blocking and rhyme has the same.

For library-level smoke tests, the pattern in `compose_lib/` is to
run the full pipeline (load → returns → cov → solve → frontier) on each
tier and check shapes / values. Examples are in earlier conversation
turns.

## Dev environment notes

- The project's **own** venv lives in `.venv/` (gitignored).
  `pip install -r requirements.txt` is sufficient.
- During the initial build I reused **rhyme's venv** (which has nearly
  the same deps) plus a one-off `pip install cvxpy`. That's fine for
  dev iteration but the production / Streamlit Cloud install uses a
  fresh env from `requirements.txt`.
- `cvxpy>=1.4` is the **only new dep vs rhyme**. It pulls in Clarabel,
  SCS, ECOS, OSQP — first cloud build takes ~3 min, then fast.

## Repo + deployment

- GitHub: **https://github.com/yilisg/compose** (public, owned by
  `yilisg`).
- Streamlit Cloud deploy: see README's "Deploy to Streamlit Community
  Cloud" section. Every push to `main` auto-redeploys.
- No secrets needed (Yahoo doesn't require an API key; cached parquet
  ships in the repo).

## Commit conventions

Match rhyme's style:

- **Title Case**, terse, descriptive.
- Use `+` to enumerate inside a single subject ("Cycle tab + robust mode
  + analog backtest").
- No emoji.
- Footer: `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` (per
  Claude Code convention).
- For multi-line commits, use a HEREDOC body. Subject ≤ ~80 chars; body
  bullets explaining the **why**.

## Known caveats / future work

- **Cash μ historical ≈ 1.6%** in the default panel, below the default
  3% rf assumption. Means `min_var` and some `min_cvar` solutions show
  slightly negative Sharpe — that's data, not a bug.
- **Stress-blended fallback**: if either regime has fewer than 24
  monthly observations, the function silently falls back to full-sample
  LW. The `method` field tags this. Don't break the silent fallback —
  it prevents brittle behavior on Tier 0 short windows.
- **BL Views editor** lives in the Views tab and feeds
  `blend_bl_with_views` to display "BL + views" vs "BL equilibrium"
  side-by-side. The headline solver still consumes BL equilibrium so
  the default solve stays stable; making views the active μ would be a
  one-line change but changes everyone's default output, so leave it
  opt-in for now.
- **HYG / FRED splice has shrunk**: as of late 2023 FRED truncated the
  ICE BAML total-return series to ~3 years. The splice still works
  (graceful degrade) but you only get a few years of pre-HYG-inception
  HY history. To recover pre-2007 high-yield total returns you'd need
  to vendor an alternative (Bloomberg HY, etc.) and add a leg.
- **Resampled efficiency is slow**: `n_sims × T` solves. UI default is
  200 sims. Don't bump higher without a progress bar.
- **Bootstrap CIs are opt-in** in the UI (slow — full re-solve per
  draw). Tab is in `Optimize` under an expander.
- **Backtest is opt-in**: needs a button click. Don't auto-run on every
  re-render — it's expensive (one walk-forward per selected method,
  each one of which solves N windows).
- **Per-asset box-bound editor stub**: the sidebar has an "Enable
  per-asset bounds" checkbox but the per-asset editing widget itself
  is not yet surfaced (needs a Views-tab integration to live with the
  asset table). For now, when enabled, it falls back to the uniform
  bounds. Wiring up the per-asset editor inline next to the asset
  table would close that gap.
- **Tier 4–5 universes** (factor sleeves, alternatives) were brainstormed
  but skipped for v1. The `universe.py` design supports adding them
  without breaking changes.

## When in doubt

- Match **rhyme's** patterns. The two projects should feel like a suite.
- The user prefers **action over confirmation** in auto mode but expects
  decisions about visibility (public/private), destructive operations,
  and external API calls to be flagged before being taken.
- Annualize at display, keep math in monthly internally. Don't mix.
- If a new optimization method is added, follow the
  `Solution`-returning convention so the rest of the UI works without
  changes.
