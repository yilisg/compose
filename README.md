# Compose

> *Every portfolio is a composition.*

Compose is a strategic asset allocation (SAA) workbench. Pick an asset
universe, pick a way to estimate expected returns and covariance, pick an
objective, and Compose solves for the weights that satisfy your constraints.
It ships with four preset universes (Tier 0 → Tier 3, three to ten assets)
and a cached Yahoo-Finance price panel so the app works offline.

## What it does

1. Loads monthly returns for one of four preset asset universes (or your CSV).
2. Estimates expected returns μ (historical, Black-Litterman equilibrium,
   Jorion shrinkage, or manually entered CMAs).
3. Estimates the covariance matrix Σ (sample, **Ledoit-Wolf shrinkage** by
   default, OAS shrinkage, or a stress-blended Σ from SPX-drawdown periods).
4. Solves for optimal weights under one of several objectives:
   - **Max Sharpe** (default)
   - Max return subject to a tracking-error or volatility cap
   - Min variance
   - Risk parity (equal risk contribution)
   - Hierarchical Risk Parity (López de Prado)
   - Min-CVaR
   - Resampled efficiency (Michaud)
5. Respects box, group, and turnover constraints.
6. Plots the efficient frontier and the Capital Market Line, with the solved
   portfolio starred. Reports risk contributions, bootstrap confidence
   intervals on the weights, and stress-scenario weight drift.

## Run it locally

```bash
git clone https://github.com/yilisg/compose.git
cd compose
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The cached default panel (`data/default_prices.parquet`) ships in the repo,
so the app runs offline out of the box.

## Refresh the price panel

```bash
source .venv/bin/activate
python refresh_prices.py
```

Pulls fresh daily prices from Yahoo Finance for every ticker in every tier
and rewrites `data/default_prices.parquet`.

## Asset universes

| Tier | Size | Assets |
|---|---|---|
| **Tier 0 — Starter** | 3 | US Equity (SPY), US Aggregate Bonds (AGG), Cash (^IRX) |
| **Tier 1 — Global 60/40** | 5 | + International Developed Equity (EFA), Gold (GLD) |
| **Tier 2 — All-Weather starter** | 7 | + Emerging Markets Equity (EEM), Broad Commodities (DBC) |
| **Tier 3 — Diversified Multi-Asset** | 10 | Splits bonds into Treasuries (IEF), IG Credit (LQD), HY (HYG), TIPS (TIP); adds REITs (VNQ) |

## Upload your own returns

In the sidebar, pick **Upload CSV or JSON**. Format:

- First column = date (any parseable format).
- Every other column = either prices (Compose will compute log returns) or
  already-returns (toggle in the sidebar).

## File layout

```
app.py                     # Streamlit UI
refresh_prices.py          # One-shot script to rebuild the cached price panel
compose_lib/
  universe.py              # Tier definitions
  data_fetch.py            # Yahoo fetcher
  returns.py               # Prices -> monthly returns
  covariance.py            # Sample / LW / OAS / stress-blended Σ
  expected_returns.py      # Historical / BL / Jorion / manual μ
  optimize.py              # All optimizers
  frontier.py              # Efficient frontier + CML
  diagnostics.py           # Risk contributions, bootstrap CIs
data/
  default_prices.parquet
METHODOLOGY.md
requirements.txt
```

## References

- Markowitz (1952). *Portfolio Selection.* Journal of Finance.
- Ledoit & Wolf (2004). *Honey, I Shrunk the Sample Covariance Matrix.* JPM.
- Black & Litterman (1992). *Global Portfolio Optimization.* FAJ.
- Michaud & Michaud (2008). *Efficient Asset Management.*
- López de Prado (2016). *Building Diversified Portfolios That Outperform Out of Sample.* JPM.
- Rockafellar & Uryasev (2000). *Optimization of Conditional Value-at-Risk.*
