## What Compose does

Compose is a strategic asset allocation workbench. You pick an asset
universe, choose how to estimate expected returns (μ) and covariance (Σ),
pick an optimization objective, and Compose solves for portfolio weights
that satisfy your constraints. The same core routines also trace the
**efficient frontier** and the **Capital Market Line** so you can see where
your solved portfolio sits relative to every feasible alternative.

## The pipeline, step by step

1. **Load prices** from the cached Yahoo panel (or an uploaded CSV).
2. **Compute monthly returns.** Equities, bonds, commodities, gold, and
   REITs are converted to month-end log returns. Cash is special: the
   Yahoo `^IRX` series is the 13-week T-bill *yield* in percent, which we
   convert to a daily return (≈ yield / 252) and compound to a monthly
   return.
3. **Estimate μ.** Four paths — historical, Jorion shrinkage,
   Black-Litterman equilibrium, or user-supplied CMAs.
4. **Estimate Σ.** Four paths — sample, Ledoit-Wolf (default), OAS, or
   stress-blended.
5. **Optimize.** One of seven objectives — Max Sharpe (default),
   Max Return s.t. TE / vol, Min Variance, Risk Parity, HRP, Min-CVaR,
   or Resampled Efficiency.
6. **Report.** Solved weights, annualized return / volatility / Sharpe,
   per-asset risk contributions, the efficient frontier with the tangent
   portfolio starred, and optional bootstrap confidence intervals on the
   weights.

## Covariance estimators

### Sample covariance

`Σ̂ = 1/(T-1) · Σ (r_t - r̄)(r_t - r̄)ᵀ`.

Unbiased but noisy. With `N` assets and `T` observations, sample Σ has
`N(N+1)/2` free parameters — noise dominates unless `T >> N²`. At monthly
frequency, 10 assets ≈ 55 parameters and 18 years ≈ 216 obs, which is
about the lower edge of "reliable." Compose includes sample Σ as a
reference; you should almost never use it as the production estimator.

### Ledoit-Wolf shrinkage (**default**)

Shrinks the sample toward a scaled identity target `F = μ_hat · I`:

`Σ_LW = λF + (1-λ)Σ̂`

where `λ ∈ [0, 1]` is chosen to minimize the Frobenius-norm distance
between `Σ_LW` and the true (unobserved) covariance. Ledoit & Wolf (2004)
derive a closed-form expression for the optimal `λ` that does not require
any tuning. Implemented by `sklearn.covariance.LedoitWolf`.

**Why it's the default**: across a wide range of sample sizes, assets, and
return distributions, it dominates the sample estimator in out-of-sample
performance while remaining cheap and interpretable. The reported
shrinkage intensity `λ` is a useful single-number read on "how noisy is
this sample": λ near 1 means the sample is worthless and you're basically
using equal variance; λ near 0 means the sample is fine on its own.

### OAS shrinkage

Chen, Wiesel, Eldar & Hero (2010) derived an alternative closed-form
shrinkage intensity under a Gaussian assumption. In practice it tends to
be very close to Ledoit-Wolf and sometimes slightly better when `N` is
small and returns are approximately Gaussian. Exposed as a toggle for
users who want the comparison.

### Stress-blended covariance

Splits history into two regimes using SPX drawdown:
- **Normal**: months where SPX cumulative drawdown from its running peak
  is better than `-dd_threshold` (default 10%).
- **Stress**: months with drawdown `≤ -dd_threshold`.

Each block gets its own shrunk covariance; the returned Σ is a weighted
blend `(1-p) · Σ_normal + p · Σ_stress`. With `p = 0` you get full-sample
LW; with `p = 1` you assume you're *in* a crisis right now. The default
of `p = 0.3` mildly penalizes the correlation structure that only shows
up in benign markets.

Falls back to full-sample LW if either regime has fewer than 24 obs.

## Expected-return estimators

### Historical mean

`μ̂_i = (1/T) Σ r_{i,t}`. Noisy in the extreme — the standard error on a
20-year monthly mean is roughly `σ / √240 ≈ 0.015 · σ`. For an asset with
15% annual volatility, that's ~1%/yr of error on the mean estimate alone.
Shown as a reference and should not be used raw in production MVO.

### Jorion shrinkage

Jorion (1986) shrinks the vector of historical means toward the
minimum-variance-portfolio return `μ_MVP`:

`μ_shrunk = λ · μ_MVP · 𝟙 + (1-λ) · μ̂`

with shrinkage intensity

`λ = (N+2) / ((N+2) + T · (μ̂-μ_MVP)ᵀ Σ⁻¹ (μ̂-μ_MVP))`

This pushes all assets toward a common mean, which damps the
error-maximizing behavior of MVO without losing the relative ordering of
expected returns.

### Black-Litterman equilibrium (reverse-optimized)

If the market is at equilibrium and investors hold market-cap weights
`w_mkt`, then the market's implied *excess* returns satisfy

`π = δ · Σ · w_mkt`

where `δ` is a risk-aversion coefficient (typically 2–4 for equities; we
default to 3). This is the μ that would have to be true for MVO to
produce `w_mkt`. Compose approximates `w_mkt` for the selected universe
with a default benchmark (60/40 SPX/AGG for Tier 0–2, equal-weight
otherwise). The result is a much better *prior* on μ than the historical
mean — it embeds the collective wisdom of the market without requiring
any forecast.

The full BL posterior (with user views via P and q) is implemented in
`blend_bl_with_views` but is not yet surfaced in the UI.

### Manual CMAs

Users paste annualized expected returns per asset, usually from a
published Capital Market Assumptions sheet (BlackRock, JPM, GMO,
Research Affiliates). Compose de-annualizes to monthly internally.

## Optimizers

### Max Sharpe (default)

The tangent portfolio on the efficient frontier — the feasible portfolio
with the highest `(μ_P - r_f) / σ_P`. Compose traces the frontier at a
grid of target returns (each sub-problem is a convex QP: min `wᵀΣw` s.t.
`μᵀw ≥ target` and user constraints) and returns the grid point with the
highest Sharpe. This produces the frontier as a byproduct of solving for
the tangent — no wasted computation.

### Max Return s.t. TE

Maximize `μᵀw` subject to `(w - w_B)ᵀ Σ (w - w_B) ≤ TE_max²` and the
usual constraints. This is the information-ratio-maximizing objective for
a benchmarked mandate — "how much return can I wring out without
straying more than `TE_max` from the benchmark?"

### Max Return s.t. Vol

Maximize `μᵀw` subject to `σ_P ≤ vol_max`. Same idea as Max Return s.t.
TE but on absolute rather than benchmark-relative risk.

### Min Variance

Minimize `wᵀΣw` subject to constraints. Defensive; useful when you
don't trust μ at all.

### Risk Parity (equal risk contribution)

Solves for weights such that each asset contributes the same amount to
portfolio volatility:

`w_i · (Σw)_i / √(wᵀΣw) = σ_P / N`  for all `i`.

Implemented with a Bruder-Roncalli fixed-point iteration. Independent of
`μ` by design — it's the allocation that levels *risk* contributions, not
the one that maximizes expected return.

### Hierarchical Risk Parity

López de Prado (2016). Three steps:
1. **Tree clustering**. Compute `d_ij = √(½(1-ρ_ij))` (a true metric on
   correlation), run single-linkage hierarchical clustering.
2. **Quasi-diagonalization**. Reorder assets by the leaf order of the
   tree so that similar assets are adjacent.
3. **Recursive bisection**. Split the reordered covariance in half and
   weight each half inversely to its cluster variance; recurse.

HRP avoids matrix inversion entirely (Σ only appears in diagonal sub-
blocks), which makes it extremely robust at the cost of ignoring μ.
Unlike plain risk parity, HRP puts correlated assets on the same "branch"
and gives the branch a single weight.

### Min-CVaR

Minimize the conditional-value-at-risk at level `α` (default 95%) — the
average of the worst `(1-α)` tail of monthly returns. Rockafellar &
Uryasev (2000) showed this reduces to a linear program:

  min   η + (1/((1-α)T)) · Σ z_t
  s.t.  z_t ≥ -r_tᵀ w - η,  z_t ≥ 0,  sum(w) = 1, box/group/TE constraints.

`η` at the optimum is the VaR. Unlike variance, CVaR is coherent and
naturally captures tail risk — a good choice when "drawdown" is the
constraint you actually care about.

### Resampled Efficiency (Michaud)

Wraps any optimizer with a Monte Carlo loop: simulate T monthly returns
from `MvN(μ, Σ)`, re-estimate `(μ̂, Σ̂)`, solve, collect weights, repeat
`n_sims` times, average the weights. The averaging washes out the
corner-portfolio pathology of MVO — resampled frontiers are smoother and
more diversified than the underlying MVO frontier. Slow (`n_sims × T`
solves) and not theoretically grounded in the strict sense, but widely
used in practice.

## Efficient frontier and CML

The **efficient frontier** is the Pareto set in (σ, μ) space — for each
level of risk, the highest achievable return. Compose traces it by
solving `min_variance_at_target` at a grid of target returns.

The **Capital Market Line** (CML) is the tangent line from the risk-free
rate to the tangent portfolio (the max-Sharpe point):

`r(σ) = r_f + σ · (μ_T - r_f) / σ_T`

Under the CAPM assumptions, every rational investor holds some mix of the
tangent portfolio and the risk-free asset — sliding along the CML. In
practice the CML is a useful overlay because:
- The **slope** of the CML is the tangent portfolio's Sharpe ratio.
- Any portfolio below the CML is dominated (a CML portfolio with the
  same σ offers more return).
- The **tangent point** is where the CML just touches the efficient
  frontier — the "best risk-adjusted" portfolio.

## Diagnostics

### Risk contributions

Decomposes portfolio vol into per-asset contributions:

`RC_i = w_i · (Σw)_i / σ_P`,  with `Σ_i RC_i = σ_P`.

Especially useful for spotting concentration — a portfolio can look
diversified by weights while being 80% driven by a single asset's risk.

### Bootstrap CIs on weights

Stationary-block bootstrap (Politis & Romano 1994) of the monthly return
series with expected block length 6: each bootstrap draw generates a new
sample from which Compose re-estimates `(μ̂, Σ̂)`, re-solves the
optimization, and records the weights. The distribution across draws
gives an honest read on how much the solved weights depend on the
particular sample path — a wide CI on an asset's weight is a sign that
the optimization is on thin ice there.

## References

- Markowitz (1952). *Portfolio Selection.* Journal of Finance.
- Jorion (1986). *Bayes-Stein Estimation for Portfolio Analysis.* JFQA.
- Black & Litterman (1992). *Global Portfolio Optimization.* FAJ.
- Politis & Romano (1994). *The Stationary Bootstrap.* JASA.
- Michaud (1998). *Efficient Asset Management.*
- Rockafellar & Uryasev (2000). *Optimization of Conditional Value-at-Risk.*
- Ledoit & Wolf (2004). *Honey, I Shrunk the Sample Covariance Matrix.* JPM.
- Chen, Wiesel, Eldar & Hero (2010). *Shrinkage Algorithms for MMSE Covariance Estimation.* IEEE TSP.
- Bruder & Roncalli (2012). *Managing Risk Exposures Using the Risk Budgeting Approach.*
- López de Prado (2016). *Building Diversified Portfolios That Outperform Out of Sample.* JPM.
