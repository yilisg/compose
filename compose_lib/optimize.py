"""Portfolio optimizers.

All optimizers return a `Solution` object with the weights and a handful of
derived stats. Convex problems (min variance, max return s.t. vol, max return
s.t. TE, min CVaR) are expressed in cvxpy; max Sharpe is computed by tracing
the efficient frontier and picking argmax(Sharpe), which naturally produces
the frontier for plotting; risk parity uses a fixed-point iteration; HRP
follows López de Prado (2016); resampled efficiency is a Monte Carlo
wrapper around any of the above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


MONTHS = 12


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Constraints:
    """Box + group + turnover + TE + sum-to-one. Every field is optional.

    - box: lower/upper bounds per asset code (defaults to [0, 1]).
    - group_bounds: {group_name: (lo, hi)}.
    - group_map: {asset_code: group_name}.
    - turnover: ||w - w_current||_1 ≤ turnover.
    - current_weights: only meaningful with `turnover`.
    - tracking_error: sqrt((w - w_B)' Σ (w - w_B)) ≤ tracking_error (annual).
    - benchmark: {asset_code: weight}.
    - allow_short: if False (default), forces w ≥ 0.
    """
    box: dict[str, tuple[float, float]] = field(default_factory=dict)
    group_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    group_map: dict[str, str] = field(default_factory=dict)
    turnover: float | None = None
    current_weights: dict[str, float] | None = None
    tracking_error: float | None = None         # annualized
    benchmark: dict[str, float] | None = None
    allow_short: bool = False


@dataclass
class Solution:
    weights: pd.Series
    exp_return: float          # annualized
    volatility: float          # annualized
    sharpe: float              # annualized, net of rf
    method: str
    meta: dict = field(default_factory=dict)
    active_assets: list[str] | None = None  # codes used in the optimization
                                             # (subset of full universe when
                                             # pro-rating around missing data)


# ---------------------------------------------------------------------------
# Constraint assembly (cvxpy)
# ---------------------------------------------------------------------------


def _cvx_constraints(w: cp.Variable, codes: list[str], cov: pd.DataFrame,
                     cons: Constraints) -> list:
    out = [cp.sum(w) == 1]
    if not cons.allow_short:
        out.append(w >= 0)

    lb = np.array([cons.box.get(c, (0.0, 1.0))[0] for c in codes])
    ub = np.array([cons.box.get(c, (0.0, 1.0))[1] for c in codes])
    out += [w >= lb, w <= ub]

    for g, (glo, ghi) in cons.group_bounds.items():
        members = [i for i, c in enumerate(codes) if cons.group_map.get(c) == g]
        if not members:
            continue
        out += [cp.sum(w[members]) >= glo, cp.sum(w[members]) <= ghi]

    if cons.turnover is not None and cons.current_weights is not None:
        w_cur = np.array([cons.current_weights.get(c, 0.0) for c in codes])
        out.append(cp.norm(w - w_cur, 1) <= cons.turnover)

    if cons.tracking_error is not None and cons.benchmark is not None:
        w_bm = np.array([cons.benchmark.get(c, 0.0) for c in codes])
        te_m = cons.tracking_error / np.sqrt(MONTHS)  # convert annual TE to monthly
        out.append(cp.quad_form(w - w_bm, cp.psd_wrap(cov.values)) <= te_m ** 2)

    return out


# ---------------------------------------------------------------------------
# Pro-rating helper
# ---------------------------------------------------------------------------


def subset_constraints(cons: Constraints, active: list[str]) -> Constraints:
    """Return a new `Constraints` restricted to the `active` asset codes.

    Group bounds are kept literal — a 90% equity cap on the full universe
    remains a 90% equity cap on the subset (the cap is on the share of
    *the active universe* devoted to equity, which is what the brief asks
    for). Box and benchmark dicts are filtered. Benchmark is renormalized
    to sum to 1 over the active set so TE constraints stay consistent."""
    box = {c: cons.box.get(c, (0.0, 1.0)) for c in active}
    group_map = {c: cons.group_map.get(c) for c in active if cons.group_map.get(c)}
    bench = None
    if cons.benchmark is not None:
        bench_sub = {c: cons.benchmark.get(c, 0.0) for c in active}
        total = sum(bench_sub.values())
        if total > 0:
            bench = {c: v / total for c, v in bench_sub.items()}
        else:
            bench = {c: 1.0 / len(active) for c in active}
    cur = None
    if cons.current_weights is not None:
        cur_sub = {c: cons.current_weights.get(c, 0.0) for c in active}
        total = sum(cur_sub.values())
        cur = {c: v / total for c, v in cur_sub.items()} if total > 0 else cur_sub
    return Constraints(
        box=box,
        group_bounds=dict(cons.group_bounds),  # literal, per docstring
        group_map=group_map,
        turnover=cons.turnover,
        current_weights=cur,
        tracking_error=cons.tracking_error,
        benchmark=bench,
        allow_short=cons.allow_short,
    )


# ---------------------------------------------------------------------------
# Core convex solves
# ---------------------------------------------------------------------------


def _solve(prob: cp.Problem) -> None:
    for solver in (cp.CLARABEL, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return
        except (cp.SolverError, Exception):
            continue
    raise RuntimeError(f"All solvers failed; last status = {prob.status}")


def _stats(w: np.ndarray, mu_m: np.ndarray, cov_m: np.ndarray,
           rf_annual: float = 0.0) -> tuple[float, float, float]:
    mu_ann = float((1.0 + w @ mu_m) ** MONTHS - 1.0)
    vol_ann = float(np.sqrt(w @ cov_m @ w) * np.sqrt(MONTHS))
    sharpe = (mu_ann - rf_annual) / vol_ann if vol_ann > 0 else np.nan
    return mu_ann, vol_ann, sharpe


def min_variance(cov: pd.DataFrame, mu: pd.Series,
                 cons: Constraints, rf_annual: float = 0.0) -> Solution:
    codes = list(cov.columns)
    w = cp.Variable(len(codes))
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov.values))),
        _cvx_constraints(w, codes, cov, cons),
    )
    _solve(prob)
    wv = np.asarray(w.value).ravel()
    mu_a, vol_a, sr = _stats(wv, mu.values, cov.values, rf_annual)
    return Solution(
        weights=pd.Series(wv, index=codes),
        exp_return=mu_a, volatility=vol_a, sharpe=sr,
        method="min_variance",
    )


def min_variance_at_target(cov: pd.DataFrame, mu: pd.Series,
                           target_monthly_return: float,
                           cons: Constraints,
                           rf_annual: float = 0.0) -> Solution:
    """Return the min-variance portfolio whose expected monthly return is
    at least `target_monthly_return`. Used to trace the efficient frontier."""
    codes = list(cov.columns)
    w = cp.Variable(len(codes))
    base = _cvx_constraints(w, codes, cov, cons)
    base.append(mu.values @ w >= target_monthly_return)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov.values))), base,
    )
    _solve(prob)
    wv = np.asarray(w.value).ravel()
    mu_a, vol_a, sr = _stats(wv, mu.values, cov.values, rf_annual)
    return Solution(
        weights=pd.Series(wv, index=codes),
        exp_return=mu_a, volatility=vol_a, sharpe=sr,
        method="min_var_at_target",
        meta={"target_monthly": target_monthly_return},
    )


def max_return_at_vol(cov: pd.DataFrame, mu: pd.Series,
                      vol_cap_annual: float,
                      cons: Constraints,
                      rf_annual: float = 0.0) -> Solution:
    codes = list(cov.columns)
    w = cp.Variable(len(codes))
    base = _cvx_constraints(w, codes, cov, cons)
    vol_m = vol_cap_annual / np.sqrt(MONTHS)
    base.append(cp.quad_form(w, cp.psd_wrap(cov.values)) <= vol_m ** 2)
    prob = cp.Problem(cp.Maximize(mu.values @ w), base)
    _solve(prob)
    wv = np.asarray(w.value).ravel()
    mu_a, vol_a, sr = _stats(wv, mu.values, cov.values, rf_annual)
    return Solution(
        weights=pd.Series(wv, index=codes),
        exp_return=mu_a, volatility=vol_a, sharpe=sr,
        method="max_return_at_vol",
        meta={"vol_cap_annual": vol_cap_annual},
    )


def max_return_at_te(cov: pd.DataFrame, mu: pd.Series,
                     te_cap_annual: float,
                     cons: Constraints,
                     rf_annual: float = 0.0) -> Solution:
    """Max return subject to TE ≤ te_cap_annual vs cons.benchmark."""
    if cons.benchmark is None:
        raise ValueError("max_return_at_te requires cons.benchmark")
    cons2 = Constraints(**{**cons.__dict__, "tracking_error": te_cap_annual})
    codes = list(cov.columns)
    w = cp.Variable(len(codes))
    prob = cp.Problem(
        cp.Maximize(mu.values @ w),
        _cvx_constraints(w, codes, cov, cons2),
    )
    _solve(prob)
    wv = np.asarray(w.value).ravel()
    mu_a, vol_a, sr = _stats(wv, mu.values, cov.values, rf_annual)
    return Solution(
        weights=pd.Series(wv, index=codes),
        exp_return=mu_a, volatility=vol_a, sharpe=sr,
        method="max_return_at_te",
        meta={"te_cap_annual": te_cap_annual},
    )


def max_sharpe(cov: pd.DataFrame, mu: pd.Series, cons: Constraints,
               rf_annual: float = 0.0, n_grid: int = 40) -> Solution:
    """Trace the efficient frontier at `n_grid` target-return levels, return
    the portfolio with the highest (ex-ante, annualized) Sharpe ratio."""
    codes = list(cov.columns)
    rf_m = (1.0 + rf_annual) ** (1.0 / MONTHS) - 1.0
    mu_lo = float(mu.min())
    mu_hi = float(mu.max())
    grid = np.linspace(mu_lo, mu_hi, n_grid)

    pts: list[Solution] = []
    for tgt in grid:
        try:
            pts.append(min_variance_at_target(cov, mu, tgt, cons, rf_annual))
        except RuntimeError:
            continue
    if not pts:
        raise RuntimeError("Efficient frontier is empty — constraints may be infeasible.")

    # Pick best Sharpe
    sharpes = np.array([(p.exp_return - rf_annual) / p.volatility if p.volatility > 0
                        else -np.inf for p in pts])
    best = pts[int(np.argmax(sharpes))]
    best.method = "max_sharpe"
    best.meta = {"n_frontier_points": len(pts), "rf_annual": rf_annual}
    return best


# ---------------------------------------------------------------------------
# Risk parity (equal risk contribution)
# ---------------------------------------------------------------------------


def risk_parity(cov: pd.DataFrame, mu: pd.Series, cons: Constraints,
                rf_annual: float = 0.0,
                max_iter: int = 500, tol: float = 1e-9) -> Solution:
    """Bruder-Roncalli fixed-point iteration for equal-risk-contribution.
    Long-only, ignores box/group constraints (normalizes to sum=1 only).
    Robust for 3-10 assets."""
    codes = list(cov.columns)
    S = cov.values
    n = len(codes)
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_vol = np.sqrt(w @ S @ w)
        mrc = S @ w / port_vol  # marginal risk contribution
        rc = w * mrc            # risk contribution
        target = port_vol / n
        grad = rc - target
        step = 0.5 / n
        w_new = np.maximum(w - step * grad / (port_vol + 1e-12), 1e-6)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    mu_a, vol_a, sr = _stats(w, mu.values, S, rf_annual)
    return Solution(
        weights=pd.Series(w, index=codes),
        exp_return=mu_a, volatility=vol_a, sharpe=sr,
        method="risk_parity",
    )


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (López de Prado 2016)
# ---------------------------------------------------------------------------


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Return the leaf order from a SciPy linkage matrix."""
    tree = to_tree(link, rd=False)
    return tree.pre_order()


def _recursive_bisect(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Inverse-variance-weighted recursive bisection on the already-ordered
    covariance submatrix."""
    n = len(order)
    w = np.ones(n)
    clusters = [list(range(n))]
    while clusters:
        next_clusters = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            var_l = _cluster_var(cov, [order[i] for i in left])
            var_r = _cluster_var(cov, [order[i] for i in right])
            alpha = 1.0 - var_l / (var_l + var_r)
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= (1.0 - alpha)
            next_clusters += [left, right]
        clusters = next_clusters
    return w


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp(cov: pd.DataFrame, mu: pd.Series, cons: Constraints,
        rf_annual: float = 0.0) -> Solution:
    codes = list(cov.columns)
    S = cov.values
    # Correlation-based distance
    s = np.sqrt(np.diag(S))
    corr = S / np.outer(s, s)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = _quasi_diag(link)
    w_ordered = _recursive_bisect(S[np.ix_(order, order)], order)

    # Map back to original asset order
    w = np.zeros(len(codes))
    for slot, orig in enumerate(order):
        w[orig] = w_ordered[slot]
    w = w / w.sum()

    mu_a, vol_a, sr = _stats(w, mu.values, S, rf_annual)
    return Solution(
        weights=pd.Series(w, index=codes),
        exp_return=mu_a, volatility=vol_a, sharpe=sr,
        method="hrp",
        meta={"cluster_order": [codes[i] for i in order]},
    )


# ---------------------------------------------------------------------------
# Min-CVaR (Rockafellar-Uryasev 2000)
# ---------------------------------------------------------------------------


def min_cvar(rets: pd.DataFrame, mu: pd.Series, cov: pd.DataFrame,
             cons: Constraints, alpha: float = 0.95,
             target_monthly_return: float | None = None,
             rf_annual: float = 0.0) -> Solution:
    """Minimize CVaR_α (expected loss conditional on being in the worst
    (1-α) tail), optionally subject to an expected-return floor.

    Rockafellar-Uryasev LP: let losses L_t = -r_tᵀ w, introduce VaR η and
    slack z_t ≥ L_t - η, minimize η + 1/((1-α)T) Σ z_t.
    """
    codes = list(cov.columns)
    R = rets[codes].values  # (T, N)
    T, N = R.shape
    w = cp.Variable(N)
    eta = cp.Variable()
    z = cp.Variable(T, nonneg=True)

    losses = -R @ w - eta
    constraints = _cvx_constraints(w, codes, cov, cons)
    constraints += [z >= losses]
    if target_monthly_return is not None:
        constraints.append(mu.values @ w >= target_monthly_return)

    cvar = eta + cp.sum(z) / ((1.0 - alpha) * T)
    prob = cp.Problem(cp.Minimize(cvar), constraints)
    _solve(prob)
    wv = np.asarray(w.value).ravel()
    mu_a, vol_a, sr = _stats(wv, mu.values, cov.values, rf_annual)
    return Solution(
        weights=pd.Series(wv, index=codes),
        exp_return=mu_a, volatility=vol_a, sharpe=sr,
        method=f"min_cvar_{int(alpha*100)}",
        meta={"alpha": alpha, "cvar_monthly": float(cvar.value)},
    )


# ---------------------------------------------------------------------------
# Resampled efficiency (Michaud 1998 / Michaud & Michaud 2008)
# ---------------------------------------------------------------------------


def resampled(optimizer: Callable[..., Solution],
              mu: pd.Series, cov: pd.DataFrame,
              rets: pd.DataFrame, cons: Constraints,
              n_sims: int = 200, seed: int = 0,
              rf_annual: float = 0.0, **kwargs) -> Solution:
    """Monte Carlo around (μ, Σ): resample T monthly returns per sim from
    a multivariate normal with the given μ and Σ, re-estimate (μ̂, Σ̂), run
    the optimizer, and average the solved weights. Returns a Solution with
    the averaged weights evaluated on the *original* μ and Σ."""
    codes = list(cov.columns)
    rng = np.random.default_rng(seed)
    T = len(rets)
    W = np.zeros((n_sims, len(codes)))
    mu_v, S = mu.values, cov.values
    n_ok = 0
    for i in range(n_sims):
        sim_r = rng.multivariate_normal(mu_v, S, size=T)
        sim_rets = pd.DataFrame(sim_r, columns=codes)
        mu_hat = sim_rets.mean()
        S_hat = sim_rets.cov()
        try:
            sol = optimizer(cov=S_hat, mu=mu_hat, cons=cons, rf_annual=rf_annual, **kwargs)
            W[n_ok] = sol.weights.reindex(codes).values
            n_ok += 1
        except Exception:
            continue
    if n_ok == 0:
        raise RuntimeError("All resampled sims failed")
    w_avg = W[:n_ok].mean(axis=0)
    w_avg = np.maximum(w_avg, 0.0)
    w_avg = w_avg / w_avg.sum()
    mu_a, vol_a, sr = _stats(w_avg, mu_v, S, rf_annual)
    return Solution(
        weights=pd.Series(w_avg, index=codes),
        exp_return=mu_a, volatility=vol_a, sharpe=sr,
        method=f"resampled({optimizer.__name__})",
        meta={"n_ok": n_ok, "n_sims": n_sims},
    )
