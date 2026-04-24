"""Expected-return estimators.

Four paths offered:

1. **Historical mean**. Arithmetic mean of monthly returns, annualized.
   Noisy and biased toward whatever the last 20 years happened to reward
   — included only as a reference.
2. **Black-Litterman equilibrium** (reverse-optimized). Given market-cap
   weights `w_mkt` and a risk-aversion coefficient `δ`, the equilibrium
   *excess* return is `π = δ · Σ · w_mkt`. This is the "prior" μ that the
   market would have to hold for those weights to be optimal under MVO.
   A robust anchor when you have no strong views. We approximate market-cap
   weights with the selected universe's default benchmark + equal-weight
   for anything else.
3. **Jorion shrinkage** (Jorion 1986). Shrinks the historical mean vector
   toward a grand mean (the minimum-variance-portfolio return) with a
   closed-form shrinkage intensity. Cheap and effective at killing the
   worst noise in μ.
4. **Manual CMAs**. User pastes annualized expected returns per asset,
   optionally sourced from a CMA publication (JPM / BlackRock / GMO /
   Research Affiliates). Compose annualizes or de-annualizes internally so
   the rest of the math stays in monthly units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


#: Years → months scalar.
MONTHS = 12


@dataclass
class MuResult:
    mu: pd.Series          # monthly expected returns
    mu_annual: pd.Series   # annualized
    method: str
    meta: dict


def historical_mean(rets: pd.DataFrame) -> MuResult:
    mu_m = rets.mean()
    return MuResult(
        mu=mu_m, mu_annual=(1.0 + mu_m) ** MONTHS - 1.0,
        method="historical", meta={"n_obs": len(rets)},
    )


def jorion_shrinkage(rets: pd.DataFrame) -> MuResult:
    """Shrink each asset's historical mean toward the MVP return. Shrinkage
    intensity follows Jorion (1986):
        λ = (N + 2) / ((N + 2) + T · (μ̂ - μ_mvp)' Σ⁻¹ (μ̂ - μ_mvp))
    """
    mu_hat = rets.mean().values
    S = rets.cov().values
    T, N = rets.shape
    ones = np.ones(N)
    S_inv = np.linalg.pinv(S)
    w_mvp = S_inv @ ones / (ones @ S_inv @ ones)
    mu_mvp = float(w_mvp @ mu_hat)

    diff = mu_hat - mu_mvp
    num = N + 2
    den = num + T * float(diff @ S_inv @ diff)
    lam = num / den if den > 0 else 1.0
    mu_shrunk = lam * mu_mvp + (1.0 - lam) * mu_hat

    mu_s = pd.Series(mu_shrunk, index=rets.columns)
    return MuResult(
        mu=mu_s, mu_annual=(1.0 + mu_s) ** MONTHS - 1.0,
        method="jorion",
        meta={"shrinkage": float(lam), "mu_mvp": mu_mvp, "n_obs": T},
    )


def black_litterman_equilibrium(
    cov: pd.DataFrame,
    market_weights: dict[str, float],
    risk_aversion: float = 3.0,
    risk_free_monthly: float = 0.0,
) -> MuResult:
    """Reverse-optimized equilibrium μ: π = δ · Σ · w_mkt (in *excess*
    return terms). Returns total return μ = π + r_f."""
    codes = list(cov.columns)
    w = np.array([market_weights.get(c, 0.0) for c in codes])
    if w.sum() <= 0:
        w = np.ones(len(codes)) / len(codes)
    else:
        w = w / w.sum()
    pi = risk_aversion * cov.values @ w
    mu = pi + risk_free_monthly
    mu_s = pd.Series(mu, index=codes)
    return MuResult(
        mu=mu_s, mu_annual=(1.0 + mu_s) ** MONTHS - 1.0,
        method="black_litterman",
        meta={"risk_aversion": risk_aversion,
              "weights": {c: float(ww) for c, ww in zip(codes, w)}},
    )


def manual_mu(annual_mu_by_code: dict[str, float], codes: list[str]) -> MuResult:
    """User-supplied annualized returns, de-annualized to monthly."""
    mu_ann = pd.Series([annual_mu_by_code.get(c, 0.0) for c in codes], index=codes)
    mu_m = (1.0 + mu_ann) ** (1.0 / MONTHS) - 1.0
    return MuResult(
        mu=mu_m, mu_annual=mu_ann,
        method="manual", meta={"n_missing": sum(c not in annual_mu_by_code for c in codes)},
    )


def blend_bl_with_views(
    equilibrium: MuResult,
    cov: pd.DataFrame,
    P: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray | None = None,
    tau: float = 0.05,
) -> MuResult:
    """Full Black-Litterman posterior when user has views.

    P: (k, N) view matrix
    q: (k,) view values (monthly expected returns implied by each view)
    omega: (k, k) view uncertainty; defaults to diag(τ · P Σ Pᵀ)
    tau: scale factor on the prior covariance (0.025–0.1 is typical)
    """
    codes = list(cov.columns)
    Sigma = cov.values
    pi = equilibrium.mu.reindex(codes).values
    tau_S = tau * Sigma
    if omega is None:
        omega = np.diag(np.diag(P @ tau_S @ P.T))
    tau_S_inv = np.linalg.pinv(tau_S)
    omega_inv = np.linalg.pinv(omega)
    A = tau_S_inv + P.T @ omega_inv @ P
    b = tau_S_inv @ pi + P.T @ omega_inv @ q
    mu_post = np.linalg.solve(A, b)
    mu_s = pd.Series(mu_post, index=codes)
    return MuResult(
        mu=mu_s, mu_annual=(1.0 + mu_s) ** MONTHS - 1.0,
        method="black_litterman_with_views",
        meta={"n_views": int(P.shape[0]), "tau": tau},
    )
