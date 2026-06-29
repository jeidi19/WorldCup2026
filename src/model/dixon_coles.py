"""Likelihood Dixon-Coles per gol di partite tra nazionali (Issue #4).

Modello
-------
Per una partita tra squadra di casa `h` e ospite `a`:

    λ = exp(α[h] + β[a] + γ · is_home)        gol attesi della casa
    μ = exp(α[a] + β[h])                       gol attesi dell'ospite

con `is_home = 1` se la partita NON è in campo neutro (flag `neutral=False`), 0 altrimenti.

I gol seguono **due Poisson indipendenti** con una correzione **Dixon-Coles** che
"raddrizza" i 4 punteggi bassi correlati:

    τ(0,0) = 1 − λ·μ·ρ
    τ(0,1) = 1 + λ·ρ
    τ(1,0) = 1 + μ·ρ
    τ(1,1) = 1 − ρ
    τ(x,y) = 1   altrimenti

    P(x,y) = τ(x,y; λ, μ, ρ) · Poisson(x; λ) · Poisson(y; μ)

La **negative log-likelihood pesata** è

    NLL(θ) = − Σ_i w_i · log P(x_i, y_i; θ)
           + λ_id · ( mean(α)^2 + mean(β)^2 )

dove `λ_id` è il `identifiability_penalty_strength` dal config: forza media degli α
e β a zero per rimuovere l'invarianza per shift simultaneo. Il floor `tau_floor` su τ
prima del log evita NaN se durante l'ottimizzazione τ scende ≤ 0.

Parametri vettoriali
--------------------
Lo scipy.optimize.minimize lavora con un vettore piatto. Layout:

    params = [α_0, ..., α_{n-1},
              β_0, ..., β_{n-1},
              γ, ρ]                            # lunghezza 2·n_teams + 2

`decode_params(params, n_teams)` e `encode_params(α, β, γ, ρ)` sono gli adattatori.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from src.model.indexing import MatchData

# Indici dei 4 casi speciali della correzione DC: (home_goals, away_goals)
_DC_SPECIAL_CASES: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


# ---------------------------------------------------------------------------
# Tau (correzione DC)
# ---------------------------------------------------------------------------

def tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Correzione Dixon-Coles τ(x, y; λ, μ, ρ) vettorizzata.

    Restituisce un array di shape `(n_matches,)`. È 1.0 ovunque tranne nei 4 casi
    speciali `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`.
    """
    out = np.ones_like(lam, dtype=np.float64)
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)
    out[mask_00] = 1.0 - lam[mask_00] * mu[mask_00] * rho
    out[mask_01] = 1.0 + lam[mask_01] * rho
    out[mask_10] = 1.0 + mu[mask_10] * rho
    out[mask_11] = 1.0 - rho
    return out


# ---------------------------------------------------------------------------
# Encode/decode dei parametri
# ---------------------------------------------------------------------------

def n_params(n_teams: int) -> int:
    return 2 * n_teams + 2


def encode_params(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: float,
    rho: float,
) -> np.ndarray:
    """Costruisce il vettore piatto `params` dalle componenti."""
    if alpha.shape != beta.shape:
        raise ValueError(f"alpha e beta devono avere stessa shape ({alpha.shape} vs {beta.shape})")
    return np.concatenate([alpha, beta, np.array([gamma, rho], dtype=np.float64)])


def decode_params(params: np.ndarray, n_teams: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Splitta `params` in `(α, β, γ, ρ)`."""
    expected = n_params(n_teams)
    if params.shape != (expected,):
        raise ValueError(
            f"params ha shape {params.shape}, atteso ({expected},) per n_teams={n_teams}"
        )
    alpha = params[:n_teams]
    beta = params[n_teams : 2 * n_teams]
    gamma = float(params[2 * n_teams])
    rho = float(params[2 * n_teams + 1])
    return alpha, beta, gamma, rho


def initial_params(n_teams: int, gamma_init: float = 0.3, rho_init: float = -0.05) -> np.ndarray:
    """Inizializzazione sensata: α = β = 0, γ piccolo positivo, ρ piccolo negativo."""
    alpha = np.zeros(n_teams, dtype=np.float64)
    beta = np.zeros(n_teams, dtype=np.float64)
    return encode_params(alpha, beta, gamma_init, rho_init)


# ---------------------------------------------------------------------------
# Match log-likelihood
# ---------------------------------------------------------------------------

def _log_poisson_pmf(k: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """log Poisson PMF: k * log(mean) − mean − log(k!) usando gammaln (stabile)."""
    return k * np.log(mean) - mean - gammaln(k + 1.0)


def match_log_likelihood(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: float,
    rho: float,
    data: MatchData,
    tau_floor: float = 1e-10,
) -> np.ndarray:
    """`log P(x_i, y_i; θ)` per ogni partita (array di shape `(n_matches,)`)."""
    log_lam = alpha[data.home_idx] + beta[data.away_idx] + gamma * data.home_advantage
    log_mu = alpha[data.away_idx] + beta[data.home_idx]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    log_p_home = data.home_goals * log_lam - lam - gammaln(data.home_goals + 1.0)
    log_p_away = data.away_goals * log_mu - mu - gammaln(data.away_goals + 1.0)

    tau_vals = tau(data.home_goals, data.away_goals, lam, mu, rho)
    log_tau = np.log(np.maximum(tau_vals, tau_floor))

    return log_p_home + log_p_away + log_tau


# ---------------------------------------------------------------------------
# NLL pesata + penalty di identificabilità
# ---------------------------------------------------------------------------

def dixon_coles_nll(
    params: np.ndarray,
    data: MatchData,
    weights: np.ndarray,
    *,
    identifiability_penalty_strength: float = 1e4,
    tau_floor: float = 1e-10,
) -> float:
    """Negative log-likelihood pesata del modello Dixon-Coles.

    `params` ha layout `[α_0..α_{n-1}, β_0..β_{n-1}, γ, ρ]` (lunghezza `2n+2`).
    `weights` è l'array dei pesi (uno per partita).

    Aggiunge una penalty quadratica `λ_id · (mean(α)^2 + mean(β)^2)` per rimuovere
    l'invarianza per shift simultaneo (identificabilità). `tau_floor` evita `log(≤0)`.
    """
    if weights.shape != (data.n_matches,):
        raise ValueError(
            f"weights ha shape {weights.shape}, atteso ({data.n_matches},)"
        )
    alpha, beta, gamma, rho = decode_params(params, data.n_teams)

    log_p = match_log_likelihood(alpha, beta, gamma, rho, data, tau_floor=tau_floor)
    nll = -float(np.sum(weights * log_p))

    penalty = identifiability_penalty_strength * (
        float(alpha.mean()) ** 2 + float(beta.mean()) ** 2
    )
    return nll + penalty


def dixon_coles_nll_from_config(
    params: np.ndarray,
    data: MatchData,
    weights: np.ndarray,
    config,
) -> float:
    """Wrapper che usa gli iperparametri da `AppConfig.dixon_coles`."""
    return dixon_coles_nll(
        params,
        data,
        weights,
        identifiability_penalty_strength=config.dixon_coles.identifiability_penalty_strength,
        tau_floor=config.dixon_coles.tau_floor,
    )


__all__ = [
    "tau",
    "encode_params",
    "decode_params",
    "initial_params",
    "n_params",
    "match_log_likelihood",
    "dixon_coles_nll",
    "dixon_coles_nll_from_config",
]
