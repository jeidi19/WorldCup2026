"""Esiti dei 90 minuti regolari (Issue #6).

Dato un `DixonColesModel` fittato e due squadre `home` / `away`, costruiamo la matrice
`P(x, y)` per `x, y ∈ 0..max_goals` (default `max_goals=10`, quindi 11×11), applichiamo
la correzione Dixon-Coles sui 4 casi speciali, rinormalizziamo per gestire la coda fuori
dalla matrice (~10⁻¹⁰ per λ, μ tipici), e ne estraiamo i tre esiti:

- `P(home vince 90')` = somma su `x > y`
- `P(pari 90')`        = traccia (diagonale)
- `P(away vince 90')`  = somma su `x < y`

`match_outcome_90` ritorna un `Outcome90` (frozen dataclass) con tutto il necessario per
la cascata 90'/supplementari/rigori (#7): `expected_home_goals = λ` e
`expected_away_goals = μ` riusati direttamente come `λ_ET = λ/3`, `μ_ET = μ/3`.

`is_neutral`: se `True`, `γ` non viene applicato (campo neutro). Per #8 (host policy
2026) la chiamata verrà preceduta da un calcolo della scala di γ a partire dalla sede.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from src.config import HostAdvantage2026
from src.model.fit import DixonColesModel
from src.model.host_policy import host_advantage_scale

DEFAULT_MAX_GOALS = 10


@dataclass(frozen=True)
class Outcome90:
    """Esiti probabilistici dei 90 minuti regolari per una partita."""

    home_team: str
    away_team: str
    is_neutral: bool
    p_home_win: float
    p_draw: float
    p_away_win: float
    expected_home_goals: float   # λ
    expected_away_goals: float   # μ

    def as_tuple(self) -> tuple[float, float, float]:
        """Le tre probabilità nell'ordine (home, draw, away). Sommano a 1."""
        return self.p_home_win, self.p_draw, self.p_away_win


# ---------------------------------------------------------------------------
# Helper low-level (lavorano su λ, μ, ρ direttamente: utili in #7 per ET)
# ---------------------------------------------------------------------------

def _resolve_home_advantage_scale(
    home: str,
    is_neutral: bool,
    home_advantage_scale: float | None,
    venue_country: str | None,
    host_policy: HostAdvantage2026 | None,
) -> float:
    """Determina il moltiplicatore di γ in base ai parametri (priorità esplicita)."""
    if home_advantage_scale is not None:
        return float(home_advantage_scale)
    if venue_country is not None:
        if host_policy is None:
            from src.config import load_config  # lazy: evita circolari nel boot
            host_policy = load_config().host_advantage_2026
        return host_advantage_scale(home, venue_country, host_policy)
    return 0.0 if is_neutral else 1.0


def expected_goals(
    model: DixonColesModel,
    home: str,
    away: str,
    *,
    is_neutral: bool = False,
    home_advantage_scale: float | None = None,
    venue_country: str | None = None,
    host_policy: HostAdvantage2026 | None = None,
) -> tuple[float, float]:
    """Restituisce `(λ, μ)` per la partita.

    λ = exp(α[home] + β[away] + γ · scale_home), μ = exp(α[away] + β[home]).

    Lo scale_home (moltiplicatore di γ) viene risolto in questo ordine di priorità:
    1. `home_advantage_scale` esplicito (float, override diretto);
    2. `venue_country` + `host_policy`: applica la host policy 2026 (Issue #8);
    3. `is_neutral`: True → 0.0 (campo neutro), False → 1.0 (in casa).
    """
    a_h = model.alpha_of(home)
    b_h = model.beta_of(home)
    a_a = model.alpha_of(away)
    b_a = model.beta_of(away)
    scale = _resolve_home_advantage_scale(
        home, is_neutral, home_advantage_scale, venue_country, host_policy
    )
    lam = float(np.exp(a_h + b_a + model.gamma * scale))
    mu = float(np.exp(a_a + b_h))
    return lam, mu


def build_score_matrix(
    lam: float,
    mu: float,
    rho: float,
    *,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """Matrice P(x, y) con correzione DC sui 4 casi speciali, rinormalizzata.

    Esposta separatamente perché in #7 i supplementari riutilizzano questa con
    `λ_ET = λ/3`, `μ_ET = μ/3`.
    """
    if max_goals < 1:
        raise ValueError(f"max_goals deve essere >= 1, ricevuto {max_goals}")

    k = np.arange(max_goals + 1, dtype=np.float64)
    log_fact = gammaln(k + 1.0)
    # log Poisson PMF
    log_pois_home = k * np.log(lam) - lam - log_fact
    log_pois_away = k * np.log(mu) - mu - log_fact
    pois_home = np.exp(log_pois_home)
    pois_away = np.exp(log_pois_away)

    matrix = np.outer(pois_home, pois_away)

    # Correzione Dixon-Coles sui 4 punteggi bassi
    matrix[0, 0] *= 1.0 - lam * mu * rho
    matrix[0, 1] *= 1.0 + lam * rho
    matrix[1, 0] *= 1.0 + mu * rho
    matrix[1, 1] *= 1.0 - rho

    # Clip floor: la τ può dare negativi in scenari patologici (|ρ| grande + λμ alti)
    matrix = np.clip(matrix, 0.0, None)

    # Rinormalizzazione (gestisce la coda fuori dalla matrice, trascurabile per λ, μ usuali)
    total = matrix.sum()
    if total <= 0:
        raise ValueError(
            f"Score matrix degenere (massa totale {total}). "
            f"Controlla λ={lam}, μ={mu}, ρ={rho}."
        )
    matrix /= total
    return matrix


# ---------------------------------------------------------------------------
# API user-facing
# ---------------------------------------------------------------------------

def compute_score_matrix(
    model: DixonColesModel,
    home: str,
    away: str,
    *,
    is_neutral: bool = False,
    home_advantage_scale: float | None = None,
    venue_country: str | None = None,
    host_policy: HostAdvantage2026 | None = None,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """Matrice `P(x, y)` per la partita, rinormalizzata e con correzione DC."""
    lam, mu = expected_goals(
        model, home, away,
        is_neutral=is_neutral,
        home_advantage_scale=home_advantage_scale,
        venue_country=venue_country,
        host_policy=host_policy,
    )
    return build_score_matrix(lam, mu, model.rho, max_goals=max_goals)


def match_outcome_90(
    model: DixonColesModel,
    home: str,
    away: str,
    *,
    is_neutral: bool = False,
    home_advantage_scale: float | None = None,
    venue_country: str | None = None,
    host_policy: HostAdvantage2026 | None = None,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> Outcome90:
    """Esiti probabilistici dei 90 minuti regolari.

    Per il vantaggio casa si veda `expected_goals` (priorità: `home_advantage_scale` >
    `venue_country` + `host_policy` > `is_neutral`). `is_neutral` riportato in
    `Outcome90` riflette lo scale effettivo (0 ↔ neutro, > 0 ↔ vantaggio attivo).
    """
    lam, mu = expected_goals(
        model, home, away,
        is_neutral=is_neutral,
        home_advantage_scale=home_advantage_scale,
        venue_country=venue_country,
        host_policy=host_policy,
    )
    matrix = build_score_matrix(lam, mu, model.rho, max_goals=max_goals)

    i, j = np.indices(matrix.shape)
    p_home_win = float(matrix[i > j].sum())
    p_draw = float(np.diag(matrix).sum())
    p_away_win = float(matrix[i < j].sum())

    # is_neutral effettivo: vero se lo scale risultante è 0 (γ non applicato)
    effective_scale = _resolve_home_advantage_scale(
        home, is_neutral, home_advantage_scale, venue_country, host_policy
    )
    effective_neutral = effective_scale == 0.0

    return Outcome90(
        home_team=home,
        away_team=away,
        is_neutral=effective_neutral,
        p_home_win=p_home_win,
        p_draw=p_draw,
        p_away_win=p_away_win,
        expected_home_goals=lam,
        expected_away_goals=mu,
    )


__all__ = [
    "Outcome90",
    "DEFAULT_MAX_GOALS",
    "expected_goals",
    "build_score_matrix",
    "compute_score_matrix",
    "match_outcome_90",
]
