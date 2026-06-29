"""Cascata 90' / supplementari / rigori → P(passa) (Issue #7).

Per ogni partita a eliminazione diretta del Mondiale 2026, calcoliamo la probabilità
che la squadra di casa **superi il turno** combinando i tre rami:

    P(home passa) = P(home vince 90')
                  + P(pari 90') · [ P(home vince ET) + P(pari ET) · P(home vince rigori) ]

dove:
- gli esiti dei 90' usano la matrice 11×11 con (λ, μ, ρ) del modello;
- gli esiti dei supplementari usano la stessa procedura con `λ_ET = λ · k_λ` e
  `μ_ET = μ · k_μ` (default `k_λ = k_μ = 1/3`, ~30 min vs 90 min);
- i rigori sono una coin flip pesata: P(favorito sui 90' vince) =
  `base_prob + edge_to_favorite` (default 0.50, edge 0).

`AdvanceOutcome` espone tutti i passaggi della cascata per debug/analisi e per il MC
del bracket (#16).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.config import AppConfig, HostAdvantage2026
from src.model.fit import DixonColesModel
from src.model.outcomes import (
    DEFAULT_MAX_GOALS,
    Outcome90,
    build_score_matrix,
    expected_goals,
    match_outcome_90,
)


@dataclass(frozen=True)
class AdvanceOutcome:
    """Cascata completa: 90' → ET → rigori → P(passa) per la squadra di casa."""

    home_team: str
    away_team: str
    is_neutral: bool

    # 90 minuti regolari
    p_home_win_90: float
    p_draw_90: float
    p_away_win_90: float
    expected_home_goals_90: float
    expected_away_goals_90: float

    # Supplementari (mini-matrice con tassi scalati)
    p_home_win_et: float
    p_draw_et: float
    p_away_win_et: float
    expected_home_goals_et: float
    expected_away_goals_et: float

    # Rigori
    p_home_win_penalty: float
    penalty_favorite: Literal["home", "away", "tie"]

    # Cascata finale
    p_home_advance: float
    p_away_advance: float

    def __post_init__(self) -> None:
        # Sanity guard: i tre esiti dei 90' e degli ET devono sommare a 1
        # (gli errori di rinormalizzazione potrebbero introdurre piccoli drift).
        for label, vals in (
            ("90'", (self.p_home_win_90, self.p_draw_90, self.p_away_win_90)),
            ("ET",  (self.p_home_win_et, self.p_draw_et, self.p_away_win_et)),
        ):
            total = sum(vals)
            if not (abs(total - 1.0) < 1e-9):
                raise ValueError(
                    f"AdvanceOutcome: esiti {label} non sommano a 1 (total={total})"
                )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _split_score_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Da matrice P(x, y) ai 3 esiti (home_win, draw, away_win) — sommano a 1."""
    i, j = np.indices(matrix.shape)
    p_home_win = float(matrix[i > j].sum())
    p_draw = float(np.diag(matrix).sum())
    p_away_win = float(matrix[i < j].sum())
    return p_home_win, p_draw, p_away_win


def penalty_shootout_winner_probability(
    p_home_win_90: float,
    p_away_win_90: float,
    *,
    base_prob: float,
    edge_to_favorite: float,
) -> tuple[float, Literal["home", "away", "tie"]]:
    """Restituisce `(P(home vince rigori), chi è il favorito)`.

    Il "favorito" è la squadra con la maggior P(vincere sui 90'); riceve `+edge_to_favorite`
    rispetto al base. In caso di pareggio esatto delle due probabilità sui 90', i rigori
    sono coin flip (`base_prob`).
    """
    if p_home_win_90 > p_away_win_90:
        return base_prob + edge_to_favorite, "home"
    if p_home_win_90 < p_away_win_90:
        return base_prob - edge_to_favorite, "away"
    return base_prob, "tie"


# ---------------------------------------------------------------------------
# Core: advance_probability
# ---------------------------------------------------------------------------

def advance_probability(
    model: DixonColesModel,
    home: str,
    away: str,
    *,
    is_neutral: bool = False,
    home_advantage_scale: float | None = None,
    venue_country: str | None = None,
    host_policy: HostAdvantage2026 | None = None,
    extra_time_lambda_factor: float = 1.0 / 3.0,
    extra_time_mu_factor: float = 1.0 / 3.0,
    penalty_base_prob: float = 0.5,
    penalty_edge_to_favorite: float = 0.0,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> AdvanceOutcome:
    """Probabilità che `home` superi il turno contro `away` (eliminazione diretta).

    Cascata: 90' → supplementari (mini-matrice con tassi scalati) → rigori.
    Per il vantaggio casa, vedi `expected_goals` (priorità: `home_advantage_scale` >
    `venue_country` + `host_policy` > `is_neutral`).
    """
    # 90'
    lam, mu = expected_goals(
        model, home, away,
        is_neutral=is_neutral,
        home_advantage_scale=home_advantage_scale,
        venue_country=venue_country,
        host_policy=host_policy,
    )
    matrix_90 = build_score_matrix(lam, mu, model.rho, max_goals=max_goals)
    p_h_90, p_d_90, p_a_90 = _split_score_matrix(matrix_90)

    # Supplementari (~30 min ≈ 1/3 del tempo, gol attesi scalati di conseguenza)
    lam_et = lam * extra_time_lambda_factor
    mu_et = mu * extra_time_mu_factor
    matrix_et = build_score_matrix(lam_et, mu_et, model.rho, max_goals=max_goals)
    p_h_et, p_d_et, p_a_et = _split_score_matrix(matrix_et)

    # Rigori
    p_h_penalty, favorite = penalty_shootout_winner_probability(
        p_h_90, p_a_90,
        base_prob=penalty_base_prob,
        edge_to_favorite=penalty_edge_to_favorite,
    )

    # Cascata
    p_h_advance = p_h_90 + p_d_90 * (p_h_et + p_d_et * p_h_penalty)
    p_a_advance = 1.0 - p_h_advance

    # is_neutral effettivo: vero se lo scale risultante è 0 (γ non applicato)
    from src.model.outcomes import _resolve_home_advantage_scale  # late import per evitare cicli
    effective_scale = _resolve_home_advantage_scale(
        home, is_neutral, home_advantage_scale, venue_country, host_policy
    )
    effective_neutral = effective_scale == 0.0

    return AdvanceOutcome(
        home_team=home,
        away_team=away,
        is_neutral=effective_neutral,
        p_home_win_90=p_h_90,
        p_draw_90=p_d_90,
        p_away_win_90=p_a_90,
        expected_home_goals_90=lam,
        expected_away_goals_90=mu,
        p_home_win_et=p_h_et,
        p_draw_et=p_d_et,
        p_away_win_et=p_a_et,
        expected_home_goals_et=lam_et,
        expected_away_goals_et=mu_et,
        p_home_win_penalty=p_h_penalty,
        penalty_favorite=favorite,
        p_home_advance=p_h_advance,
        p_away_advance=p_a_advance,
    )


def advance_probability_from_config(
    model: DixonColesModel,
    home: str,
    away: str,
    config: AppConfig,
    *,
    is_neutral: bool = False,
    venue_country: str | None = None,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> AdvanceOutcome:
    """Wrapper che legge ET, rigori e (se `venue_country` è fornito) host policy da `AppConfig`."""
    return advance_probability(
        model, home, away,
        is_neutral=is_neutral,
        venue_country=venue_country,
        host_policy=config.host_advantage_2026 if venue_country is not None else None,
        extra_time_lambda_factor=config.extra_time.lambda_factor,
        extra_time_mu_factor=config.extra_time.mu_factor,
        penalty_base_prob=config.penalty_shootout.base_prob_winner,
        penalty_edge_to_favorite=config.penalty_shootout.edge_to_favorite,
        max_goals=max_goals,
    )


__all__ = [
    "AdvanceOutcome",
    "penalty_shootout_winner_probability",
    "advance_probability",
    "advance_probability_from_config",
]
