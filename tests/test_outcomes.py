"""Test degli esiti dei 90' (Issue #6).

Copre le DoD:
- DoD #1: P(home) + P(draw) + P(away) = 1 dopo rinormalizzazione.
- DoD #2: due squadre identiche → P(home) ≈ P(away); P(draw) ragionevole (~0.25–0.30).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from src.model.fit import DixonColesModel
from src.model.outcomes import (
    DEFAULT_MAX_GOALS,
    Outcome90,
    build_score_matrix,
    compute_score_matrix,
    expected_goals,
    match_outcome_90,
)


# ---------------------------------------------------------------------------
# Fixture: modello fittizio "simmetrico" con 2 squadre
# ---------------------------------------------------------------------------

def _model_two_teams(
    teams: tuple[str, ...] = ("Equal_A", "Equal_B"),
    alpha: tuple[float, ...] = (0.0, 0.0),
    beta: tuple[float, ...] = (0.0, 0.0),
    gamma: float = 0.3,
    rho: float = -0.08,
) -> DixonColesModel:
    return DixonColesModel(
        teams=teams,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        rho=rho,
        n_matches_train=0,
        reference_date=None,
        fitted_at="2024-01-01T00:00:00+00:00",
        final_nll=0.0,
        n_iterations=0,
        converged=True,
        optimization_message="dummy",
        half_life_years=2.0,
        identifiability_penalty_strength=1e4,
        tau_floor=1e-10,
    )


# ---------------------------------------------------------------------------
# build_score_matrix (low-level)
# ---------------------------------------------------------------------------

def test_score_matrix_shape_default():
    m = build_score_matrix(1.5, 1.2, -0.05)
    assert m.shape == (DEFAULT_MAX_GOALS + 1, DEFAULT_MAX_GOALS + 1)


def test_score_matrix_sums_to_one_after_renormalization():
    """DoD #1: la massa totale dopo rinormalizzazione è 1."""
    m = build_score_matrix(1.5, 1.2, -0.07)
    assert m.sum() == pytest.approx(1.0, abs=1e-12)


def test_score_matrix_is_nonnegative():
    m = build_score_matrix(1.5, 1.2, -0.1)
    assert (m >= 0).all()


def test_score_matrix_equals_poisson_outer_when_rho_zero():
    """Con ρ=0, P(x, y) = Poisson(x; λ) · Poisson(y; μ) (Poisson indipendenti puri)."""
    lam, mu = 1.7, 1.1
    m = build_score_matrix(lam, mu, rho=0.0, max_goals=DEFAULT_MAX_GOALS)
    expected = np.outer(
        poisson.pmf(np.arange(DEFAULT_MAX_GOALS + 1), lam),
        poisson.pmf(np.arange(DEFAULT_MAX_GOALS + 1), mu),
    )
    # Rinormalizzato vs no
    expected = expected / expected.sum()
    np.testing.assert_allclose(m, expected, atol=1e-12)


def test_score_matrix_dixon_coles_correction_in_low_scores():
    """La correzione DC modifica solo i 4 punteggi bassi rispetto al Poisson puro."""
    lam, mu, rho = 1.4, 1.0, -0.10
    m_dc = build_score_matrix(lam, mu, rho)
    m_pois = build_score_matrix(lam, mu, 0.0)
    # Fuori dai 4 casi speciali, post-rinormalizzazione le due matrici differiscono
    # solo per il fattore di normalizzazione (che è ~1 perché DC redistribuisce massa).
    # Controllo strutturale: il rapporto m_dc/m_pois è costante fuori dai 4 casi.
    ratios = []
    for x in range(2, 5):
        for y in range(2, 5):
            ratios.append(m_dc[x, y] / m_pois[x, y])
    ratios = np.array(ratios)
    assert np.allclose(ratios, ratios[0], rtol=1e-9), (
        "Fuori dai 4 casi speciali la correzione DC deve essere proporzionale ovunque"
    )


def test_score_matrix_rejects_invalid_max_goals():
    with pytest.raises(ValueError):
        build_score_matrix(1.0, 1.0, 0.0, max_goals=0)


# ---------------------------------------------------------------------------
# expected_goals
# ---------------------------------------------------------------------------

def test_expected_goals_uses_gamma_at_home():
    model = _model_two_teams(gamma=0.3)
    lam_home, mu_home = expected_goals(model, "Equal_A", "Equal_B", is_neutral=False)
    lam_neut, mu_neut = expected_goals(model, "Equal_A", "Equal_B", is_neutral=True)
    # In casa: λ_home = exp(0 + 0 + 0.3) = exp(0.3); neutro: exp(0) = 1
    assert lam_home == pytest.approx(np.exp(0.3))
    assert lam_neut == pytest.approx(1.0)
    # μ non dipende da γ (l'ospite non riceve il vantaggio)
    assert mu_home == pytest.approx(1.0)
    assert mu_neut == pytest.approx(1.0)


def test_expected_goals_strength_differences():
    model = _model_two_teams(alpha=(0.5, -0.5), beta=(-0.3, 0.3))
    # Strong vs Weak in casa
    lam, mu = expected_goals(model, "Equal_A", "Equal_B", is_neutral=False)
    # λ = exp(α_A + β_B + γ) = exp(0.5 + 0.3 + 0.3) = exp(1.1)
    # μ = exp(α_B + β_A) = exp(-0.5 + (-0.3)) = exp(-0.8)
    assert lam == pytest.approx(np.exp(1.1))
    assert mu == pytest.approx(np.exp(-0.8))
    # Strong batte Weak in attacco e difesa → λ >> μ
    assert lam > mu


# ---------------------------------------------------------------------------
# match_outcome_90: DoD #1 e DoD #2
# ---------------------------------------------------------------------------

def test_outcomes_sum_to_one():
    """DoD #1: P(home) + P(draw) + P(away) = 1."""
    model = _model_two_teams()
    outcome = match_outcome_90(model, "Equal_A", "Equal_B")
    total = outcome.p_home_win + outcome.p_draw + outcome.p_away_win
    assert total == pytest.approx(1.0, abs=1e-12)


def test_identical_teams_on_neutral_field_have_symmetric_outcomes():
    """DoD #2 (forma stretta): squadre identiche in campo neutro → P(home) = P(away) esatte."""
    model = _model_two_teams(alpha=(0.0, 0.0), beta=(0.0, 0.0))
    out = match_outcome_90(model, "Equal_A", "Equal_B", is_neutral=True)
    assert out.p_home_win == pytest.approx(out.p_away_win, abs=1e-12)


def test_identical_teams_at_home_favour_home():
    """Squadre identiche, ma con γ applicato → home vantaggio."""
    model = _model_two_teams(alpha=(0.0, 0.0), beta=(0.0, 0.0), gamma=0.3)
    out = match_outcome_90(model, "Equal_A", "Equal_B", is_neutral=False)
    assert out.p_home_win > out.p_away_win


def test_identical_teams_draw_in_reasonable_range():
    """DoD #2: P(draw) tra 0.22 e 0.32 per squadre identiche."""
    model = _model_two_teams(alpha=(0.0, 0.0), beta=(0.0, 0.0), gamma=0.3, rho=-0.08)
    out = match_outcome_90(model, "Equal_A", "Equal_B", is_neutral=False)
    assert 0.22 < out.p_draw < 0.32, f"P(draw)={out.p_draw} fuori range plausibile"


def test_strong_home_team_dominates_weak_away():
    """Sanity: squadra forte in casa vs debole in trasferta → P(home) >> P(away)."""
    model = _model_two_teams(alpha=(0.8, -0.8), beta=(-0.6, 0.6), gamma=0.3)
    out = match_outcome_90(model, "Equal_A", "Equal_B")
    assert out.p_home_win > 0.8
    assert out.p_away_win < 0.05


def test_match_outcome_90_returns_lambda_and_mu():
    """`Outcome90` espone λ, μ per la cascata 90'/ET/rigori (#7)."""
    model = _model_two_teams(alpha=(0.5, -0.2), beta=(0.1, -0.3))
    out = match_outcome_90(model, "Equal_A", "Equal_B", is_neutral=False)
    lam_check, mu_check = expected_goals(model, "Equal_A", "Equal_B", is_neutral=False)
    assert out.expected_home_goals == pytest.approx(lam_check)
    assert out.expected_away_goals == pytest.approx(mu_check)


def test_outcome90_is_frozen():
    """L'oggetto è immutable (frozen dataclass)."""
    out = match_outcome_90(_model_two_teams(), "Equal_A", "Equal_B")
    with pytest.raises((AttributeError, Exception)):
        out.p_home_win = 0.5   # type: ignore[misc]


def test_outcome90_as_tuple():
    out = match_outcome_90(_model_two_teams(), "Equal_A", "Equal_B")
    assert out.as_tuple() == (out.p_home_win, out.p_draw, out.p_away_win)


# ---------------------------------------------------------------------------
# Coda fuori 11x11 trascurabile
# ---------------------------------------------------------------------------

def test_tail_outside_matrix_is_negligible_for_typical_rates():
    """Coda fuori 11×11 per (λ, μ) di partite "tipiche" del Mondiale (entrambi ~1-2 gol attesi).

    Per (λ=2.0, μ=1.5) la coda è ≈ 9·10⁻⁶ (NON 10⁻⁶ come si potrebbe stimare a mente: la
    massa fuori include anche le celle home≤10 ∧ away>10 e simmetriche). Per fini pratici
    della cascata 90'/ET/rigori la rinormalizzazione assorbe la coda senza alterare P(home),
    P(draw), P(away) in modo percettibile.

    Per λ alti (es. top vs weak con λ ≈ 5–10), la coda cresce a 1% o più; in quei casi
    la rinormalizzazione conserva i 3 esiti aggregati ma altera P(score=x) per x grandi
    (non usato nella cascata, vedi #7).
    """
    lam, mu = 2.0, 1.5
    k = np.arange(DEFAULT_MAX_GOALS + 1)
    pre_norm = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu)).sum()
    tail = 1.0 - pre_norm
    # Bound naturale per λ, μ ≤ 2: coda < 10⁻⁴ con margine di sicurezza
    assert tail < 1e-4, f"tail={tail:.2e} per λ={lam}, μ={mu}"
    # Per partite "molto sbilanciate" la coda cresce — sanity contro-fattuale:
    pre_norm_extreme = np.outer(poisson.pmf(k, 5.0), poisson.pmf(k, 1.0)).sum()
    tail_extreme = 1.0 - pre_norm_extreme
    assert tail_extreme > 1e-4, "Per λ=5 ci aspettiamo coda > 1e-4 (sanity)"


# ---------------------------------------------------------------------------
# Errori
# ---------------------------------------------------------------------------

def test_unknown_team_raises():
    model = _model_two_teams()
    with pytest.raises(KeyError):
        match_outcome_90(model, "Equal_A", "Zelandia")


# ---------------------------------------------------------------------------
# compute_score_matrix wrapper
# ---------------------------------------------------------------------------

def test_compute_score_matrix_matches_low_level():
    """`compute_score_matrix` deve produrre la stessa matrice di `build_score_matrix`."""
    model = _model_two_teams(alpha=(0.2, -0.2), beta=(0.1, -0.1))
    m_top = compute_score_matrix(model, "Equal_A", "Equal_B", is_neutral=False)
    lam, mu = expected_goals(model, "Equal_A", "Equal_B", is_neutral=False)
    m_low = build_score_matrix(lam, mu, model.rho)
    np.testing.assert_array_equal(m_top, m_low)
