"""Test della cascata 90'/ET/rigori (Issue #7).

Copre le DoD:
- DoD #1: `P(home passa) + P(away passa) = 1`.
- DoD #2: la squadra più forte ha P(passa) > 0.5 e resta più forte anche nel ramo ET
          (eredita i λ scalati).
- DoD #3: forze uguali → P(passa) = 0.5 esatto.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.model.cascade import (
    AdvanceOutcome,
    advance_probability,
    advance_probability_from_config,
    penalty_shootout_winner_probability,
)
from src.model.fit import DixonColesModel


# ---------------------------------------------------------------------------
# Fixture: modelli minimi
# ---------------------------------------------------------------------------

def _model(alpha=(0.0, 0.0), beta=(0.0, 0.0), gamma=0.3, rho=-0.08,
           teams=("Equal_A", "Equal_B")) -> DixonColesModel:
    return DixonColesModel(
        teams=teams, alpha=alpha, beta=beta, gamma=gamma, rho=rho,
        n_matches_train=0, reference_date=None,
        fitted_at="2024-01-01T00:00:00+00:00",
        final_nll=0.0, n_iterations=0, converged=True, optimization_message="",
        half_life_years=2.0, identifiability_penalty_strength=1e4, tau_floor=1e-10,
    )


# ---------------------------------------------------------------------------
# penalty_shootout_winner_probability
# ---------------------------------------------------------------------------

def test_penalty_coin_flip_when_equal_p90():
    p, favorite = penalty_shootout_winner_probability(
        0.4, 0.4, base_prob=0.5, edge_to_favorite=0.02
    )
    assert p == 0.5
    assert favorite == "tie"


def test_penalty_edge_goes_to_home_when_favored():
    p, favorite = penalty_shootout_winner_probability(
        0.55, 0.30, base_prob=0.5, edge_to_favorite=0.02
    )
    assert p == pytest.approx(0.52)
    assert favorite == "home"


def test_penalty_edge_goes_to_away_when_favored():
    p, favorite = penalty_shootout_winner_probability(
        0.30, 0.55, base_prob=0.5, edge_to_favorite=0.02
    )
    assert p == pytest.approx(0.48)
    assert favorite == "away"


def test_penalty_edge_zero_is_always_coin_flip():
    """Con edge=0, P(home vince rigori) = 0.5 indipendentemente dal favorito."""
    for ph, pa in [(0.6, 0.2), (0.2, 0.6), (0.4, 0.4)]:
        p, _ = penalty_shootout_winner_probability(
            ph, pa, base_prob=0.5, edge_to_favorite=0.0
        )
        assert p == 0.5


# ---------------------------------------------------------------------------
# DoD #1: P(home) + P(away) = 1 (advance)
# ---------------------------------------------------------------------------

def test_advance_probabilities_sum_to_one():
    """DoD #1."""
    model = _model(alpha=(0.4, -0.2), beta=(-0.1, 0.2))
    out = advance_probability(model, "Equal_A", "Equal_B")
    total = out.p_home_advance + out.p_away_advance
    assert total == pytest.approx(1.0, abs=1e-12)


def test_advance_probabilities_sum_to_one_in_many_configs():
    """DoD #1 stress-test: varie combinazioni di forze e is_neutral."""
    for alpha_a, beta_a, alpha_b, beta_b in [
        (0.0, 0.0, 0.0, 0.0),
        (0.5, -0.4, -0.3, 0.2),
        (-0.5, 0.3, 0.4, -0.2),
        (1.0, -1.0, -1.0, 1.0),
    ]:
        model = _model(alpha=(alpha_a, alpha_b), beta=(beta_a, beta_b))
        for neutral in (True, False):
            out = advance_probability(model, "Equal_A", "Equal_B", is_neutral=neutral)
            total = out.p_home_advance + out.p_away_advance
            assert total == pytest.approx(1.0, abs=1e-12), (
                f"forze={alpha_a, beta_a, alpha_b, beta_b}, neutral={neutral}"
            )


# ---------------------------------------------------------------------------
# DoD #3: forze uguali → P(passa) = 0.5 esatto (su campo neutro)
# ---------------------------------------------------------------------------

def test_equal_strengths_neutral_field_gives_50_50_advance():
    """DoD #3: due squadre identiche in campo neutro → P(A passa) = 0.5 esatte."""
    model = _model(alpha=(0.0, 0.0), beta=(0.0, 0.0))
    out = advance_probability(model, "Equal_A", "Equal_B", is_neutral=True)
    assert out.p_home_advance == pytest.approx(0.5, abs=1e-12)
    assert out.p_away_advance == pytest.approx(0.5, abs=1e-12)
    # Simmetria a tutti i livelli:
    assert out.p_home_win_90 == pytest.approx(out.p_away_win_90)
    assert out.p_home_win_et == pytest.approx(out.p_away_win_et)
    assert out.p_home_win_penalty == 0.5
    assert out.penalty_favorite == "tie"


# ---------------------------------------------------------------------------
# DoD #2: squadra più forte ha P(passa) > 0.5 e resta più forte anche in ET
# ---------------------------------------------------------------------------

def test_stronger_team_passes_more_than_half():
    """DoD #2 (forte vs debole)."""
    model = _model(alpha=(0.5, -0.5), beta=(-0.3, 0.3))   # Equal_A è "forte"
    out = advance_probability(model, "Equal_A", "Equal_B", is_neutral=True)
    assert out.p_home_advance > 0.5
    assert out.p_home_advance > out.p_away_advance


def test_stronger_team_remains_favorite_in_et():
    """DoD #2 (eredita λ scalati nel ramo ET)."""
    model = _model(alpha=(0.5, -0.5), beta=(-0.3, 0.3))
    out = advance_probability(model, "Equal_A", "Equal_B", is_neutral=True)
    # Anche con λ/3, μ/3, il forte resta favorito
    assert out.p_home_win_et > out.p_away_win_et
    # E i tassi ET sono effettivamente scalati 1/3
    assert out.expected_home_goals_et == pytest.approx(out.expected_home_goals_90 / 3, rel=1e-9)
    assert out.expected_away_goals_et == pytest.approx(out.expected_away_goals_90 / 3, rel=1e-9)


def test_stronger_team_remains_favorite_at_home():
    """Con vantaggio casa, la forte è ancora più favorita."""
    model = _model(alpha=(0.5, -0.5), beta=(-0.3, 0.3), gamma=0.3)
    out_neutral = advance_probability(model, "Equal_A", "Equal_B", is_neutral=True)
    out_home = advance_probability(model, "Equal_A", "Equal_B", is_neutral=False)
    assert out_home.p_home_advance > out_neutral.p_home_advance


# ---------------------------------------------------------------------------
# Composizione e self-consistency
# ---------------------------------------------------------------------------

def test_cascade_formula_matches_components():
    """Verifica esplicitamente la formula:
    P(home passa) = P(home 90') + P(draw 90') · [P(home ET) + P(draw ET) · P(home rigori)]"""
    model = _model(alpha=(0.3, -0.3), beta=(0.1, -0.1))
    out = advance_probability(model, "Equal_A", "Equal_B")
    expected = (
        out.p_home_win_90
        + out.p_draw_90 * (out.p_home_win_et + out.p_draw_et * out.p_home_win_penalty)
    )
    assert out.p_home_advance == pytest.approx(expected, abs=1e-12)


def test_outcome_structure_is_frozen():
    out = advance_probability(_model(), "Equal_A", "Equal_B")
    with pytest.raises((AttributeError, Exception)):
        out.p_home_advance = 0.5    # type: ignore[misc]


def test_penalty_edge_to_favorite_amplifies_advantage():
    """Con edge_to_favorite > 0, il forte ottiene un boost addizionale."""
    model = _model(alpha=(0.3, -0.3), beta=(-0.1, 0.1))
    out_zero = advance_probability(model, "Equal_A", "Equal_B",
                                    penalty_edge_to_favorite=0.0)
    out_two = advance_probability(model, "Equal_A", "Equal_B",
                                   penalty_edge_to_favorite=0.02)
    assert out_two.p_home_advance > out_zero.p_home_advance
    assert out_two.p_home_win_penalty == pytest.approx(0.52)
    assert out_zero.p_home_win_penalty == 0.5


def test_invalid_outcome_construction_raises():
    """`__post_init__` rifiuta esiti che non sommano a 1."""
    with pytest.raises(ValueError, match="non sommano a 1"):
        AdvanceOutcome(
            home_team="A", away_team="B", is_neutral=False,
            p_home_win_90=0.5, p_draw_90=0.3, p_away_win_90=0.5,  # somma 1.3 (rotta)
            expected_home_goals_90=1.0, expected_away_goals_90=1.0,
            p_home_win_et=0.4, p_draw_et=0.2, p_away_win_et=0.4,
            expected_home_goals_et=0.3, expected_away_goals_et=0.3,
            p_home_win_penalty=0.5, penalty_favorite="tie",
            p_home_advance=0.5, p_away_advance=0.5,
        )


# ---------------------------------------------------------------------------
# Wrapper from_config
# ---------------------------------------------------------------------------

def test_advance_probability_from_config_matches_explicit():
    """Il wrapper from-config deve produrre lo stesso risultato della chiamata esplicita."""
    cfg = load_config()
    model = _model(alpha=(0.3, -0.3), beta=(-0.1, 0.1))
    out_cfg = advance_probability_from_config(model, "Equal_A", "Equal_B", cfg)
    out_explicit = advance_probability(
        model, "Equal_A", "Equal_B",
        extra_time_lambda_factor=cfg.extra_time.lambda_factor,
        extra_time_mu_factor=cfg.extra_time.mu_factor,
        penalty_base_prob=cfg.penalty_shootout.base_prob_winner,
        penalty_edge_to_favorite=cfg.penalty_shootout.edge_to_favorite,
    )
    assert out_cfg.p_home_advance == pytest.approx(out_explicit.p_home_advance)
    assert out_cfg.p_home_win_et == pytest.approx(out_explicit.p_home_win_et)


# ---------------------------------------------------------------------------
# Sanity contro l'ET shrinkage
# ---------------------------------------------------------------------------

def test_extra_time_draw_probability_higher_than_regulation():
    """Con tassi λ/3, μ/3, P(pari) in ET deve essere ≥ P(pari) in 90' (meno gol → più 0-0)."""
    model = _model(alpha=(0.0, 0.0), beta=(0.0, 0.0), gamma=0.0)
    out = advance_probability(model, "Equal_A", "Equal_B", is_neutral=True)
    assert out.p_draw_et > out.p_draw_90, (
        f"P(draw ET)={out.p_draw_et}, P(draw 90')={out.p_draw_90}"
    )
