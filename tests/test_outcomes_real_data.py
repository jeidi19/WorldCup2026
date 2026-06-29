"""Test di integrazione di `match_outcome_90` sul modello reale (Issue #6).

Skippa se `data/models/dixon_coles.json` non esiste.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.model.fit import DixonColesModel
from src.model.outcomes import match_outcome_90


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_JSON = REPO_ROOT / "data" / "models" / "dixon_coles.json"

pytestmark = pytest.mark.skipif(
    not MODEL_JSON.exists(),
    reason="data/models/dixon_coles.json non disponibile",
)


@pytest.fixture(scope="module")
def model() -> DixonColesModel:
    return DixonColesModel.load(MODEL_JSON)


def test_top_team_at_home_vs_weak_team_dominates(model):
    """Argentina (#1) in casa vs San Marino → P(home) > 0.95."""
    if "Argentina" not in model.teams or "San Marino" not in model.teams:
        pytest.skip("Squadre attese non nel modello")
    out = match_outcome_90(model, "Argentina", "San Marino", is_neutral=False)
    assert out.p_home_win > 0.95
    assert out.p_away_win < 0.02
    assert (out.p_home_win + out.p_draw + out.p_away_win) == pytest.approx(1.0, abs=1e-12)


def test_home_advantage_flips_outcome_for_equal_strength(model):
    """Vantaggio casa: per due squadre con strength simile, P(home) > P(away) in casa e
    P(home) = P(away) in neutro. Test simmetrico che isola γ.

    Scegliamo dinamicamente due squadre con strength vicina (differenza < 0.05 in
    α − β): qualsiasi coppia consecutiva nel ranking soddisfa il requisito.
    """
    df = model.to_dataframe()
    # Trova prima coppia consecutiva con strength molto vicina (< 0.05)
    diffs = df["strength"].diff(-1).abs()   # |strength[i] − strength[i+1]|
    candidates = df[diffs < 0.05].head(20)
    assert not candidates.empty, "Nessuna coppia consecutiva con strength vicina"
    idx = candidates.index[0]
    a, b = df.iloc[idx]["team"], df.iloc[idx + 1]["team"]

    out_home = match_outcome_90(model, a, b, is_neutral=False)
    out_neut = match_outcome_90(model, a, b, is_neutral=True)

    # Vantaggio casa applicato a `a`: P(home) deve aumentare passando da neutro a in casa
    assert out_home.p_home_win > out_neut.p_home_win, (
        f"{a} vs {b}: P(home, neutro)={out_neut.p_home_win:.3f} -> "
        f"P(home, in casa)={out_home.p_home_win:.3f}"
    )
    # E i pareggi sono in range plausibile
    assert 0.15 < out_home.p_draw < 0.40


def test_neutral_field_symmetric_for_equal_strength(model):
    """Per due squadre vicine come strength su campo neutro, le P(home)/P(away) sono vicine."""
    # Prendo due squadre con strength simile dalla top
    df = model.to_dataframe()
    # Filtro a quelle in dataset reale
    s_top = df.head(20)
    a, b = s_top.iloc[3]["team"], s_top.iloc[4]["team"]   # quarta e quinta strength
    out = match_outcome_90(model, a, b, is_neutral=True)
    diff = abs(out.p_home_win - out.p_away_win)
    # Su campo neutro, due squadre di strength vicina hanno probabilità di vincita ≈
    assert diff < 0.10, f"P(home)={out.p_home_win}, P(away)={out.p_away_win}, diff={diff}"


def test_outcomes_always_sum_to_one_on_real_matches(model):
    """Sanity sweep: per 30 coppie random, P(home)+P(draw)+P(away) = 1."""
    import random
    random.seed(0)
    teams = list(model.teams)
    for _ in range(30):
        a, b = random.sample(teams, 2)
        out = match_outcome_90(model, a, b)
        total = out.p_home_win + out.p_draw + out.p_away_win
        assert total == pytest.approx(1.0, abs=1e-10), f"{a} vs {b}: total={total}"


def test_expected_goals_positive_and_finite(model):
    """λ, μ devono essere positivi e finiti per qualunque coppia."""
    import math
    out = match_outcome_90(model, "France", "Italy")
    assert math.isfinite(out.expected_home_goals) and out.expected_home_goals > 0
    assert math.isfinite(out.expected_away_goals) and out.expected_away_goals > 0
