"""Test di integrazione di `advance_probability` sul modello reale (Issue #7).

Skippa se `data/models/dixon_coles.json` non esiste.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.config import load_config
from src.model.cascade import advance_probability, advance_probability_from_config
from src.model.fit import DixonColesModel


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_JSON = REPO_ROOT / "data" / "models" / "dixon_coles.json"

pytestmark = pytest.mark.skipif(
    not MODEL_JSON.exists(),
    reason="data/models/dixon_coles.json non disponibile",
)


@pytest.fixture(scope="module")
def model() -> DixonColesModel:
    return DixonColesModel.load(MODEL_JSON)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_top_vs_weak_passes_near_certainly(model, config):
    """Argentina vs San Marino in neutro: P(Argentina passa) > 0.99 (Argentina è #1)."""
    if "Argentina" not in model.teams or "San Marino" not in model.teams:
        pytest.skip("Squadre attese non nel modello")
    out = advance_probability_from_config(
        model, "Argentina", "San Marino", config, is_neutral=True
    )
    assert out.p_home_advance > 0.99


def test_close_match_balanced_advance(model, config):
    """Due squadre con strength molto vicina → P(passa) entrambe vicino a 0.5."""
    df = model.to_dataframe()
    diffs = df["strength"].diff(-1).abs()
    candidates = df[diffs < 0.05]
    assert not candidates.empty
    idx = candidates.index[0]
    a, b = df.iloc[idx]["team"], df.iloc[idx + 1]["team"]
    out = advance_probability_from_config(model, a, b, config, is_neutral=True)
    diff = abs(out.p_home_advance - out.p_away_advance)
    assert diff < 0.15, (
        f"{a} vs {b}: P(home)={out.p_home_advance:.3f}, "
        f"P(away)={out.p_away_advance:.3f}, diff={diff:.3f}"
    )


def test_advance_sums_to_one_on_random_pairs(model, config):
    random.seed(7)
    teams = list(model.teams)
    for _ in range(30):
        a, b = random.sample(teams, 2)
        out = advance_probability_from_config(model, a, b, config)
        total = out.p_home_advance + out.p_away_advance
        assert total == pytest.approx(1.0, abs=1e-10), (
            f"{a} vs {b}: total={total}"
        )


def test_et_inherits_scaled_lambda(model, config):
    """λ_ET = λ/3, μ_ET = μ/3 esatti per qualunque coppia."""
    out = advance_probability_from_config(model, "France", "Germany", config)
    assert out.expected_home_goals_et == pytest.approx(out.expected_home_goals_90 / 3, rel=1e-9)
    assert out.expected_away_goals_et == pytest.approx(out.expected_away_goals_90 / 3, rel=1e-9)


def test_advance_probability_at_least_one_real_world_cup_pair(model, config):
    """Esempio reale: Brazil vs Japan (gruppo dataset Mondiale 2026 in corso)."""
    if not {"Brazil", "Japan"} <= set(model.teams):
        pytest.skip("Brazil o Japan non in model")
    out = advance_probability_from_config(model, "Brazil", "Japan", config, is_neutral=True)
    # Brazil ha strength superiore -> favorito
    assert out.p_home_advance > out.p_away_advance
    # Ma Japan ha rating decente (#13) -> non schiacciante
    assert 0.50 < out.p_home_advance < 0.85
