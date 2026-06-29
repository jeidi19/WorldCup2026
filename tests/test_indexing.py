"""Test del TeamIndexer e di prepare_match_data (Issue #4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model.indexing import MatchData, TeamIndexer, prepare_match_data


# ---------------------------------------------------------------------------
# TeamIndexer
# ---------------------------------------------------------------------------

def test_indexer_is_alphabetic_and_deterministic():
    a = TeamIndexer(["Italy", "Argentina", "Brazil", "Italy"])
    b = TeamIndexer(["Brazil", "Italy", "Argentina"])
    # Stesso input set → stesso ordine (alfabetico) → stessi indici
    assert a.teams == b.teams == ("Argentina", "Brazil", "Italy")
    assert a.to_idx("Argentina") == 0
    assert a.to_idx("Italy") == 2


def test_indexer_len_and_contains():
    idx = TeamIndexer(["A", "B"])
    assert len(idx) == idx.n_teams == 2
    assert "A" in idx
    assert "Z" not in idx


def test_indexer_to_idx_array_preserves_order():
    idx = TeamIndexer(["A", "B", "C"])
    result = idx.to_idx_array(pd.Series(["B", "C", "A", "A"]))
    np.testing.assert_array_equal(result, np.array([1, 2, 0, 0], dtype=np.int64))


def test_indexer_unknown_team_raises():
    idx = TeamIndexer(["A", "B"])
    with pytest.raises(KeyError):
        idx.to_idx("Z")


def test_indexer_empty_input_raises():
    with pytest.raises(ValueError):
        TeamIndexer([])


def test_indexer_from_dataframe():
    df = pd.DataFrame({"home_team": ["Italy", "Spain"], "away_team": ["Brazil", "Italy"]})
    idx = TeamIndexer.from_match_dataframe(df)
    assert idx.teams == ("Brazil", "Italy", "Spain")


# ---------------------------------------------------------------------------
# MatchData / prepare_match_data
# ---------------------------------------------------------------------------

def test_prepare_match_data_basic():
    df = pd.DataFrame(
        {
            "home_team": ["Italy", "Spain", "Brazil"],
            "away_team": ["Spain", "Brazil", "Italy"],
            "home_score": [2, 1, 0],
            "away_score": [1, 1, 0],
            "neutral": [False, True, False],
        }
    )
    indexer = TeamIndexer.from_match_dataframe(df)
    data = prepare_match_data(df, indexer)

    assert data.n_matches == 3
    assert data.n_teams == 3
    assert data.home_idx.dtype == np.int64
    assert data.home_advantage.dtype == np.float64
    # neutral=[F, T, F] → home_advantage=[1, 0, 1]
    np.testing.assert_array_equal(data.home_advantage, [1.0, 0.0, 1.0])


def test_prepare_match_data_rejects_missing_columns():
    df = pd.DataFrame({"home_team": ["Italy"], "away_team": ["Spain"]})
    indexer = TeamIndexer(["Italy", "Spain"])
    with pytest.raises(ValueError, match="Colonne richieste mancanti"):
        prepare_match_data(df, indexer)


def test_matchdata_rejects_incoherent_shapes():
    with pytest.raises(ValueError, match="lunghezza incoerente"):
        MatchData(
            home_idx=np.array([0, 1]),
            away_idx=np.array([1, 0, 1]),     # mismatch
            home_goals=np.array([0, 0]),
            away_goals=np.array([0, 0]),
            home_advantage=np.array([1.0, 1.0]),
            n_teams=2,
        )
