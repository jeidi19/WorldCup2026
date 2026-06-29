"""Test della pulizia del dataset (Issue #2).

Copre le due DoD:
- Nessun NaN sui campi chiave dopo pulizia.
- Date monotone ordinabili dopo sort.

Più test specifici di policy: deny-list applicata, entità estinte mantenute,
match_id deterministico, gestione collisioni.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import clean as cm


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """Mini-dataset che esercita tutti i casi limite."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "1990-06-08",  # Italy-Yugoslavia (estinta, da mantenere)
                    "2024-03-26",  # Italy-Ecuador
                    "2010-05-15",  # Sealand-Sark (entrambe in deny-list, droppare)
                    "2026-06-30",  # France-Sweden (NaN, futura, droppare)
                    "2024-03-26",  # Italy-Ecuador duplicato stesso giorno (suffisso match_id)
                    "2000-04-01",  # Italy-Catalonia (Catalonia in deny-list, droppare)
                ]
            ),
            "home_team": ["Italy", "Italy", "Sealand", "France", "Italy", "Italy"],
            "away_team": ["Yugoslavia", "Ecuador", "Sark", "Sweden", "Ecuador", "Catalonia"],
            "home_score": [1.0, 2.0, 3.0, float("nan"), 2.0, 4.0],
            "away_score": [0.0, 0.0, 1.0, float("nan"), 0.0, 0.0],
            "tournament": ["Friendly"] * 6,
            "city": ["Rome", "New York", "Sealand", "Paris", "Genova", "Barcelona"],
            "country": ["Italy", "United States", "Sealand", "France", "Italy", "Spain"],
            "neutral": [False, True, False, False, False, False],
        }
    )


def test_drop_na_scores_casts_to_int(raw_sample):
    df = cm.drop_na_scores(raw_sample)
    assert df["home_score"].isna().sum() == 0
    assert df["away_score"].isna().sum() == 0
    assert df["home_score"].dtype == "int64"
    assert df["away_score"].dtype == "int64"
    # La riga NaN (France-Sweden 2026-06-30) è stata droppata
    assert not ((df["home_team"] == "France") & (df["away_team"] == "Sweden")).any()


def test_apply_deny_list_drops_non_fifa(raw_sample):
    df = cm.apply_deny_list(raw_sample)
    assert not (df["home_team"] == "Sealand").any()
    assert not (df["away_team"] == "Sark").any()
    assert not (df["away_team"] == "Catalonia").any()


def test_apply_deny_list_keeps_extinct(raw_sample):
    """Yugoslavia è estinta ma NON in deny-list: deve sopravvivere."""
    df = cm.apply_deny_list(raw_sample)
    assert ((df["home_team"] == "Italy") & (df["away_team"] == "Yugoslavia")).any()


def test_coerce_neutral_returns_bool(raw_sample):
    df = cm.coerce_neutral(raw_sample)
    assert df["neutral"].dtype == bool


def test_match_id_is_deterministic(raw_sample):
    df1 = cm.add_match_id(raw_sample.copy())
    df2 = cm.add_match_id(raw_sample.copy())
    assert (df1["match_id"].to_numpy() == df2["match_id"].to_numpy()).all()


def test_match_id_disambiguates_collisions(raw_sample):
    df = cm.add_match_id(raw_sample.copy())
    italy_ecuador = df[(df["home_team"] == "Italy") & (df["away_team"] == "Ecuador")]
    assert len(italy_ecuador) == 2
    assert italy_ecuador["match_id"].nunique() == 2


def test_full_pipeline_dod(raw_sample):
    """DoD #2: nessun NaN sui campi chiave + date monotone ordinabili."""
    df = cm.clean(raw_sample)
    key_cols = ["date", "home_team", "away_team", "home_score", "away_score", "neutral"]
    for col in key_cols:
        assert df[col].isna().sum() == 0, f"NaN in {col}"
    assert df["date"].is_monotonic_increasing
    assert df["match_id"].is_unique


def test_full_pipeline_preserves_extinct(raw_sample):
    df = cm.clean(raw_sample)
    assert (df["away_team"] == "Yugoslavia").any()


def test_load_raw_rejects_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("date,home_team\n2020-01-01,Italy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Colonne mancanti"):
        cm.load_raw(bad_csv)
