"""Test dello split temporale (Issue #9, DoD #1: anti-leakage automatico)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.validation.temporal_split import TemporalSplit, temporal_split


def _df(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "home_team": [f"H{i}" for i in range(len(dates))],
        "away_team": [f"A{i}" for i in range(len(dates))],
        "home_score": [1] * len(dates),
        "away_score": [0] * len(dates),
        "neutral": [False] * len(dates),
    })


def test_split_separates_correctly():
    df = _df(["2019-01-01", "2020-06-01", "2021-03-15", "2022-11-30"])
    split = temporal_split(df, "2020-12-31")
    assert split.n_train == 2
    assert split.n_test == 2
    assert set(split.train_df["date"]) == {pd.Timestamp("2019-01-01"), pd.Timestamp("2020-06-01")}


def test_split_cutoff_is_inclusive_for_train():
    df = _df(["2020-12-31", "2021-01-01"])
    split = temporal_split(df, "2020-12-31")
    assert split.n_train == 1
    assert split.train_df["date"].iloc[0] == pd.Timestamp("2020-12-31")
    assert split.n_test == 1
    assert split.test_df["date"].iloc[0] == pd.Timestamp("2021-01-01")


def test_split_anti_leakage_automatic():
    """DoD #1: nessuna partita di test entra nel training."""
    df = _df(["2019-01-01", "2020-01-01", "2021-01-01"])
    split = temporal_split(df, "2020-06-01")
    # Anti-leakage check viene eseguito in __post_init__
    assert split.train_df["date"].max() <= split.cutoff_date
    assert split.test_df["date"].min() > split.cutoff_date


def test_split_rejects_empty_train():
    df = _df(["2025-01-01", "2025-06-01"])
    with pytest.raises(ValueError, match="Train set vuoto"):
        temporal_split(df, "2020-01-01")


def test_split_rejects_empty_test():
    df = _df(["2019-01-01", "2020-01-01"])
    with pytest.raises(ValueError, match="Test set vuoto"):
        temporal_split(df, "2025-01-01")


def test_split_rejects_missing_date_column():
    df = pd.DataFrame({"home_team": ["A"]})
    with pytest.raises(ValueError, match="'date'"):
        temporal_split(df, "2020-01-01")


def test_split_sorted_chronologically():
    """Le partite del train e del test sono ordinate per data crescente."""
    df = _df(["2021-03-15", "2019-01-01", "2020-06-01", "2022-11-30"])
    split = temporal_split(df, "2020-12-31")
    assert (split.train_df["date"].diff().dropna() >= pd.Timedelta(0)).all()
    assert (split.test_df["date"].diff().dropna() >= pd.Timedelta(0)).all()


def test_temporalsplit_post_init_blocks_manual_leakage():
    """Costruzione manuale di TemporalSplit con leakage deve fallire."""
    train = _df(["2020-12-31", "2025-01-01"])     # 2025 > cutoff 2020-12-31 → leakage
    test = _df(["2021-01-01"])
    with pytest.raises(AssertionError, match="Leakage"):
        TemporalSplit(cutoff_date=pd.Timestamp("2020-12-31"), train_df=train, test_df=test)
