"""Test del grid search di emivita (Issue #9, DoD #3).

Usa un mini-dataset sintetico (5 squadre, 200 partite) per eseguire fit reali in
pochi secondi. Il grid è di soli 2 valori per limitare il tempo a < 30 secondi.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.validation.grid_search import (
    GridSearchResult,
    fit_at_cutoff,
    grid_search_half_life,
)
from src.validation.temporal_split import temporal_split


@pytest.fixture(scope="module")
def mini_dataset() -> pd.DataFrame:
    """Mini dataset sintetico: 5 squadre, ~200 match Friendly, esiti casuali."""
    rng = np.random.default_rng(42)
    teams = ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE"]
    n_matches = 200
    # Genera date sparse in [2018-01-01, 2023-12-31]
    start = pd.Timestamp("2018-01-01")
    days_range = (pd.Timestamp("2023-12-31") - start).days
    days_offsets = rng.integers(0, days_range, n_matches)
    dates = sorted([start + pd.Timedelta(days=int(d)) for d in days_offsets])
    home_idx = rng.integers(0, 5, n_matches)
    away_idx = rng.integers(0, 5, n_matches)
    same = home_idx == away_idx
    away_idx[same] = (away_idx[same] + 1) % 5
    # Punteggi casuali ma con forza eterogenea
    strength = np.array([0.5, 0.2, 0.0, -0.2, -0.5])
    log_lam = strength[home_idx] - strength[away_idx] + 0.3
    log_mu = strength[away_idx] - strength[home_idx]
    home_scores = rng.poisson(np.exp(log_lam))
    away_scores = rng.poisson(np.exp(log_mu))
    return pd.DataFrame({
        "date": dates,
        "home_team": [teams[i] for i in home_idx],
        "away_team": [teams[i] for i in away_idx],
        "home_score": home_scores,
        "away_score": away_scores,
        "tournament": ["Friendly"] * n_matches,
        "neutral": [False] * n_matches,
    })


def test_fit_at_cutoff_returns_model_with_reference_date(mini_dataset):
    cfg = load_config()
    cutoff = pd.Timestamp("2022-06-30")
    split = temporal_split(mini_dataset, cutoff)
    model, indexer = fit_at_cutoff(
        split.train_df, cutoff, cfg, half_life_years=2.0,
        max_iter=200, max_fun=100_000,
    )
    # DoD #2: reference_date dei pesi = cutoff_date dello split
    assert model.reference_date == str(cutoff.date())
    # Coerenza dei parametri
    assert model.half_life_years == 2.0
    assert model.n_teams == indexer.n_teams
    assert model.gamma > 0


def test_grid_search_returns_one_row_per_half_life(mini_dataset):
    cfg = load_config()
    result = grid_search_half_life(
        mini_dataset, "2022-06-30", cfg,
        half_life_grid=[1.5, 2.5],
        max_iter=200, max_fun=100_000,
    )
    assert isinstance(result, GridSearchResult)
    assert len(result.half_life_results) == 2
    assert set(result.half_life_results["half_life_years"]) == {1.5, 2.5}
    expected_cols = {"half_life_years", "log_loss", "brier_score", "accuracy",
                     "n_eval", "n_skipped", "gamma", "rho", "converged"}
    assert expected_cols.issubset(set(result.half_life_results.columns))


def test_grid_search_picks_best_log_loss(mini_dataset):
    cfg = load_config()
    result = grid_search_half_life(
        mini_dataset, "2022-06-30", cfg,
        half_life_grid=[1.5, 2.5],
        max_iter=200, max_fun=100_000,
    )
    # DoD #3: best_half_life corrisponde al minimo della log_loss
    df = result.half_life_results
    best_row = df.loc[df["log_loss"].idxmin()]
    assert result.best_half_life == best_row["half_life_years"]
    assert result.best_log_loss == best_row["log_loss"]


def test_grid_search_log_loss_finite_on_mini_dataset(mini_dataset):
    cfg = load_config()
    result = grid_search_half_life(
        mini_dataset, "2022-06-30", cfg,
        half_life_grid=[2.0],
        max_iter=200, max_fun=100_000,
    )
    ll = result.half_life_results["log_loss"].iloc[0]
    assert math.isfinite(ll)
    # Modello sensato deve fare meglio del baseline uniforme (log 3)
    # Sul mini-dataset non e' garantito, ma <= 1.5 e' realistico
    assert ll < 1.5
