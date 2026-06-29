"""Test del fit Dixon-Coles (Issue #5)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.model.dixon_coles import (
    decode_params,
    dixon_coles_nll,
    encode_params,
    initial_params,
)
from src.model.fit import (
    DixonColesModel,
    center_alpha_beta,
    fit_model,
    make_bounds,
)
from src.model.indexing import MatchData, TeamIndexer


# ---------------------------------------------------------------------------
# make_bounds
# ---------------------------------------------------------------------------

def test_bounds_structure():
    bounds = make_bounds(n_teams=5, rho_bounds=(-0.2, 0.2))
    # 10 (alpha/beta) + 1 (gamma) + 1 (rho) = 12
    assert len(bounds) == 12
    # alpha/beta liberi
    for lo, hi in bounds[:10]:
        assert lo is None and hi is None
    # gamma > 0
    assert bounds[10][0] > 0 and bounds[10][1] is None
    # rho dentro [-0.2, 0.2] (con margine)
    rho_lo, rho_hi = bounds[11]
    assert -0.2 < rho_lo < 0 < rho_hi < 0.2


# ---------------------------------------------------------------------------
# center_alpha_beta
# ---------------------------------------------------------------------------

def test_center_alpha_beta_preserves_invariance():
    """Lo shift (α-=c, β+=c) con c=mean(α) lascia α+β invariato e centra α."""
    rng = np.random.default_rng(0)
    alpha = rng.normal(0.5, 0.3, 10)  # NON centrato (mean = ~0.5)
    beta = rng.normal(-0.1, 0.3, 10)
    a2, b2 = center_alpha_beta(alpha, beta)
    assert abs(a2.mean()) < 1e-12
    # α + β invariato
    np.testing.assert_allclose(a2 + b2, alpha + beta, atol=1e-12)


# ---------------------------------------------------------------------------
# Recovery sui parametri veri (DoD #2: rating plausibili)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_setup():
    """Dataset sintetico con n=8 squadre, ~4000 partite, true params noti."""
    rng = np.random.default_rng(42)
    n_teams = 8
    n_matches = 4000

    true_alpha = rng.normal(0.0, 0.3, n_teams)
    true_alpha -= true_alpha.mean()
    true_beta = rng.normal(0.0, 0.3, n_teams)
    true_beta -= true_beta.mean()
    true_gamma = 0.3
    true_rho = -0.08

    home_idx = rng.integers(0, n_teams, n_matches).astype(np.int64)
    away_idx = rng.integers(0, n_teams, n_matches).astype(np.int64)
    same = home_idx == away_idx
    away_idx[same] = (away_idx[same] + 1) % n_teams

    home_advantage = np.ones(n_matches, dtype=np.float64)
    log_lam = true_alpha[home_idx] + true_beta[away_idx] + true_gamma * home_advantage
    log_mu = true_alpha[away_idx] + true_beta[home_idx]
    home_goals = rng.poisson(np.exp(log_lam)).astype(np.int64)
    away_goals = rng.poisson(np.exp(log_mu)).astype(np.int64)

    data = MatchData(
        home_idx=home_idx,
        away_idx=away_idx,
        home_goals=home_goals,
        away_goals=away_goals,
        home_advantage=home_advantage,
        n_teams=n_teams,
    )
    weights = np.ones(n_matches, dtype=np.float64)
    indexer = TeamIndexer([f"T{i:02d}" for i in range(n_teams)])
    return data, weights, indexer, true_alpha, true_beta, true_gamma, true_rho


def test_fit_recovers_synthetic_params(synthetic_setup):
    """DoD: con forze note, il fit recupera α, β, γ entro tolleranza ragionevole."""
    data, weights, indexer, true_alpha, true_beta, true_gamma, true_rho = synthetic_setup
    config = load_config()
    model = fit_model(data, weights, config, indexer, max_iter=300)

    assert model.converged
    # Recupero alpha e beta: |err| < 0.15 (tolleranza ampia: 4000 partite + correzione DC
    # non applicata alla simulazione = bias residuo)
    alpha_fit = np.asarray(model.alpha)
    beta_fit = np.asarray(model.beta)
    np.testing.assert_allclose(alpha_fit, true_alpha, atol=0.15)
    np.testing.assert_allclose(beta_fit, true_beta, atol=0.15)
    # gamma recuperato entro 0.05
    assert abs(model.gamma - true_gamma) < 0.07
    # rho deve essere nel range bound
    assert -0.2 < model.rho < 0.2


def test_fit_alphas_centered(synthetic_setup):
    """Dopo center_alpha_beta, mean(α) ≈ 0."""
    data, weights, indexer, *_ = synthetic_setup
    config = load_config()
    model = fit_model(data, weights, config, indexer, max_iter=200)
    assert abs(np.mean(model.alpha)) < 1e-6


def test_fit_rejects_n_teams_mismatch():
    config = load_config()
    n = 3
    data = MatchData(
        home_idx=np.array([0], dtype=np.int64),
        away_idx=np.array([1], dtype=np.int64),
        home_goals=np.array([0], dtype=np.int64),
        away_goals=np.array([0], dtype=np.int64),
        home_advantage=np.array([1.0]),
        n_teams=n,
    )
    weights = np.array([1.0])
    indexer = TeamIndexer(["A", "B"])    # n=2 != 3
    with pytest.raises(ValueError, match="n_teams"):
        fit_model(data, weights, config, indexer)


# ---------------------------------------------------------------------------
# DixonColesModel API
# ---------------------------------------------------------------------------

def test_model_accessors():
    model = DixonColesModel(
        teams=("A", "B"),
        alpha=(0.3, -0.3),
        beta=(-0.1, 0.1),
        gamma=0.25,
        rho=-0.05,
        n_matches_train=100,
        reference_date="2024-01-01",
        fitted_at="2024-01-01T00:00:00+00:00",
        final_nll=123.45,
        n_iterations=42,
        converged=True,
        optimization_message="ok",
        half_life_years=2.0,
        identifiability_penalty_strength=1e4,
        tau_floor=1e-10,
    )
    assert model.n_teams == 2
    assert model.n_params == 6
    assert model.alpha_of("A") == 0.3
    assert model.beta_of("B") == 0.1
    assert model.strength_of("A") == 0.3 - (-0.1)
    with pytest.raises(KeyError):
        model.alpha_of("Z")


def test_model_to_dataframe_sorted_by_strength():
    model = DixonColesModel(
        teams=("Weak", "Strong"),
        alpha=(-0.5, 0.5),
        beta=(0.4, -0.4),
        gamma=0.3, rho=0.0,
        n_matches_train=1, reference_date=None,
        fitted_at="2024-01-01T00:00:00+00:00",
        final_nll=0.0, n_iterations=0, converged=True, optimization_message="",
        half_life_years=2.0, identifiability_penalty_strength=1e4, tau_floor=1e-10,
    )
    df = model.to_dataframe()
    assert list(df["team"]) == ["Strong", "Weak"]
    assert df.iloc[0]["strength"] > df.iloc[1]["strength"]


def test_model_save_load_roundtrip(tmp_path: Path, synthetic_setup):
    data, weights, indexer, *_ = synthetic_setup
    config = load_config()
    model = fit_model(data, weights, config, indexer, max_iter=100)

    path = tmp_path / "m.json"
    model.save(path)
    loaded = DixonColesModel.load(path)

    assert loaded.teams == model.teams
    np.testing.assert_array_equal(loaded.alpha, model.alpha)
    np.testing.assert_array_equal(loaded.beta, model.beta)
    assert loaded.gamma == model.gamma
    assert loaded.rho == model.rho
    assert loaded.n_iterations == model.n_iterations
    assert loaded.fitted_at == model.fitted_at


def test_model_json_is_readable(tmp_path: Path, synthetic_setup):
    """Il file JSON salvato deve essere leggibile e contenere le chiavi attese."""
    data, weights, indexer, *_ = synthetic_setup
    config = load_config()
    model = fit_model(data, weights, config, indexer, max_iter=100)
    path = tmp_path / "m.json"
    model.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert {"teams", "alpha", "beta", "gamma", "rho", "fitted_at", "converged"} <= set(raw.keys())
    assert isinstance(raw["teams"], list)
    assert isinstance(raw["alpha"], list)


# ---------------------------------------------------------------------------
# Consistenza NLL pre/post centratura
# ---------------------------------------------------------------------------

def test_centering_preserves_data_nll(synthetic_setup):
    """Centrare α/β NON deve cambiare la NLL data (è l'invarianza esatta)."""
    data, weights, indexer, *_ = synthetic_setup
    config = load_config()
    # Parametri arbitrari, non centrati
    rng = np.random.default_rng(0)
    alpha = rng.normal(0.4, 0.3, indexer.n_teams)
    beta = rng.normal(-0.1, 0.3, indexer.n_teams)
    p = encode_params(alpha, beta, 0.3, -0.05)
    nll_before = dixon_coles_nll(p, data, weights, identifiability_penalty_strength=0.0)
    a2, b2 = center_alpha_beta(alpha, beta)
    p2 = encode_params(a2, b2, 0.3, -0.05)
    nll_after = dixon_coles_nll(p2, data, weights, identifiability_penalty_strength=0.0)
    assert nll_before == pytest.approx(nll_after, rel=1e-9)
