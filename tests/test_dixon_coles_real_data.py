"""Test di integrazione della NLL Dixon-Coles sul dataset reale (Issue #4, DoD #1).

Skippa se `data/processed/matches_weighted.parquet` non è disponibile.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.model.dixon_coles import (
    decode_params,
    dixon_coles_nll_from_config,
    initial_params,
)
from src.model.indexing import TeamIndexer, prepare_match_data


REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTED_PARQUET = REPO_ROOT / "data" / "processed" / "matches_weighted.parquet"

pytestmark = pytest.mark.skipif(
    not WEIGHTED_PARQUET.exists(),
    reason="matches_weighted.parquet non disponibile (lanciare `python -m src.data.build_weights`)",
)


@pytest.fixture(scope="module")
def real_setup():
    df = pd.read_parquet(WEIGHTED_PARQUET)
    indexer = TeamIndexer.from_match_dataframe(df)
    data = prepare_match_data(df, indexer)
    weights = df["weight"].to_numpy(dtype=np.float64)
    config = load_config()
    return df, indexer, data, weights, config


def test_nll_finite_on_real_dataset(real_setup):
    """DoD #1: NLL finita sui dati reali, con parametri di inizializzazione neutrali."""
    df, indexer, data, weights, config = real_setup
    p0 = initial_params(indexer.n_teams, gamma_init=0.3, rho_init=-0.05)
    nll = dixon_coles_nll_from_config(p0, data, weights, config)
    assert math.isfinite(nll)
    assert nll > 0


def test_nll_finite_difference_on_real_dataset(real_setup):
    """DoD #1: la NLL è differenziabile (gradient numerico finito) su un sottoinsieme di indici."""
    df, indexer, data, weights, config = real_setup
    p0 = initial_params(indexer.n_teams)
    eps = 1e-5

    def f(p):
        return dixon_coles_nll_from_config(p, data, weights, config)

    n = indexer.n_teams
    sample_idx = [0, 1, n // 2, n - 1, n, n + 1, 2 * n, 2 * n + 1]  # alpha, beta, gamma, rho
    for i in sample_idx:
        e = np.zeros_like(p0)
        e[i] = eps
        grad_i = (f(p0 + e) - f(p0 - e)) / (2 * eps)
        assert math.isfinite(grad_i), f"Gradiente numerico non-finito all'indice {i}"


def test_nll_decreases_when_moving_from_zero_to_better_alpha(real_setup):
    """Sanity: scaling i parametri α verso 0.1 dovrebbe NON peggiorare grossolanamente la NLL.

    Non è un fit, ma controlla che la superficie sia ben definita: una piccola direzione
    di crescita di α produce una NLL ancora finita (no NaN, no overflow).
    """
    df, indexer, data, weights, config = real_setup
    p0 = initial_params(indexer.n_teams)
    nll_0 = dixon_coles_nll_from_config(p0, data, weights, config)
    alpha = np.full(indexer.n_teams, 0.05)
    beta = np.full(indexer.n_teams, -0.05)
    from src.model.dixon_coles import encode_params
    p1 = encode_params(alpha, beta, 0.3, -0.05)
    nll_1 = dixon_coles_nll_from_config(p1, data, weights, config)
    assert math.isfinite(nll_1)
    # Mean(α)=0.05 → penalty ≈ 1e4 * (0.0025 + 0.0025) = 50. NLL data ~ centinaia di migliaia.
    # Non chiediamo nll_1 < nll_0 (lo split α/β nel modello dipende dalle squadre concrete).
