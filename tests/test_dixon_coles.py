"""Test della likelihood Dixon-Coles (Issue #4).

Copre:
- DoD #1 (NLL finita e differenziabile sui dati): finita su sample sintetico + finite-diff.
- DoD #2 (test su mini-dataset sintetico, NLL minima vicino ai veri parametri).
- DoD #3 (τ implementata esattamente come da formula sui 4 casi speciali).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import poisson

from src.config import load_config
from src.model.dixon_coles import (
    decode_params,
    dixon_coles_nll,
    dixon_coles_nll_from_config,
    encode_params,
    initial_params,
    match_log_likelihood,
    n_params,
    tau,
)
from src.model.indexing import MatchData


# ---------------------------------------------------------------------------
# τ (correzione Dixon-Coles) — DoD #3
# ---------------------------------------------------------------------------

def test_tau_exact_at_four_special_cases():
    """τ deve riprodurre esattamente le 4 formule del piano."""
    rho = 0.07
    lam = np.array([1.3, 1.3, 1.3, 1.3], dtype=float)
    mu = np.array([0.9, 0.9, 0.9, 0.9], dtype=float)
    x = np.array([0, 0, 1, 1], dtype=int)
    y = np.array([0, 1, 0, 1], dtype=int)
    result = tau(x, y, lam, mu, rho)
    expected = np.array(
        [
            1.0 - lam[0] * mu[0] * rho,    # (0,0)
            1.0 + lam[1] * rho,            # (0,1)
            1.0 + mu[2] * rho,             # (1,0)
            1.0 - rho,                     # (1,1)
        ]
    )
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_tau_is_one_outside_special_cases():
    rho = 0.07
    lam = np.array([1.0, 1.0, 1.0, 1.0])
    mu = np.array([1.0, 1.0, 1.0, 1.0])
    x = np.array([2, 3, 0, 1])
    y = np.array([1, 4, 2, 5])
    result = tau(x, y, lam, mu, rho)
    np.testing.assert_array_equal(result, np.ones(4))


def test_tau_with_rho_zero_is_identity():
    """ρ=0 → τ=1 sempre (modello Poisson indipendente puro)."""
    rho = 0.0
    lam = np.array([1.0, 2.0, 0.5, 1.7])
    mu = np.array([0.7, 1.1, 1.4, 0.6])
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    result = tau(x, y, lam, mu, rho)
    np.testing.assert_allclose(result, np.ones(4), atol=1e-15)


# ---------------------------------------------------------------------------
# Encode / decode dei parametri
# ---------------------------------------------------------------------------

def test_n_params():
    assert n_params(10) == 22
    assert n_params(315) == 632


def test_encode_decode_roundtrip():
    n = 5
    rng = np.random.default_rng(0)
    alpha = rng.normal(0, 1, n)
    beta = rng.normal(0, 1, n)
    gamma, rho = 0.35, -0.08
    params = encode_params(alpha, beta, gamma, rho)
    assert params.shape == (n_params(n),)
    a2, b2, g2, r2 = decode_params(params, n)
    np.testing.assert_array_equal(a2, alpha)
    np.testing.assert_array_equal(b2, beta)
    assert g2 == gamma
    assert r2 == rho


def test_decode_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        decode_params(np.zeros(10), n_teams=5)   # atteso 12


def test_initial_params_shape_and_values():
    p = initial_params(8, gamma_init=0.3, rho_init=-0.05)
    a, b, g, r = decode_params(p, 8)
    np.testing.assert_array_equal(a, np.zeros(8))
    np.testing.assert_array_equal(b, np.zeros(8))
    assert g == 0.3
    assert r == -0.05


# ---------------------------------------------------------------------------
# Sanity log-Poisson
# ---------------------------------------------------------------------------

def test_log_poisson_matches_scipy():
    """Il log-Poisson della NLL deve coincidere con scipy.stats.poisson.logpmf."""
    from src.model.dixon_coles import _log_poisson_pmf
    k = np.array([0, 1, 2, 3, 5, 8], dtype=np.int64)
    mean = np.array([0.5, 1.0, 1.7, 2.3, 0.8, 3.2])
    expected = poisson.logpmf(k, mean)
    actual = _log_poisson_pmf(k.astype(float), mean)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Generatori di dati sintetici (fixtures)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_world():
    """Mini-dataset coerente con il modello DC (per la DoD #2)."""
    rng = np.random.default_rng(42)
    n_teams = 8
    n_matches = 3000
    alpha = rng.normal(0.0, 0.3, n_teams)
    alpha -= alpha.mean()  # centrato
    beta = rng.normal(0.0, 0.3, n_teams)
    beta -= beta.mean()
    gamma = 0.3
    rho = -0.1

    home_idx = rng.integers(0, n_teams, n_matches).astype(np.int64)
    away_idx = rng.integers(0, n_teams, n_matches).astype(np.int64)
    same = home_idx == away_idx
    away_idx[same] = (away_idx[same] + 1) % n_teams

    home_advantage = np.ones(n_matches, dtype=np.float64)
    log_lam = alpha[home_idx] + beta[away_idx] + gamma * home_advantage
    log_mu = alpha[away_idx] + beta[home_idx]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)
    home_goals = rng.poisson(lam).astype(np.int64)
    away_goals = rng.poisson(mu).astype(np.int64)

    data = MatchData(
        home_idx=home_idx,
        away_idx=away_idx,
        home_goals=home_goals,
        away_goals=away_goals,
        home_advantage=home_advantage,
        n_teams=n_teams,
    )
    weights = np.ones(n_matches, dtype=np.float64)
    true_params = encode_params(alpha, beta, gamma, rho)
    return data, weights, true_params


# ---------------------------------------------------------------------------
# DoD #1: NLL finita + differenziabile
# ---------------------------------------------------------------------------

def test_nll_finite_on_synthetic(synthetic_world):
    data, weights, true_params = synthetic_world
    nll = dixon_coles_nll(true_params, data, weights, identifiability_penalty_strength=1e4)
    assert math.isfinite(nll)
    assert nll > 0.0


def test_nll_finite_with_zero_weights(synthetic_world):
    """Pesi tutti a zero → NLL = penalty (0 se mean=0)."""
    data, _, true_params = synthetic_world
    zero_w = np.zeros(data.n_matches)
    nll = dixon_coles_nll(true_params, data, zero_w, identifiability_penalty_strength=0.0)
    assert nll == pytest.approx(0.0, abs=1e-9)


def test_nll_scales_linearly_with_weight(synthetic_world):
    """Raddoppiare i pesi raddoppia la parte data della NLL (penalty invariata)."""
    data, w, true_params = synthetic_world
    nll1 = dixon_coles_nll(true_params, data, w, identifiability_penalty_strength=0.0)
    nll2 = dixon_coles_nll(true_params, data, 2 * w, identifiability_penalty_strength=0.0)
    assert nll2 == pytest.approx(2 * nll1, rel=1e-10)


def test_nll_finite_difference_is_finite(synthetic_world):
    """DoD #1: la NLL è differenziabile in senso numerico (gradient finito)."""
    data, weights, true_params = synthetic_world
    f = lambda p: dixon_coles_nll(p, data, weights, identifiability_penalty_strength=1e4)
    eps = 1e-5
    for i in [0, data.n_teams, 2 * data.n_teams, 2 * data.n_teams + 1]:
        e = np.zeros_like(true_params)
        e[i] = eps
        df = (f(true_params + e) - f(true_params - e)) / (2 * eps)
        assert math.isfinite(df), f"Gradiente non-finito sull'indice {i}"


# ---------------------------------------------------------------------------
# Stabilità: clip τ ≤ 0 → NLL resta finita
# ---------------------------------------------------------------------------

def test_nll_stays_finite_when_tau_would_be_negative():
    """Costruisco scenario con λ·μ·ρ > 1 sul (0,0) → τ < 0 → NLL deve usare il floor."""
    data = MatchData(
        home_idx=np.array([0], dtype=np.int64),
        away_idx=np.array([1], dtype=np.int64),
        home_goals=np.array([0], dtype=np.int64),
        away_goals=np.array([0], dtype=np.int64),
        home_advantage=np.array([0.0]),
        n_teams=2,
    )
    # Parametri assurdi: alpha enormi → λ, μ enormi → λμρ > 1 anche con ρ piccolo
    params = encode_params(
        alpha=np.array([3.0, 3.0]),
        beta=np.array([3.0, 3.0]),
        gamma=0.0,
        rho=0.15,
    )
    weights = np.array([1.0])
    nll = dixon_coles_nll(params, data, weights, identifiability_penalty_strength=0.0, tau_floor=1e-10)
    assert math.isfinite(nll)


# ---------------------------------------------------------------------------
# Penalty di identificabilità
# ---------------------------------------------------------------------------

def test_nll_invariant_to_alpha_shift_when_penalty_zero(synthetic_world):
    """Senza penalty, l'invarianza DC (α+=c, β-=c) lascia la NLL data IDENTICA."""
    data, weights, true_params = synthetic_world
    alpha, beta, gamma, rho = decode_params(true_params, data.n_teams)
    c = 0.5
    shifted = encode_params(alpha + c, beta - c, gamma, rho)
    nll_true = dixon_coles_nll(true_params, data, weights, identifiability_penalty_strength=0.0)
    nll_shifted = dixon_coles_nll(shifted, data, weights, identifiability_penalty_strength=0.0)
    assert nll_shifted == pytest.approx(nll_true, rel=1e-10)


def test_penalty_breaks_the_shift_invariance(synthetic_world):
    """La penalty deve penalizzare la traslazione opposta α+=c, β-=c di esattamente λ_id·(c²+c²)."""
    data, weights, true_params = synthetic_world
    alpha, beta, gamma, rho = decode_params(true_params, data.n_teams)
    # I true_params sono già centrati (mean(α) ≈ 0, mean(β) ≈ 0)
    c = 0.5
    lam_id = 1e4
    shifted = encode_params(alpha + c, beta - c, gamma, rho)
    nll_true = dixon_coles_nll(true_params, data, weights, identifiability_penalty_strength=lam_id)
    nll_shifted = dixon_coles_nll(shifted, data, weights, identifiability_penalty_strength=lam_id)
    # NLL data invariato; penalty cresce da ~0 a λ_id·(c² + c²)
    expected_increase = lam_id * 2 * c**2
    assert (nll_shifted - nll_true) == pytest.approx(expected_increase, rel=1e-6)


# ---------------------------------------------------------------------------
# DoD #2: NLL minima vicino ai veri parametri
# ---------------------------------------------------------------------------

def test_nll_lower_at_true_than_at_random_perturbations(synthetic_world):
    """DoD #2: per perturbazioni random, NLL(perturbed) > NLL(true) nella stragrande maggioranza."""
    data, weights, true_params = synthetic_world
    nll_true = dixon_coles_nll(true_params, data, weights, identifiability_penalty_strength=1e4)

    rng = np.random.default_rng(7)
    higher = 0
    n_runs = 20
    for _ in range(n_runs):
        perturbation = rng.normal(0.0, 0.3, size=true_params.shape)
        perturbed = true_params + perturbation
        # Mantieni rho nel range bound; clip per evitare NLL = +inf per altri motivi
        perturbed[-1] = np.clip(perturbed[-1], -0.19, 0.19)
        nll_p = dixon_coles_nll(perturbed, data, weights, identifiability_penalty_strength=1e4)
        if nll_p > nll_true:
            higher += 1
    assert higher >= 18, f"Solo {higher}/{n_runs} perturbazioni aumentano la NLL"


def test_nll_lower_at_true_than_at_uniform_alpha(synthetic_world):
    """Settare tutti gli α a 0 (un punto MOLTO lontano dai veri) deve aumentare la NLL."""
    data, weights, true_params = synthetic_world
    alpha, beta, gamma, rho = decode_params(true_params, data.n_teams)
    zero_alpha_params = encode_params(np.zeros_like(alpha), beta, gamma, rho)
    nll_true = dixon_coles_nll(true_params, data, weights, identifiability_penalty_strength=1e4)
    nll_zero = dixon_coles_nll(zero_alpha_params, data, weights, identifiability_penalty_strength=1e4)
    assert nll_zero > nll_true


# ---------------------------------------------------------------------------
# Wrapper from-config
# ---------------------------------------------------------------------------

def test_nll_from_config_matches_explicit(synthetic_world):
    cfg = load_config()
    data, weights, true_params = synthetic_world
    nll_explicit = dixon_coles_nll(
        true_params,
        data,
        weights,
        identifiability_penalty_strength=cfg.dixon_coles.identifiability_penalty_strength,
        tau_floor=cfg.dixon_coles.tau_floor,
    )
    nll_wrapped = dixon_coles_nll_from_config(true_params, data, weights, cfg)
    assert nll_explicit == pytest.approx(nll_wrapped, rel=1e-12)


# ---------------------------------------------------------------------------
# match_log_likelihood: shape e firme
# ---------------------------------------------------------------------------

def test_match_log_likelihood_shape(synthetic_world):
    data, _, true_params = synthetic_world
    alpha, beta, gamma, rho = decode_params(true_params, data.n_teams)
    ll = match_log_likelihood(alpha, beta, gamma, rho, data)
    assert ll.shape == (data.n_matches,)
    assert np.isfinite(ll).all()
    # log P ∈ (-∞, 0]
    assert (ll <= 1e-9).all()


def test_nll_rejects_weights_with_wrong_shape(synthetic_world):
    data, _, true_params = synthetic_world
    with pytest.raises(ValueError, match="weights"):
        dixon_coles_nll(true_params, data, np.ones(data.n_matches + 1))
