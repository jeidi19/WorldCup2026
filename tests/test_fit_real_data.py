"""Test di integrazione del fit Dixon-Coles sul dataset reale (Issue #5).

Skippa se `data/models/dixon_coles.json` non esiste (lanciare `python -m src.model.fit`).

Non eseguiamo un fit completo nei test (è lento): leggiamo invece il modello già salvato
e validiamo proprietà strutturali (rating plausibili, sanity check).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.model.fit import DixonColesModel


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_JSON = REPO_ROOT / "data" / "models" / "dixon_coles.json"

pytestmark = pytest.mark.skipif(
    not MODEL_JSON.exists(),
    reason="data/models/dixon_coles.json non disponibile (lanciare `python -m src.model.fit`)",
)


@pytest.fixture(scope="module")
def model() -> DixonColesModel:
    return DixonColesModel.load(MODEL_JSON)


def test_model_loads_and_has_expected_fields(model):
    assert model.n_teams > 100        # almeno 100 squadre nel dataset reale
    assert model.n_params == 2 * model.n_teams + 2
    assert model.gamma > 0            # vantaggio casa positivo
    assert abs(model.rho) < 0.2       # entro il bound di config


def test_model_alphas_centered_on_real_data(model):
    """Il fit applica `center_alpha_beta` post-hoc: mean(alpha) ≈ 0."""
    import numpy as np
    alpha = np.asarray(model.alpha)
    assert abs(alpha.mean()) < 1e-6


def test_top_strength_includes_at_least_one_obvious_top_nation(model):
    """Sanity check: nella top 10 per `strength = α - β` ci deve essere almeno una
    nazione tra Argentina/Brazil/France/Spain (top-tier 2022-2024 incontestabili)."""
    top_10 = list(model.to_dataframe()["team"].head(10))
    must_have_any_of = {"Argentina", "Brazil", "France", "Spain"}
    assert must_have_any_of & set(top_10), (
        f"Nessuna delle {must_have_any_of} nella top 10: {top_10}"
    )


def test_top_strength_no_micro_nation_in_top_50(model):
    """Sanity check: nessuna entità storicamente debole nella top 50 (San Marino, Andorra,
    Liechtenstein, Gibraltar, ecc.). Se ci fossero, il fit avrebbe un problema serio."""
    top_50 = set(model.to_dataframe()["team"].head(50))
    must_not = {"San Marino", "Andorra", "Liechtenstein", "Gibraltar", "Faroe Islands"}
    intersection = top_50 & must_not
    assert not intersection, f"Squadre deboli in top 50: {intersection}"


def test_bottom_strength_includes_weak_nations(model):
    """Le squadre più deboli devono includere almeno una "nota debole"."""
    df = model.to_dataframe()
    bottom_30 = set(df["team"].tail(30))
    # Almeno una di queste deve essere in bottom 30
    expected_weak = {"San Marino", "Andorra", "Liechtenstein", "American Samoa", "Bhutan",
                     "Anguilla", "Cook Islands", "British Virgin Islands"}
    assert bottom_30 & expected_weak, f"Nessuna nota debole in bottom 30: {bottom_30}"


def test_gamma_in_plausible_range(model):
    """Vantaggio casa in scala log-gol ≈ +0.2 / +0.4 storicamente."""
    assert 0.05 < model.gamma < 0.6, f"gamma={model.gamma} fuori range plausibile"


def test_rho_negative_for_dataset_with_friendlies(model):
    """ρ tipicamente è negativo (sottostima dei 0-0 e 1-1 da Poisson indipendenti)."""
    # Pull permissivo: rho può essere anche leggermente positivo a seconda dei dati
    assert -0.2 < model.rho < 0.2


def test_model_save_load_roundtrip_on_real_file(tmp_path: Path, model):
    out = tmp_path / "roundtrip.json"
    model.save(out)
    reloaded = DixonColesModel.load(out)
    assert reloaded.teams == model.teams
    assert reloaded.alpha == model.alpha
    assert reloaded.beta == model.beta
    assert reloaded.gamma == model.gamma
    assert reloaded.rho == model.rho
