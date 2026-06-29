"""Smoke test del loader di configurazione (Issue #0, DoD #2 e #3)."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import AppConfig, DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads_and_is_typed():
    config = load_config()
    assert isinstance(config, AppConfig)


def test_default_config_path_exists():
    assert DEFAULT_CONFIG_PATH.exists(), (
        f"config.yaml non trovato in {DEFAULT_CONFIG_PATH}"
    )


def test_fixed_values_match_handoff():
    """I valori 'fissati' (sezione 1 di docs/CLAUDE.md) non devono drifte."""
    config = load_config()
    assert config.perimeter.scope == "knockout_only"
    assert config.competition_weights.amichevole == 0.4
    assert config.competition_weights.mondiali == 1.0
    assert config.competition_weights.qualificazioni == 0.8
    assert config.competition_weights.sub_continentali == 0.6
    assert config.competition_weights.default_unmapped == 0.6
    assert config.kelly.fraction == 0.25
    assert config.penalty_shootout.base_prob_winner == 0.50
    assert config.include_friendlies is True
    assert config.outputs.enable_output_a_calibrated is True
    assert config.outputs.enable_output_b_pure_model is True


def test_open_defaults_match_handoff():
    config = load_config()
    assert config.time_decay.half_life_years == 2.0
    assert config.betting.min_edge_threshold == pytest.approx(0.03)


def test_dixon_coles_defaults():
    config = load_config()
    assert config.dixon_coles.identifiability_penalty_strength == pytest.approx(1e4)
    assert config.dixon_coles.tau_floor == pytest.approx(1e-10)
    assert config.goals.rho_bounds == (-0.2, 0.2)


def test_host_policy_2026():
    config = load_config()
    assert set(config.host_advantage_2026.host_countries) == {"USA", "Canada", "Mexico"}
    assert config.host_advantage_2026.gamma_full_at_home > config.host_advantage_2026.gamma_reduced_co_host
    assert config.host_advantage_2026.gamma_neutral == 0.0


def test_extra_time_factors_one_third():
    config = load_config()
    assert config.extra_time.lambda_factor == pytest.approx(1 / 3, rel=1e-6)
    assert config.extra_time.mu_factor == pytest.approx(1 / 3, rel=1e-6)


def test_paths_are_relative_to_repo_root():
    config = load_config()
    assert config.paths.data_raw == Path("data/raw")
    assert config.paths.data_processed == Path("data/processed")


def test_unknown_keys_rejected(tmp_path: Path):
    """Pydantic in modalità strict deve rifiutare chiavi sconosciute (early failure)."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("kelly:\n  fraction: 0.25\n  unknown_field: 42\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad_yaml)


def test_invalid_value_rejected(tmp_path: Path):
    """Una probabilità fuori [0,1] deve far fallire la validazione."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("penalty_shootout:\n  base_prob_winner: 1.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad_yaml)
