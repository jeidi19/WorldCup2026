"""Test delle metriche di valutazione (Issue #9)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.model.fit import DixonColesModel
from src.validation.evaluation import (
    evaluate_outcomes_90,
    log_loss_constant_baseline,
    log_loss_uniform_baseline,
)


def _test_df(matches: list[tuple[str, str, int, int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01"] * len(matches)),
            "home_team": [m[0] for m in matches],
            "away_team": [m[1] for m in matches],
            "home_score": [m[2] for m in matches],
            "away_score": [m[3] for m in matches],
            "neutral": [m[4] for m in matches],
        }
    )


def _flat_model(teams: tuple[str, ...]) -> DixonColesModel:
    """Modello con α=β=0, γ=0, ρ=0: ogni match in neutro dà P quasi simmetriche."""
    n = len(teams)
    return DixonColesModel(
        teams=teams,
        alpha=tuple([0.0] * n), beta=tuple([0.0] * n),
        gamma=0.0, rho=0.0,
        n_matches_train=0, reference_date=None,
        fitted_at="2024-01-01T00:00:00+00:00",
        final_nll=0.0, n_iterations=0, converged=True, optimization_message="",
        half_life_years=2.0, identifiability_penalty_strength=1e4, tau_floor=1e-10,
    )


# ---------------------------------------------------------------------------
# log_loss_uniform_baseline & log_loss_constant_baseline
# ---------------------------------------------------------------------------

def test_log_loss_uniform_equals_log_three():
    assert log_loss_uniform_baseline() == pytest.approx(math.log(3.0), abs=1e-15)


def test_log_loss_constant_uniform_matches_log_three():
    df = _test_df([("A", "B", 1, 0, True), ("C", "D", 0, 0, True), ("E", "F", 0, 1, True)])
    ll = log_loss_constant_baseline(df, 1 / 3, 1 / 3, 1 / 3)
    assert ll == pytest.approx(math.log(3.0), rel=1e-12)


def test_log_loss_constant_perfect_is_zero():
    """Se la baseline assegna P=1 all'esito vero, log-loss = 0."""
    # Costruisco 3 match con label noto. Per ottenere log-loss = 0, devo usare
    # SOLO match dove l'esito coincide con la massa concentrata: tutti "home wins".
    df = _test_df([("A", "B", 1, 0, True), ("C", "D", 2, 0, True)])
    ll = log_loss_constant_baseline(df, 1.0, 0.0, 0.0)
    assert ll == 0.0


def test_log_loss_constant_rejects_unnormalized():
    df = _test_df([("A", "B", 1, 0, True)])
    with pytest.raises(ValueError, match="sommare a 1"):
        log_loss_constant_baseline(df, 0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# evaluate_outcomes_90
# ---------------------------------------------------------------------------

def test_evaluate_skips_unknown_teams():
    model = _flat_model(("A", "B"))
    df = _test_df([("A", "B", 1, 0, True),       # entrambi noti
                   ("A", "Z", 2, 1, True),       # Z sconosciuto -> skip
                   ("X", "Y", 0, 0, True)])      # entrambi sconosciuti -> skip
    metrics = evaluate_outcomes_90(model, df)
    assert metrics.n_matches_evaluated == 1
    assert metrics.n_matches_skipped == 2


def test_evaluate_returns_finite_metrics_on_flat_model():
    """Modello α=β=0 ρ=0 → P(home) ≈ P(away) ≈ 0.3-0.4, P(draw) ≈ 0.25-0.30 in neutro.
    Log-loss deve essere finita e ragionevole (≈ log 3 ± qualcosa).
    """
    model = _flat_model(("A", "B"))
    # Mix bilanciato di esiti per non favorire una classe
    df = _test_df([("A", "B", 1, 0, True), ("A", "B", 0, 0, True), ("A", "B", 0, 1, True)])
    metrics = evaluate_outcomes_90(model, df)
    assert 0.0 < metrics.log_loss < 2.0
    assert 0.0 < metrics.brier_score < 1.0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_evaluate_records_observed_frequencies():
    model = _flat_model(("A", "B"))
    df = _test_df([("A", "B", 1, 0, True), ("A", "B", 1, 0, True),   # 2 home wins
                   ("A", "B", 0, 0, True)])                            # 1 draw
    metrics = evaluate_outcomes_90(model, df)
    assert metrics.p_home_observed == pytest.approx(2 / 3)
    assert metrics.p_draw_observed == pytest.approx(1 / 3)
    assert metrics.p_away_observed == 0.0


def test_evaluate_rejects_missing_columns():
    model = _flat_model(("A", "B"))
    df = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
    with pytest.raises(ValueError, match="mancante delle colonne"):
        evaluate_outcomes_90(model, df)


def test_evaluate_rejects_all_skipped():
    model = _flat_model(("A", "B"))
    df = _test_df([("X", "Y", 1, 0, True)])
    with pytest.raises(ValueError, match="Nessun match valutabile"):
        evaluate_outcomes_90(model, df)


def test_log_loss_higher_for_wrong_predictions():
    """Modello che concentra massa sull'esito sbagliato deve avere log-loss alta."""
    # Modello con forte home-bias: assegna P(home)~0.9 quasi a tutti i match.
    # Lo costruisco settando alpha alto per "A" e basso per "B".
    n = 2
    model = DixonColesModel(
        teams=("A", "B"),
        alpha=(2.0, -2.0), beta=(-2.0, 2.0),
        gamma=0.5, rho=0.0,
        n_matches_train=0, reference_date=None,
        fitted_at="2024-01-01T00:00:00+00:00",
        final_nll=0.0, n_iterations=0, converged=True, optimization_message="",
        half_life_years=2.0, identifiability_penalty_strength=1e4, tau_floor=1e-10,
    )
    # Test set in cui "A" PERDE sistematicamente in casa (opposto al bias)
    df_wrong = _test_df([("A", "B", 0, 5, False)] * 5)
    metrics_wrong = evaluate_outcomes_90(model, df_wrong)
    # Test set in cui "A" vince come da bias
    df_right = _test_df([("A", "B", 5, 0, False)] * 5)
    metrics_right = evaluate_outcomes_90(model, df_right)
    assert metrics_wrong.log_loss > metrics_right.log_loss
