"""Metriche di valutazione del modello sul test set (Issue #9).

Multinomial log-loss + Brier multi-classe + accuracy sugli esiti dei 90 minuti
(home_win / draw / away_win). Standard nella letteratura DC e usato come obiettivo
del grid search di `ξ` (Issue #9) e `w` (Issue #13).

Per ogni partita del test set:
- `p = (P(home), P(draw), P(away))` da `match_outcome_90`;
- `label ∈ {0, 1, 2}` (one-hot) dall'esito reale;
- log-loss contribution: `-log(p[label])` (clippato a `eps`);
- Brier contribution: `Σ_k (p_k − y_k)²`;
- accuracy: `1` se `argmax(p) == label`.

Le partite con squadre non presenti nel modello (`KeyError` in `match_outcome_90`)
vengono SALTATE e contate in `n_matches_skipped`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.config import HostAdvantage2026
from src.model.fit import DixonColesModel
from src.model.outcomes import match_outcome_90


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metriche aggregate del modello sul test set."""

    n_matches_evaluated: int
    n_matches_skipped: int        # partite con team non in `model.teams`
    log_loss: float                # multinomial log-loss
    brier_score: float             # multi-class Brier
    accuracy: float                # % argmax(p) == label

    # Frequenze osservate nel test (sanity check)
    p_home_observed: float
    p_draw_observed: float
    p_away_observed: float


def _label_index(home_score: int, away_score: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def evaluate_outcomes_90(
    model: DixonColesModel,
    test_df: pd.DataFrame,
    *,
    host_policy: HostAdvantage2026 | None = None,
    eps: float = 1e-15,
) -> EvaluationMetrics:
    """Multinomial log-loss + Brier + accuracy sugli esiti 90'.

    `test_df` deve avere le colonne `home_team`, `away_team`, `home_score`,
    `away_score`, `neutral`. Le partite con team non presenti nel modello sono
    saltate e contate.
    """
    required = {"home_team", "away_team", "home_score", "away_score", "neutral"}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"test_df mancante delle colonne: {sorted(missing)}")

    teams_in_model = set(model.teams)
    n_eval = 0
    n_skipped = 0
    log_loss_sum = 0.0
    brier_sum = 0.0
    correct = 0
    label_counts = [0, 0, 0]

    for row in test_df.itertuples(index=False):
        if row.home_team not in teams_in_model or row.away_team not in teams_in_model:
            n_skipped += 1
            continue

        out = match_outcome_90(
            model, row.home_team, row.away_team,
            is_neutral=bool(row.neutral),
            host_policy=host_policy,
        )
        probs = (out.p_home_win, out.p_draw, out.p_away_win)
        label = _label_index(int(row.home_score), int(row.away_score))

        p_true = max(probs[label], eps)
        log_loss_sum += -math.log(p_true)
        brier_sum += sum((pk - (1.0 if k == label else 0.0)) ** 2 for k, pk in enumerate(probs))

        predicted = max(range(3), key=lambda k: probs[k])
        if predicted == label:
            correct += 1
        label_counts[label] += 1
        n_eval += 1

    if n_eval == 0:
        raise ValueError(
            "Nessun match valutabile (tutte le squadre del test fuori dal modello)"
        )

    return EvaluationMetrics(
        n_matches_evaluated=n_eval,
        n_matches_skipped=n_skipped,
        log_loss=log_loss_sum / n_eval,
        brier_score=brier_sum / n_eval,
        accuracy=correct / n_eval,
        p_home_observed=label_counts[0] / n_eval,
        p_draw_observed=label_counts[1] / n_eval,
        p_away_observed=label_counts[2] / n_eval,
    )


def log_loss_constant_baseline(
    test_df: pd.DataFrame,
    p_home: float,
    p_draw: float,
    p_away: float,
    *,
    eps: float = 1e-15,
) -> float:
    """Log-loss di un baseline costante (es. medie storiche o uniforme `1/3`).

    Utile per confrontare il modello con un "informed baseline" (DoD M3).
    """
    if not math.isclose(p_home + p_draw + p_away, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"p_home + p_draw + p_away deve sommare a 1 "
            f"(ricevuti {p_home}, {p_draw}, {p_away})"
        )
    n = 0
    ll = 0.0
    for row in test_df.itertuples(index=False):
        label = _label_index(int(row.home_score), int(row.away_score))
        p_true = max((p_home, p_draw, p_away)[label], eps)
        ll += -math.log(p_true)
        n += 1
    if n == 0:
        raise ValueError("test_df vuoto")
    return ll / n


def log_loss_uniform_baseline() -> float:
    """Log-loss del modello a massima entropia (P = 1/3 ovunque): `log(3) ≈ 1.0986`."""
    return math.log(3.0)


__all__ = [
    "EvaluationMetrics",
    "evaluate_outcomes_90",
    "log_loss_constant_baseline",
    "log_loss_uniform_baseline",
]
