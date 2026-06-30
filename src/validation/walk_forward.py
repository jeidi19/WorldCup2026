"""Walk-forward validation: più cutoff progressivi (Issue #9 — opzionale).

Per ogni cutoff in `cutoffs`:
1. Train ≤ cutoff, test > cutoff
2. Fit con `half_life_years` fissa
3. Eval sul test

Restituisce un `DataFrame` con una riga per cutoff (log-loss, Brier, accuracy, n).

Più robusto allo split-singolo ma molto più lento: ogni cutoff è un fit di ~10 min
sul dataset reale. Non eseguito automaticamente — solo a richiesta.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from src.config import AppConfig
from src.data.build_weights import drop_multi_sport
from src.validation.evaluation import evaluate_outcomes_90
from src.validation.grid_search import fit_at_cutoff
from src.validation.temporal_split import temporal_split

logger = logging.getLogger(__name__)


def walk_forward(
    df_clean: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp | str],
    config: AppConfig,
    *,
    half_life_years: float,
    max_iter: int = 500,
    max_fun: int = 500_000,
) -> pd.DataFrame:
    """Esegue uno walk-forward: fit + eval per ogni cutoff.

    Ritorna `DataFrame` con colonne: cutoff_date, n_train, n_test, log_loss, brier_score,
    accuracy.
    """
    rows: list[dict] = []
    for raw_cutoff in cutoffs:
        cutoff = pd.Timestamp(raw_cutoff)
        logger.info("Walk-forward: cutoff=%s", cutoff.date())
        split = temporal_split(df_clean, cutoff)
        model, _ = fit_at_cutoff(
            split.train_df, cutoff, config,
            half_life_years=half_life_years,
            max_iter=max_iter, max_fun=max_fun,
        )
        test_eval = drop_multi_sport(split.test_df)
        metrics = evaluate_outcomes_90(model, test_eval)
        rows.append({
            "cutoff_date": str(cutoff.date()),
            "n_train": split.n_train,
            "n_test": metrics.n_matches_evaluated,
            "log_loss": metrics.log_loss,
            "brier_score": metrics.brier_score,
            "accuracy": metrics.accuracy,
            "gamma": model.gamma,
            "rho": model.rho,
        })
    return pd.DataFrame(rows)


__all__ = ["walk_forward"]
