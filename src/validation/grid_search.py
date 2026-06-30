"""Grid search di `ξ` (emivita) per la validazione temporale (Issue #9).

Per ogni `half_life_years` nel grid:
1. Split temporale (train ≤ cutoff, test > cutoff) — anti-leakage automatico.
2. Drop tornei multi-sport (coerente con la pipeline di build_weights).
3. Calcolo pesi sul train con `reference_date=cutoff` e `half_life=hl`.
4. Fit Dixon-Coles via L-BFGS-B (Issue #5).
5. Evaluate sul test: log-loss multinomiale + Brier + accuracy (Issue #9 eval).

Il ξ ottimo è quello che minimizza la log-loss out-of-sample.

DoD #2: `reference_date` dei pesi == `cutoff_date` dello split (passato esplicitamente).
DoD #3: la funzione ritorna il ξ ottimo + log-loss per ogni valore del grid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.config import AppConfig
from src.data.build_weights import drop_multi_sport
from src.data.weights import compute_weights_from_config
from src.model.fit import DixonColesModel, fit_model
from src.model.indexing import TeamIndexer, prepare_match_data
from src.validation.evaluation import EvaluationMetrics, evaluate_outcomes_90
from src.validation.temporal_split import TemporalSplit, temporal_split

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridSearchResult:
    """Risultato della grid search di `ξ` su un singolo cutoff."""

    cutoff_date: str
    n_train: int
    n_test_evaluated: int
    half_life_results: pd.DataFrame   # 1 riga per half_life, colonne: half_life_years, log_loss, brier_score, accuracy, ...

    @property
    def best_half_life(self) -> float:
        idx = self.half_life_results["log_loss"].idxmin()
        return float(self.half_life_results.loc[idx, "half_life_years"])

    @property
    def best_log_loss(self) -> float:
        return float(self.half_life_results["log_loss"].min())


def _override_half_life(config: AppConfig, half_life_years: float) -> AppConfig:
    """Crea una copia di `config` con `time_decay.half_life_years` sostituito."""
    new_time_decay = config.time_decay.model_copy(
        update={"half_life_years": half_life_years}
    )
    return config.model_copy(update={"time_decay": new_time_decay})


def fit_at_cutoff(
    train_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    config: AppConfig,
    *,
    half_life_years: float,
    max_iter: int = 500,
    max_fun: int = 500_000,
) -> tuple[DixonColesModel, TeamIndexer]:
    """Fit del modello con `reference_date=cutoff` e una specifica emivita.

    Applica il drop multi-sport prima del fit (coerente con la pipeline).
    Ritorna `(model, indexer)`.
    """
    train_filtered = drop_multi_sport(train_df)
    cfg_override = _override_half_life(config, half_life_years)
    weights = compute_weights_from_config(
        train_filtered, reference_date=cutoff_date, config=cfg_override
    )
    indexer = TeamIndexer.from_match_dataframe(train_filtered)
    data = prepare_match_data(train_filtered, indexer)
    model = fit_model(
        data, weights, cfg_override, indexer,
        reference_date=cutoff_date,
        max_iter=max_iter,
        max_fun=max_fun,
    )
    return model, indexer


def grid_search_half_life(
    df_clean: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
    config: AppConfig,
    *,
    half_life_grid: Sequence[float] | None = None,
    max_iter: int = 500,
    max_fun: int = 500_000,
) -> GridSearchResult:
    """Grid search di emivita su un cutoff.

    `half_life_grid` default: `config.time_decay.half_life_years_grid`.
    """
    cutoff = pd.Timestamp(cutoff_date)
    if half_life_grid is None:
        half_life_grid = config.time_decay.half_life_years_grid

    split = temporal_split(df_clean, cutoff)
    logger.info(
        "Split temporale: cutoff=%s, train=%d, test=%d",
        cutoff.date(), split.n_train, split.n_test,
    )

    # Pre-drop multi-sport dal test (coerenza con il train + esclude squadre B/U-23)
    test_eval_df = drop_multi_sport(split.test_df)
    logger.info("Test set post-drop multi-sport: %d match", len(test_eval_df))

    rows: list[dict] = []
    for hl in half_life_grid:
        logger.info("Fit con half_life=%.2f anni...", hl)
        model, _ = fit_at_cutoff(
            split.train_df, cutoff, config,
            half_life_years=hl, max_iter=max_iter, max_fun=max_fun,
        )
        metrics = evaluate_outcomes_90(model, test_eval_df)
        rows.append({
            "half_life_years": float(hl),
            "log_loss": metrics.log_loss,
            "brier_score": metrics.brier_score,
            "accuracy": metrics.accuracy,
            "n_eval": metrics.n_matches_evaluated,
            "n_skipped": metrics.n_matches_skipped,
            "gamma": model.gamma,
            "rho": model.rho,
            "converged": model.converged,
        })
        logger.info(
            "  hl=%.2f -> log_loss=%.4f, brier=%.4f, accuracy=%.3f",
            hl, metrics.log_loss, metrics.brier_score, metrics.accuracy,
        )

    df_results = pd.DataFrame(rows).sort_values("half_life_years").reset_index(drop=True)
    best_idx = df_results["log_loss"].idxmin()
    best_hl = float(df_results.loc[best_idx, "half_life_years"])
    best_ll = float(df_results.loc[best_idx, "log_loss"])
    logger.info("Best half_life = %.2f anni (log_loss=%.4f)", best_hl, best_ll)

    return GridSearchResult(
        cutoff_date=str(cutoff.date()),
        n_train=split.n_train,
        n_test_evaluated=int(df_results["n_eval"].iloc[0]),  # uguale per ogni hl
        half_life_results=df_results,
    )


__all__ = [
    "GridSearchResult",
    "fit_at_cutoff",
    "grid_search_half_life",
]
