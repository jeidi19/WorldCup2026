"""Calcolo dei pesi per la likelihood Dixon-Coles (Issue #3).

Ogni partita riceve un peso `w = w_time · w_comp`:

- **Peso temporale**: `w_time = exp(-ξ · giorni_fa)` con `ξ = ln(2) / emivita_giorni`.
  L'emivita è in anni (default 2, grid tunabile). `giorni_fa = reference_date - date_partita`.
  Una partita con `date == reference_date` pesa 1; una a una emivita di distanza pesa 0.5.

- **Peso competizione**: moltiplicatore in `[0.4, 1.0]` dal config, scelto in base al bucket
  del torneo (`competition_policies.classify_tournament`). Tornei non riconosciuti ricevono
  `default_unmapped` (0.6) e vengono raccolti in un counter per revisione manuale.

`reference_date` è SEMPRE parametrica: serve per la validazione temporale, dove "oggi" è una
data nel passato. Niente `datetime.now()` nascosto nelle funzioni.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Mapping

import numpy as np
import pandas as pd

from src.data.competition_policies import (
    ALL_BUCKETS,
    classify_tournament,
)

DAYS_PER_YEAR = 365.25


def half_life_to_xi_per_day(half_life_years: float) -> float:
    """Converte emivita (anni) nel tasso di decadimento giornaliero `ξ = ln(2) / emivita_giorni`."""
    if half_life_years <= 0:
        raise ValueError(f"half_life_years deve essere > 0, ricevuto {half_life_years}")
    return math.log(2.0) / (half_life_years * DAYS_PER_YEAR)


def compute_time_weights(
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    reference_date: pd.Timestamp,
    half_life_years: float,
) -> np.ndarray:
    """Peso temporale per ogni partita.

    Una partita con `date == reference_date` pesa 1.0; una a `half_life_years` di
    distanza pesa 0.5. Partite future rispetto a `reference_date` ricevono peso > 1
    (matematicamente coerente; sono comunque filtrate dal training set se in futuro).
    """
    xi = half_life_to_xi_per_day(half_life_years)
    dates_dt = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    ref_dt = pd.Timestamp(reference_date)
    days_ago = (ref_dt - dates_dt).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(-xi * days_ago)


def _bucket_weights_from_config(config_competition_weights) -> dict[str, float]:
    """Estrae il dict {bucket_id: peso} dal pydantic CompetitionWeights."""
    return {
        "amichevole": config_competition_weights.amichevole,
        "qualificazioni": config_competition_weights.qualificazioni,
        "nations_league": config_competition_weights.nations_league,
        "sub_continentali": config_competition_weights.sub_continentali,
        "finali_continentali": config_competition_weights.finali_continentali,
        "mondiali": config_competition_weights.mondiali,
    }


def compute_competition_weights(
    tournaments: pd.Series | np.ndarray,
    bucket_weights: Mapping[str, float],
    default_unmapped_weight: float,
    unmapped_counter: Counter[str] | None = None,
) -> np.ndarray:
    """Peso competizione per ogni partita.

    Per ogni nome di torneo applica `classify_tournament`. Se il bucket è noto
    legge il moltiplicatore da `bucket_weights`; altrimenti usa `default_unmapped_weight`
    e (se fornito) incrementa il contatore dei tornei non mappati.
    """
    missing = ALL_BUCKETS - set(bucket_weights.keys())
    if missing:
        raise ValueError(f"bucket_weights non copre tutti i bucket: mancano {sorted(missing)}")

    tournaments_arr = np.asarray(pd.Series(tournaments).reset_index(drop=True))
    weights = np.empty(len(tournaments_arr), dtype=np.float64)
    for i, name in enumerate(tournaments_arr):
        bucket = classify_tournament(str(name))
        if bucket is None:
            weights[i] = default_unmapped_weight
            if unmapped_counter is not None:
                unmapped_counter[str(name)] += 1
        else:
            weights[i] = bucket_weights[bucket]
    return weights


def compute_weights(
    df: pd.DataFrame,
    reference_date: pd.Timestamp,
    *,
    half_life_years: float,
    bucket_weights: Mapping[str, float],
    default_unmapped_weight: float,
    unmapped_counter: Counter[str] | None = None,
) -> np.ndarray:
    """Peso finale `w = w_time · w_comp` per ogni riga del dataframe.

    Il dataframe deve contenere le colonne `date` (datetime) e `tournament` (str).
    `reference_date` è iniettabile (per la validazione temporale).
    """
    required = {"date", "tournament"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne richieste mancanti in df: {sorted(missing)}")

    w_time = compute_time_weights(df["date"], reference_date, half_life_years)
    w_comp = compute_competition_weights(
        df["tournament"], bucket_weights, default_unmapped_weight, unmapped_counter
    )
    return w_time * w_comp


def compute_weights_from_config(
    df: pd.DataFrame,
    reference_date: pd.Timestamp,
    config,
    unmapped_counter: Counter[str] | None = None,
) -> np.ndarray:
    """Wrapper user-facing: estrae i parametri di pesatura da `AppConfig`."""
    return compute_weights(
        df,
        reference_date,
        half_life_years=config.time_decay.half_life_years,
        bucket_weights=_bucket_weights_from_config(config.competition_weights),
        default_unmapped_weight=config.competition_weights.default_unmapped,
        unmapped_counter=unmapped_counter,
    )


__all__ = [
    "DAYS_PER_YEAR",
    "half_life_to_xi_per_day",
    "compute_time_weights",
    "compute_competition_weights",
    "compute_weights",
    "compute_weights_from_config",
]
