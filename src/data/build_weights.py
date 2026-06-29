"""Costruzione del dataset pesato per il fit Dixon-Coles (Issue #3).

Input:  `data/processed/matches_clean.parquet` (Issue #2)
Output: `data/processed/matches_weighted.parquet` + `data/processed/weights_metadata.json`

Pipeline:
1. Carica `matches_clean.parquet`.
2. Droppa le partite di tornei multi-sport / U-23 (`TOURNAMENT_DROP_LIST`).
3. Calcola `weight = w_time * w_comp` con `reference_date` = max(date) di default
   (override via `--reference-date YYYY-MM-DD`).
4. Salva il parquet con tutte le colonne originali + `weight` (float64).
5. Logga e persiste in JSON i tornei non mappati per revisione.

Esecuzione:
    python -m src.data.build_weights
    python -m src.data.build_weights --reference-date 2022-12-31
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import AppConfig, load_config
from src.data.competition_policies import TOURNAMENT_DROP_LIST
from src.data.weights import compute_weights_from_config

logger = logging.getLogger(__name__)

INPUT_PARQUET = "matches_clean.parquet"
OUTPUT_PARQUET = "matches_weighted.parquet"
METADATA_FILENAME = "weights_metadata.json"


def drop_multi_sport(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove le righe i cui tornei sono in `TOURNAMENT_DROP_LIST`."""
    mask = df["tournament"].isin(TOURNAMENT_DROP_LIST)
    n_dropped = int(mask.sum())
    n_tournaments_present = df.loc[mask, "tournament"].nunique() if n_dropped else 0
    logger.info(
        "Drop tornei multi-sport / U-23 (%d in lista, %d presenti nel dataset): %d partite droppate",
        len(TOURNAMENT_DROP_LIST),
        n_tournaments_present,
        n_dropped,
    )
    return df.loc[~mask].copy()


def build_weighted_dataset(
    df_clean: pd.DataFrame,
    reference_date: pd.Timestamp,
    config: AppConfig,
) -> tuple[pd.DataFrame, Counter[str]]:
    """Ritorna `(df_with_weight, counter_unmapped_tournaments)`."""
    df = drop_multi_sport(df_clean)
    unmapped_counter: Counter[str] = Counter()
    weights = compute_weights_from_config(df, reference_date, config, unmapped_counter)
    df = df.assign(weight=weights)
    return df, unmapped_counter


def _summarize_weights(df: pd.DataFrame) -> dict:
    w = df["weight"]
    return {
        "n_rows": int(len(df)),
        "weight_min": float(w.min()),
        "weight_max": float(w.max()),
        "weight_mean": float(w.mean()),
        "weight_median": float(w.median()),
        "weight_sum": float(w.sum()),
    }


def write_metadata(
    output_path: Path,
    reference_date: pd.Timestamp,
    half_life_years: float,
    unmapped_counter: Counter[str],
    summary: dict,
) -> Path:
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_date": str(reference_date.date()),
        "half_life_years": half_life_years,
        "summary": summary,
        "unmapped_tournaments": [
            {"tournament": t, "n_matches": n}
            for t, n in unmapped_counter.most_common()
        ],
        "n_unmapped_distinct": len(unmapped_counter),
        "n_unmapped_total_matches": int(sum(unmapped_counter.values())),
    }
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--reference-date",
        type=str,
        default=None,
        help="Data di riferimento (YYYY-MM-DD) per il decadimento temporale. "
        "Default: max(date) del dataset (= 'oggi' per il modello).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path al parquet processato. Default: data/processed/matches_clean.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path al parquet pesato. Default: data/processed/matches_weighted.parquet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    config = load_config()
    repo_root = Path(__file__).resolve().parents[2]
    input_path = (args.input or (repo_root / config.paths.data_processed / INPUT_PARQUET)).resolve()
    output_path = (args.output or (repo_root / config.paths.data_processed / OUTPUT_PARQUET)).resolve()
    metadata_path = output_path.parent / METADATA_FILENAME

    logger.info("Carico %s", input_path)
    df_clean = pd.read_parquet(input_path)
    logger.info("Input: %d righe", len(df_clean))

    if args.reference_date:
        reference_date = pd.Timestamp(args.reference_date)
        logger.info("reference_date dall'utente: %s", reference_date.date())
    else:
        reference_date = pd.Timestamp(df_clean["date"].max())
        logger.info("reference_date = max(date) del dataset: %s", reference_date.date())

    df_w, unmapped = build_weighted_dataset(df_clean, reference_date, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_w.to_parquet(output_path, index=False, compression="snappy")
    logger.info("Salvato %s (%d righe con colonna `weight`)", output_path, len(df_w))

    summary = _summarize_weights(df_w)
    logger.info(
        "Pesi: min=%.4f, mean=%.4f, median=%.4f, max=%.4f, sum=%.2f",
        summary["weight_min"],
        summary["weight_mean"],
        summary["weight_median"],
        summary["weight_max"],
        summary["weight_sum"],
    )

    if unmapped:
        logger.warning(
            "Tornei NON mappati (default_unmapped=%.2f applicato): %d distinti, %d partite totali",
            config.competition_weights.default_unmapped,
            len(unmapped),
            sum(unmapped.values()),
        )
        for name, n in unmapped.most_common(20):
            logger.warning("  %s -> %d partite", name, n)
    else:
        logger.info("Nessun torneo non mappato.")

    write_metadata(metadata_path, reference_date, config.time_decay.half_life_years, unmapped, summary)
    logger.info("Salvato %s", metadata_path)


if __name__ == "__main__":
    main()
