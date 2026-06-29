"""Pulizia e normalizzazione del dataset grezzo (Issue #2).

Input:  `data/raw/results.csv` (prodotto da Issue #1)
Output: `data/processed/matches_clean.parquet`

Passi della pipeline:
1. Carica il CSV e tipizza `date` come datetime.
2. Droppa righe con NaN su `home_score`/`away_score` (tipicamente partite future del
   Mondiale 2026 ancora da giocare). I punteggi diventano `int64`.
3. Applica `normalize_team_name` su `home_team`/`away_team` (idempotente: mapping vuoto).
4. Droppa righe in cui home/away è nella `NON_FIFA_DENY_LIST`.
5. Garantisce `neutral` come booleano puro.
6. Calcola `match_id` come hash deterministico SHA-1 troncato di `(date_iso|home|away)`.
   Eventuali collisioni (raro: doppi incontri stesso giorno) ricevono un suffisso.
7. Ordina per data crescente (DoD: date monotone ordinabili).
8. Salva come parquet compresso snappy.

Esecuzione:
    python -m src.data.clean
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.team_policies import (
    NON_FIFA_DENY_LIST,
    is_extinct,
    normalize_team_name,
)

logger = logging.getLogger(__name__)

INPUT_CSV = "results.csv"
OUTPUT_PARQUET = "matches_clean.parquet"

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV grezzo: {missing}")
    return df


def drop_na_scores(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype("int64")
    df["away_score"] = df["away_score"].astype("int64")
    logger.info(
        "Drop NaN scores: %d -> %d righe (-%d)", n_before, len(df), n_before - len(df)
    )
    return df


def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    return df


def apply_deny_list(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["home_team"].isin(NON_FIFA_DENY_LIST)
        | df["away_team"].isin(NON_FIFA_DENY_LIST)
    )
    n_dropped = int(mask.sum())
    logger.info(
        "Deny-list non-FIFA (%d entità): %d partite droppate",
        len(NON_FIFA_DENY_LIST),
        n_dropped,
    )
    return df.loc[~mask].copy()


def coerce_neutral(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["neutral"] = df["neutral"].astype(bool)
    return df


def _match_id(date_iso: str, home: str, away: str) -> str:
    h = hashlib.sha1(f"{date_iso}|{home}|{away}".encode("utf-8"))
    return h.hexdigest()[:16]


def add_match_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_iso = df["date"].dt.strftime("%Y-%m-%d")
    df["match_id"] = [
        _match_id(d, h, a)
        for d, h, a in zip(date_iso, df["home_team"], df["away_team"])
    ]
    dup_mask = df.duplicated(subset=["match_id"], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        logger.warning(
            "%d righe con match_id duplicato (stesse squadre stesso giorno); aggiungo suffisso.",
            n_dup,
        )
        df.loc[dup_mask, "match_id"] = (
            df.loc[dup_mask, "match_id"].astype(str)
            + "-"
            + df.loc[dup_mask].groupby("match_id").cumcount().astype(str)
        )
    return df


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = drop_na_scores(df_raw)
    df = normalize_names(df)
    df = apply_deny_list(df)
    df = coerce_neutral(df)
    df = add_match_id(df)
    df = sort_by_date(df)
    n_teams = len(set(df["home_team"]) | set(df["away_team"]))
    logger.info("Dataset processato: %d righe, %d squadre uniche", len(df), n_teams)
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config()
    repo_root = Path(__file__).resolve().parents[2]
    raw_path = (repo_root / config.paths.data_raw / INPUT_CSV).resolve()
    out_path = (repo_root / config.paths.data_processed / OUTPUT_PARQUET).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Carico %s", raw_path)
    df_raw = load_raw(raw_path)
    logger.info("Raw: %d righe, %d colonne", len(df_raw), len(df_raw.columns))

    df = clean(df_raw)
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.info("Salvato %s", out_path)

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    extinct_present = sorted(t for t in teams if is_extinct(t))
    logger.info(
        "Entità estinte presenti nel processato (mantenute come segnale): %s",
        extinct_present,
    )
    logger.info("Range temporale: %s -> %s", df["date"].min().date(), df["date"].max().date())


if __name__ == "__main__":
    main()
