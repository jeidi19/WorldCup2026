"""Download del dataset storico delle partite tra nazionali (Issue #1).

Fonte: Kaggle, slug `martj42/international-football-results-from-1872-to-2017`
(nonostante il nome, il dataset è aggiornato regolarmente fino al presente).

Lo script:
1. Carica le credenziali Kaggle da `.env` (var `KAGGLE_API_TOKEN`, formato `KGAT_...`).
2. Scarica il dataset via `kagglehub` (cache locale di KaggleHub).
3. Copia il `results.csv` in `data/raw/results.csv` (immutabile: mai modificare in-place).
4. Logga conteggio righe e range temporale.
5. Scrive `data/raw/metadata.json` con fonte, data download, hash SHA-256, righe, colonne.

Esecuzione:
    python -m src.data.download
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import load_config

KAGGLE_DATASET_SLUG = "martj42/international-football-results-from-1872-to-2017"
CSV_FILENAME = "results.csv"
METADATA_FILENAME = "metadata.json"

logger = logging.getLogger(__name__)


def load_dotenv(path: Path) -> None:
    """Legge un `.env` minimale e popola `os.environ` (senza sovrascrivere)."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep or not key:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _ensure_kaggle_credentials() -> None:
    """Adatta `KAGGLE_API_TOKEN` (nuovo formato) alle variabili che kagglehub si aspetta."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token and not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = token


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(destination_dir: Path) -> Path:
    """Scarica il dataset e ne copia `results.csv` in `destination_dir`. Ritorna il path del CSV."""
    import kagglehub  # import locale: dipendenza opzionale a livello di modulo

    destination_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Scarico il dataset Kaggle '%s'...", KAGGLE_DATASET_SLUG)
    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    logger.info("Cache KaggleHub: %s", cache_dir)

    source_csv = cache_dir / CSV_FILENAME
    if not source_csv.exists():
        candidates = list(cache_dir.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"Nessun CSV trovato in {cache_dir}. Contenuto: {list(cache_dir.iterdir())}"
            )
        source_csv = candidates[0]
        logger.warning("results.csv non trovato; uso il primo CSV disponibile: %s", source_csv.name)

    target_csv = destination_dir / CSV_FILENAME
    shutil.copy2(source_csv, target_csv)
    logger.info("Copiato in %s", target_csv)
    return target_csv


def summarize_csv(csv_path: Path) -> dict:
    """Logga e ritorna un sommario: righe, colonne, range temporale."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    n_rows = len(df)
    columns = list(df.columns)
    date_min = df["date"].min()
    date_max = df["date"].max()
    logger.info(
        "Dataset: %d righe, colonne=%s, date dal %s al %s",
        n_rows,
        columns,
        date_min.date(),
        date_max.date(),
    )
    return {
        "rows": n_rows,
        "columns": columns,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
    }


def write_metadata(csv_path: Path, summary: dict) -> Path:
    """Scrive `metadata.json` accanto al CSV con fonte, timestamp e hash."""
    metadata = {
        "source": f"kaggle:{KAGGLE_DATASET_SLUG}",
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file": csv_path.name,
        "sha256": _sha256(csv_path),
        **summary,
    }
    metadata_path = csv_path.parent / METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Scritto metadata in %s", metadata_path)
    return metadata_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    _ensure_kaggle_credentials()

    config = load_config()
    raw_dir = (repo_root / config.paths.data_raw).resolve()
    csv_path = download_dataset(raw_dir)
    summary = summarize_csv(csv_path)
    write_metadata(csv_path, summary)
    logger.info("Done.")


if __name__ == "__main__":
    main()
