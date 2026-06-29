"""Smoke test del modulo di download (Issue #1).

Non esegue il vero download (richiederebbe credenziali Kaggle e rete). Verifica solo:
- che il modulo sia importabile e che lo slug del dataset sia quello atteso;
- che `load_dotenv` legga correttamente le variabili da un .env minimale;
- che `summarize_csv` produca il sommario atteso su un CSV sintetico.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data import download as dl


def test_module_constants():
    assert dl.KAGGLE_DATASET_SLUG == "martj42/international-football-results-from-1872-to-2017"
    assert dl.CSV_FILENAME == "results.csv"
    assert dl.METADATA_FILENAME == "metadata.json"


def test_load_dotenv_reads_export_and_quoted_values(tmp_path: Path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# commento\n"
        "export FOO=bar\n"
        "BAZ=\"quoted value\"\n"
        "\n"
        "EMPTY=\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)
    dl.load_dotenv(env_path)
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "quoted value"


def test_load_dotenv_does_not_overwrite_existing(monkeypatch, tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("FOO=from_env_file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "preexisting")
    dl.load_dotenv(env_path)
    assert os.environ["FOO"] == "preexisting"


def test_summarize_csv_on_synthetic_data(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["1872-11-30", "2024-03-26", "1990-07-08"]),
            "home_team": ["Scotland", "Italy", "Germany"],
            "away_team": ["England", "Ecuador", "Argentina"],
            "home_score": [0, 2, 1],
            "away_score": [0, 0, 0],
            "tournament": ["Friendly", "Friendly", "FIFA World Cup"],
            "neutral": [False, False, True],
        }
    )
    csv_path = tmp_path / "results.csv"
    df.to_csv(csv_path, index=False)
    summary = dl.summarize_csv(csv_path)
    assert summary["rows"] == 3
    assert summary["date_min"] == "1872-11-30"
    assert summary["date_max"] == "2024-03-26"
    assert "neutral" in summary["columns"]


def test_write_metadata_includes_hash_and_source(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("date,home_team,away_team,home_score,away_score,tournament,neutral\n",
                        encoding="utf-8")
    summary = {"rows": 0, "columns": ["date"], "date_min": "1872-11-30", "date_max": "1872-11-30"}
    metadata_path = dl.write_metadata(csv_path, summary)
    import json
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["source"].startswith("kaggle:")
    assert "downloaded_at" in meta
    assert "sha256" in meta and len(meta["sha256"]) == 64
    assert meta["rows"] == 0
