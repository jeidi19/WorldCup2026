"""Test di integrazione sul dataset reale (Issue #3).

Si attiva solo se `data/processed/matches_clean.parquet` esiste (lo skip è esplicito).
Verifica che la pesatura sul dataset reale soddisfi proprietà di coerenza:

- Coverage: meno del 2% delle partite finisce in `default_unmapped`.
- Range pesi: `0 < weight <= 1` per ogni partita (post-drop multi-sport).
- Monotonia per torneo: a parità di torneo, partite più recenti pesano di più.
- Pesi di partite del Mondiale 2026 in date recenti devono essere vicini a 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.build_weights import build_weighted_dataset
from src.data.weights import compute_weights_from_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PARQUET = REPO_ROOT / "data" / "processed" / "matches_clean.parquet"

pytestmark = pytest.mark.skipif(
    not CLEAN_PARQUET.exists(),
    reason="matches_clean.parquet non disponibile (lanciare prima `python -m src.data.clean`)",
)


@pytest.fixture(scope="module")
def df_clean() -> pd.DataFrame:
    return pd.read_parquet(CLEAN_PARQUET)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_real_coverage_under_two_percent(df_clean, cfg):
    """Sul dataset reale (post-drop multi-sport), i tornei non mappati devono essere < 2% delle partite."""
    ref = pd.Timestamp(df_clean["date"].max())
    df_w, unmapped = build_weighted_dataset(df_clean, ref, cfg)
    n_unmapped = sum(unmapped.values())
    n_post_drop = len(df_w)
    ratio = n_unmapped / n_post_drop
    assert ratio < 0.02, (
        f"Tornei non mappati: {n_unmapped}/{n_post_drop} = {ratio:.2%} > 2% — "
        f"aggiornare TOURNAMENT_BUCKETS o TOURNAMENT_DROP_LIST. "
        f"Top 5 non mappati: {unmapped.most_common(5)}"
    )


def test_real_weights_in_valid_range(df_clean, cfg):
    ref = pd.Timestamp(df_clean["date"].max())
    df_w, _ = build_weighted_dataset(df_clean, ref, cfg)
    assert (df_w["weight"] > 0).all()
    assert (df_w["weight"] <= 1.0 + 1e-12).all()


def test_real_recent_world_cup_match_weights_close_to_one(df_clean, cfg):
    """Una partita di FIFA World Cup nella reference_date stessa pesa ≈ 1.0 (w_time=1, w_comp=1)."""
    ref = pd.Timestamp(df_clean["date"].max())
    df_w, _ = build_weighted_dataset(df_clean, ref, cfg)
    wc_today = df_w[(df_w["tournament"] == "FIFA World Cup") & (df_w["date"] == ref)]
    if len(wc_today) == 0:
        pytest.skip("Nessuna partita Mondiale alla reference_date esatta")
    assert (wc_today["weight"] > 0.99).all()


def test_real_monotonic_within_tournament(df_clean, cfg):
    """Friendly: per stesso torneo (w_comp costante), w_time deve essere monotono in date."""
    ref = pd.Timestamp(df_clean["date"].max())
    df_w, _ = build_weighted_dataset(df_clean, ref, cfg)
    friendly = df_w[df_w["tournament"] == "Friendly"].sort_values("date")
    # date crescente => weight crescente (w_comp costante per Friendly)
    assert (np.diff(friendly["weight"].to_numpy()) >= -1e-12).all()


def test_real_reference_date_in_past_works(df_clean, cfg):
    """Validazione temporale: ref nel passato → partite "future" hanno peso > 1 ma il computo non esplode."""
    ref = pd.Timestamp("2010-01-01")
    weights = compute_weights_from_config(df_clean, ref, cfg)
    assert np.isfinite(weights).all()
    assert weights.shape == (len(df_clean),)
