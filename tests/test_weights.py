"""Test della pesatura temporale + competizione (Issue #3).

Copre le DoD:
- Una partita di "oggi" (date == reference_date) pesa ~1 · w_comp.
- Una partita a 1 emivita pesa ~0.5 · w_comp.
- `reference_date` è iniettabile (no `datetime.now()` interno).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.competition_policies import (
    BUCKET_AMICHEVOLE,
    BUCKET_FINALI_CONTINENTALI,
    BUCKET_MONDIALI,
    BUCKET_QUALIFICAZIONI,
    BUCKET_SUB_CONTINENTALI,
)
from src.data.weights import (
    DAYS_PER_YEAR,
    _bucket_weights_from_config,
    compute_competition_weights,
    compute_time_weights,
    compute_weights,
    compute_weights_from_config,
    half_life_to_xi_per_day,
)


# ---------------------------------------------------------------------------
# Tassi di decadimento
# ---------------------------------------------------------------------------

def test_half_life_to_xi_per_day():
    xi = half_life_to_xi_per_day(2.0)
    assert xi == pytest.approx(math.log(2.0) / (2.0 * DAYS_PER_YEAR), rel=1e-12)


def test_half_life_must_be_positive():
    with pytest.raises(ValueError):
        half_life_to_xi_per_day(0.0)
    with pytest.raises(ValueError):
        half_life_to_xi_per_day(-1.0)


# ---------------------------------------------------------------------------
# Peso temporale: DoD #1
# ---------------------------------------------------------------------------

def test_time_weight_today_is_one():
    ref = pd.Timestamp("2024-01-01")
    w = compute_time_weights(pd.Series([ref]), reference_date=ref, half_life_years=2.0)
    assert w[0] == pytest.approx(1.0, abs=1e-12)


def test_time_weight_one_half_life_is_half():
    ref = pd.Timestamp("2024-01-01")
    half_life = 2.0
    one_half_life_ago = ref - pd.Timedelta(days=half_life * DAYS_PER_YEAR)
    w = compute_time_weights(
        pd.Series([one_half_life_ago]), reference_date=ref, half_life_years=half_life
    )
    assert w[0] == pytest.approx(0.5, abs=1e-12)


def test_time_weight_two_half_lives_is_quarter():
    ref = pd.Timestamp("2024-01-01")
    two_hl_ago = ref - pd.Timedelta(days=2 * 2.0 * DAYS_PER_YEAR)
    w = compute_time_weights(pd.Series([two_hl_ago]), reference_date=ref, half_life_years=2.0)
    assert w[0] == pytest.approx(0.25, abs=1e-12)


def test_time_weight_future_match_above_one():
    """Una partita futura rispetto al ref riceve peso > 1 (sono filtrate altrove dal training)."""
    ref = pd.Timestamp("2020-01-01")
    future = ref + pd.Timedelta(days=365)
    w = compute_time_weights(pd.Series([future]), reference_date=ref, half_life_years=2.0)
    assert w[0] > 1.0


def test_time_weight_monotonic_in_distance():
    """Più una partita è vecchia, minore il peso (monotonia decrescente in days_ago)."""
    ref = pd.Timestamp("2024-01-01")
    dates = pd.Series(pd.date_range(end=ref, periods=10, freq="365D"))
    w = compute_time_weights(dates, reference_date=ref, half_life_years=2.0)
    # dates è crescente → days_ago decrescente → w crescente
    assert (np.diff(w) >= 0).all()


# ---------------------------------------------------------------------------
# Peso competizione
# ---------------------------------------------------------------------------

@pytest.fixture
def bucket_weights() -> dict[str, float]:
    return {
        BUCKET_AMICHEVOLE: 0.4,
        BUCKET_QUALIFICAZIONI: 0.8,
        "nations_league": 0.8,
        BUCKET_SUB_CONTINENTALI: 0.6,
        BUCKET_FINALI_CONTINENTALI: 1.0,
        BUCKET_MONDIALI: 1.0,
    }


def test_competition_weight_known_buckets(bucket_weights):
    tournaments = pd.Series(
        [
            "Friendly",
            "FIFA World Cup",
            "UEFA Euro",
            "FIFA World Cup qualification",
            "UEFA Nations League",
            "CECAFA Cup",
        ]
    )
    w = compute_competition_weights(tournaments, bucket_weights, default_unmapped_weight=0.6)
    assert w.tolist() == [0.4, 1.0, 1.0, 0.8, 0.8, 0.6]


def test_competition_weight_unknown_uses_default_and_logs(bucket_weights):
    tournaments = pd.Series(["Friendly", "Some Unknown Cup", "Some Unknown Cup"])
    counter: Counter[str] = Counter()
    w = compute_competition_weights(
        tournaments, bucket_weights, default_unmapped_weight=0.6, unmapped_counter=counter
    )
    assert w.tolist() == [0.4, 0.6, 0.6]
    assert counter == Counter({"Some Unknown Cup": 2})


def test_competition_weight_rejects_incomplete_bucket_map():
    """Se mancano bucket dalla mappa, deve sollevare un errore esplicito."""
    bad_map = {BUCKET_AMICHEVOLE: 0.4}
    with pytest.raises(ValueError, match="bucket_weights"):
        compute_competition_weights(pd.Series(["Friendly"]), bad_map, default_unmapped_weight=0.6)


# ---------------------------------------------------------------------------
# compute_weights: DoD #1 ("oggi pesa ~1·w_comp"; "1 emivita pesa ~0.5·w_comp")
#                  DoD #2 (reference_date iniettabile)
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_df() -> pd.DataFrame:
    ref = pd.Timestamp("2024-01-01")
    return pd.DataFrame(
        {
            "date": [
                ref,                                            # oggi
                ref - pd.Timedelta(days=2 * DAYS_PER_YEAR),     # 1 emivita fa (con HL=2)
                ref - pd.Timedelta(days=10 * DAYS_PER_YEAR),    # ben più vecchia
            ],
            "tournament": ["Friendly", "Friendly", "FIFA World Cup"],
        }
    )


def test_compute_weights_today_equals_w_comp(mini_df, bucket_weights):
    ref = pd.Timestamp("2024-01-01")
    w = compute_weights(
        mini_df,
        reference_date=ref,
        half_life_years=2.0,
        bucket_weights=bucket_weights,
        default_unmapped_weight=0.6,
    )
    assert w[0] == pytest.approx(0.4, abs=1e-12)  # Friendly oggi: 1.0 * 0.4


def test_compute_weights_one_half_life_equals_half_w_comp(mini_df, bucket_weights):
    ref = pd.Timestamp("2024-01-01")
    w = compute_weights(
        mini_df,
        reference_date=ref,
        half_life_years=2.0,
        bucket_weights=bucket_weights,
        default_unmapped_weight=0.6,
    )
    # Friendly 1 emivita fa: 0.5 * 0.4 = 0.2
    assert w[1] == pytest.approx(0.2, abs=1e-12)


def test_compute_weights_reference_date_is_injectable(mini_df, bucket_weights):
    """Cambiando reference_date cambiano i pesi (no datetime.now() interno)."""
    w_2024 = compute_weights(
        mini_df,
        reference_date=pd.Timestamp("2024-01-01"),
        half_life_years=2.0,
        bucket_weights=bucket_weights,
        default_unmapped_weight=0.6,
    )
    w_2010 = compute_weights(
        mini_df,
        reference_date=pd.Timestamp("2010-01-01"),
        half_life_years=2.0,
        bucket_weights=bucket_weights,
        default_unmapped_weight=0.6,
    )
    # Con ref nel 2010, la partita "oggi" (2024-01-01) è 14 anni nel futuro → peso > 1
    assert w_2010[0] > 1.0
    # E i due vettori devono essere distinti
    assert not np.allclose(w_2024, w_2010)


def test_compute_weights_deterministic(mini_df, bucket_weights):
    """Stesso df + stessa ref → stessi pesi (idempotente)."""
    ref = pd.Timestamp("2024-01-01")
    kwargs = dict(
        half_life_years=2.0,
        bucket_weights=bucket_weights,
        default_unmapped_weight=0.6,
    )
    w1 = compute_weights(mini_df, reference_date=ref, **kwargs)
    w2 = compute_weights(mini_df, reference_date=ref, **kwargs)
    assert np.array_equal(w1, w2)


def test_compute_weights_preserves_shape(mini_df, bucket_weights):
    w = compute_weights(
        mini_df,
        reference_date=pd.Timestamp("2024-01-01"),
        half_life_years=2.0,
        bucket_weights=bucket_weights,
        default_unmapped_weight=0.6,
    )
    assert w.shape == (len(mini_df),)


def test_compute_weights_rejects_missing_columns(bucket_weights):
    df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match="Colonne richieste mancanti"):
        compute_weights(
            df,
            reference_date=pd.Timestamp("2024-01-01"),
            half_life_years=2.0,
            bucket_weights=bucket_weights,
            default_unmapped_weight=0.6,
        )


def test_compute_weights_empty_df(bucket_weights):
    empty = pd.DataFrame({"date": pd.to_datetime([]), "tournament": []})
    w = compute_weights(
        empty,
        reference_date=pd.Timestamp("2024-01-01"),
        half_life_years=2.0,
        bucket_weights=bucket_weights,
        default_unmapped_weight=0.6,
    )
    assert w.shape == (0,)


# ---------------------------------------------------------------------------
# Wrapper from-config: integrazione con AppConfig
# ---------------------------------------------------------------------------

def test_bucket_weights_from_config_has_all_buckets():
    cfg = load_config()
    bw = _bucket_weights_from_config(cfg.competition_weights)
    from src.data.competition_policies import ALL_BUCKETS
    assert set(bw.keys()) == ALL_BUCKETS


def test_compute_weights_from_config_matches_explicit():
    """Il wrapper from-config deve produrre gli stessi pesi della chiamata esplicita."""
    cfg = load_config()
    ref = pd.Timestamp("2024-01-01")
    df = pd.DataFrame(
        {
            "date": [ref, ref - pd.Timedelta(days=365)],
            "tournament": ["Friendly", "FIFA World Cup qualification"],
        }
    )
    w_via_config = compute_weights_from_config(df, reference_date=ref, config=cfg)
    w_explicit = compute_weights(
        df,
        reference_date=ref,
        half_life_years=cfg.time_decay.half_life_years,
        bucket_weights=_bucket_weights_from_config(cfg.competition_weights),
        default_unmapped_weight=cfg.competition_weights.default_unmapped,
    )
    assert np.allclose(w_via_config, w_explicit)
