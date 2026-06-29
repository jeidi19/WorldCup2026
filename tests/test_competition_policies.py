"""Test delle policy di classificazione dei tornei (Issue #3)."""
from __future__ import annotations

from src.data.competition_policies import (
    ALL_BUCKETS,
    BUCKET_AMICHEVOLE,
    BUCKET_FINALI_CONTINENTALI,
    BUCKET_MONDIALI,
    BUCKET_NATIONS_LEAGUE,
    BUCKET_QUALIFICAZIONI,
    BUCKET_SUB_CONTINENTALI,
    TOURNAMENT_BUCKETS,
    TOURNAMENT_DROP_LIST,
    classify_tournament,
    should_drop_tournament,
)


# ---------------------------------------------------------------------------
# Coerenza strutturale degli insiemi
# ---------------------------------------------------------------------------

def test_all_buckets_are_six():
    assert ALL_BUCKETS == {
        BUCKET_AMICHEVOLE,
        BUCKET_QUALIFICAZIONI,
        BUCKET_NATIONS_LEAGUE,
        BUCKET_SUB_CONTINENTALI,
        BUCKET_FINALI_CONTINENTALI,
        BUCKET_MONDIALI,
    }


def test_explicit_buckets_values_are_valid():
    """Ogni torneo nel mapping esplicito deve puntare a un bucket noto."""
    for tournament, bucket in TOURNAMENT_BUCKETS.items():
        assert bucket in ALL_BUCKETS, (
            f"Torneo {tournament!r} mappato a bucket sconosciuto: {bucket!r}"
        )


def test_drop_list_disjoint_from_buckets():
    """Un torneo non può essere insieme droppato e classificato (contraddittorio)."""
    in_both = set(TOURNAMENT_DROP_LIST) & set(TOURNAMENT_BUCKETS.keys())
    assert not in_both, f"Tornei in entrambi DROP_LIST e BUCKETS: {in_both}"


def test_known_top_tournaments_are_explicitly_mapped():
    """I top tornei (per partite cumulate) devono essere esplicitamente mappati."""
    must_have_explicit = {
        "Friendly",
        "FIFA World Cup",
        "Copa América",
        "African Cup of Nations",
        "AFC Asian Cup",
        "UEFA Euro",
        "Gold Cup",
        "Oceania Nations Cup",
        "UEFA Nations League",
        "CONCACAF Nations League",
        "CECAFA Cup",
        "COSAFA Cup",
        "Gulf Cup",
        "AFF Championship",
        "British Home Championship",
        "Confederations Cup",
    }
    assert must_have_explicit.issubset(TOURNAMENT_BUCKETS.keys())


# ---------------------------------------------------------------------------
# classify_tournament
# ---------------------------------------------------------------------------

def test_classify_mondiali():
    assert classify_tournament("FIFA World Cup") == BUCKET_MONDIALI
    assert classify_tournament("Confederations Cup") == BUCKET_MONDIALI
    assert classify_tournament("Finalissima") == BUCKET_MONDIALI


def test_classify_finali_continentali():
    assert classify_tournament("UEFA Euro") == BUCKET_FINALI_CONTINENTALI
    assert classify_tournament("Copa América") == BUCKET_FINALI_CONTINENTALI
    assert classify_tournament("African Cup of Nations") == BUCKET_FINALI_CONTINENTALI
    assert classify_tournament("AFC Asian Cup") == BUCKET_FINALI_CONTINENTALI
    assert classify_tournament("Gold Cup") == BUCKET_FINALI_CONTINENTALI


def test_classify_nations_league_explicit():
    assert classify_tournament("UEFA Nations League") == BUCKET_NATIONS_LEAGUE
    assert classify_tournament("CONCACAF Nations League") == BUCKET_NATIONS_LEAGUE


def test_classify_nations_league_pattern_fallback():
    """Un'eventuale 'X Nations League' sconosciuto cade comunque in nations_league."""
    assert classify_tournament("AFC Nations League") == BUCKET_NATIONS_LEAGUE


def test_classify_qualification_pattern_fallback():
    """Pattern `... qualification` deve dare BUCKET_QUALIFICAZIONI senza mapping esplicito."""
    assert classify_tournament("FIFA World Cup qualification") == BUCKET_QUALIFICAZIONI
    assert classify_tournament("UEFA Euro qualification") == BUCKET_QUALIFICAZIONI
    assert classify_tournament("African Cup of Nations qualification") == BUCKET_QUALIFICAZIONI
    assert classify_tournament("AFC Asian Cup qualification") == BUCKET_QUALIFICAZIONI
    assert classify_tournament("CFU Caribbean Cup qualification") == BUCKET_QUALIFICAZIONI
    assert classify_tournament("CONCACAF Championship qualification") == BUCKET_QUALIFICAZIONI


def test_classify_qualification_takes_precedence_over_nations_league():
    """Una qualificazione di Nations League deve finire in qualificazioni, non in NL."""
    assert (
        classify_tournament("UEFA Nations League qualification")
        == BUCKET_QUALIFICAZIONI
    )


def test_classify_sub_continentali():
    assert classify_tournament("CECAFA Cup") == BUCKET_SUB_CONTINENTALI
    assert classify_tournament("COSAFA Cup") == BUCKET_SUB_CONTINENTALI
    assert classify_tournament("Gulf Cup") == BUCKET_SUB_CONTINENTALI
    assert classify_tournament("AFF Championship") == BUCKET_SUB_CONTINENTALI
    assert classify_tournament("SAFF Cup") == BUCKET_SUB_CONTINENTALI
    assert classify_tournament("Arab Cup") == BUCKET_SUB_CONTINENTALI


def test_classify_amichevole():
    assert classify_tournament("Friendly") == BUCKET_AMICHEVOLE
    assert classify_tournament("Merdeka Tournament") == BUCKET_AMICHEVOLE
    assert classify_tournament("British Home Championship") == BUCKET_AMICHEVOLE
    assert classify_tournament("King's Cup") == BUCKET_AMICHEVOLE


def test_classify_unknown_returns_none():
    """Tornei sconosciuti tornano None (verranno mappati al default_unmapped)."""
    assert classify_tournament("Some Random Cup 1923") is None
    assert classify_tournament("xxx") is None


# ---------------------------------------------------------------------------
# should_drop_tournament
# ---------------------------------------------------------------------------

def test_should_drop_multi_sport():
    assert should_drop_tournament("Asian Games")
    assert should_drop_tournament("Southeast Asian Games")
    assert should_drop_tournament("Island Games")
    assert should_drop_tournament("Pan American Games")


def test_should_drop_conifa_pattern():
    """Pattern drop: tutti i tornei CONIFA / ConIFA (federazione non-FIFA)."""
    assert should_drop_tournament("CONIFA World Football Cup")
    assert should_drop_tournament("CONIFA European Football Cup")
    assert should_drop_tournament("CONIFA Asia Cup")
    assert should_drop_tournament("CONIFA Africa Football Cup")
    assert should_drop_tournament("CONIFA South America Football Cup")
    assert should_drop_tournament("CONIFA World Football Cup qualification")
    assert should_drop_tournament("ConIFA Challenger Cup")    # diversa capitalizzazione


def test_should_not_drop_normal_tournaments():
    assert not should_drop_tournament("Friendly")
    assert not should_drop_tournament("FIFA World Cup")
    assert not should_drop_tournament("CECAFA Cup")
    assert not should_drop_tournament("Some unknown thing")
