"""Test di consistenza delle policy sui nomi delle squadre (Issue #2)."""
from __future__ import annotations

from src.data.team_policies import (
    EXTINCT_TEAMS,
    NAME_ALIASES,
    NON_FIFA_DENY_LIST,
    is_extinct,
    is_in_deny_list,
    normalize_team_name,
)


def test_deny_list_and_extinct_are_disjoint():
    """Una squadra non può essere insieme in deny-list e in estinte (sarebbe ambiguo)."""
    assert NON_FIFA_DENY_LIST.isdisjoint(EXTINCT_TEAMS)


def test_aliases_do_not_target_deny_list():
    """Un alias non deve mai puntare a una squadra in deny-list (incoerenza)."""
    for src_name, dst_name in NAME_ALIASES.items():
        assert dst_name not in NON_FIFA_DENY_LIST, (
            f"Alias {src_name!r} -> {dst_name!r} mappa a una entità in deny-list"
        )


def test_aliases_do_not_chain():
    """Gli alias devono essere risolti in un solo passo (no catene src -> mid -> dst)."""
    for src_name, dst_name in NAME_ALIASES.items():
        assert dst_name not in NAME_ALIASES, (
            f"Alias chain rilevato: {src_name!r} -> {dst_name!r} -> ..."
        )


def test_predicates():
    assert is_in_deny_list("Sealand")
    assert is_in_deny_list("Padania")
    assert not is_in_deny_list("Italy")
    assert is_extinct("Yugoslavia")
    assert is_extinct("German DR")
    assert not is_extinct("Serbia")
    assert not is_extinct("Italy")


def test_normalize_passthrough_when_no_alias():
    assert normalize_team_name("Italy") == "Italy"
    assert normalize_team_name("Côte d'Ivoire") == "Côte d'Ivoire"


def test_known_extinct_listed():
    expected = {
        "German DR",
        "Yugoslavia",
        "Czechoslovakia",
        "North Vietnam",
        "Vietnam Republic",
    }
    assert expected.issubset(EXTINCT_TEAMS)


def test_known_non_fifa_listed():
    expected = {
        "Sealand",
        "Padania",
        "Yorkshire",
        "Catalonia",
        "Basque Country",
        "Tibet",
        "Vatican City",
    }
    assert expected.issubset(NON_FIFA_DENY_LIST)
