"""Test della host policy per il Mondiale 2026 (Issue #8).

Copre:
- DoD #2: la funzione applica γ pieno/ridotto/0 secondo i tre casi del piano.
- API integrata: `match_outcome_90` e `advance_probability` rispettano `venue_country`.

DoD #1 (γ stimato positivo e di ordine plausibile ~+0.3/+0.4) è verificato in
`test_fit_real_data.py::test_gamma_in_plausible_range`.
"""
from __future__ import annotations

import pytest

from src.config import HostAdvantage2026, load_config
from src.model.cascade import advance_probability, advance_probability_from_config
from src.model.fit import DixonColesModel
from src.model.host_policy import host_advantage_scale
from src.model.outcomes import expected_goals, match_outcome_90


def _model(alpha=(0.0, 0.0, 0.0, 0.0, 0.0),
           beta=(0.0, 0.0, 0.0, 0.0, 0.0),
           gamma=0.3, rho=-0.05,
           teams=("United States", "Canada", "Mexico",   # i 3 host del 2026 (necessari)
                  "Argentina", "Brazil")):                # 2 non-host per testare match neutro
    """Modello-fixture minimo per la host policy: 3 host + 2 non-host.

    I 3 host sono richiesti dal piano (USA/Canada/Mexico). I 2 non-host servono
    a testare partite "non-host vs non-host" (γ = 0 effettivo) e a verificare la
    simmetria con `is_neutral=True`.

    Nei test, `venue_country` può essere qualunque stringa-paese: NON deve essere
    nel `teams` del modello (è una stringa di policy, non un identificativo di
    squadra).
    """
    return DixonColesModel(
        teams=teams, alpha=alpha, beta=beta, gamma=gamma, rho=rho,
        n_matches_train=0, reference_date=None,
        fitted_at="2024-01-01T00:00:00+00:00",
        final_nll=0.0, n_iterations=0, converged=True, optimization_message="",
        half_life_years=2.0, identifiability_penalty_strength=1e4, tau_floor=1e-10,
    )


@pytest.fixture
def default_policy() -> HostAdvantage2026:
    return HostAdvantage2026(
        host_teams=["United States", "Canada", "Mexico"],
        gamma_full_at_home=1.0,
        gamma_reduced_co_host=0.5,
        gamma_neutral=0.0,
    )


# ---------------------------------------------------------------------------
# host_advantage_scale: i 3 casi del piano
# ---------------------------------------------------------------------------

def test_full_gamma_when_home_plays_in_its_own_country(default_policy):
    """USA in USA → γ pieno (1.0)."""
    assert host_advantage_scale("United States", "United States", default_policy) == 1.0
    assert host_advantage_scale("Canada", "Canada", default_policy) == 1.0
    assert host_advantage_scale("Mexico", "Mexico", default_policy) == 1.0


def test_reduced_gamma_when_host_plays_in_another_host(default_policy):
    """USA in Canada → γ ridotto (0.5)."""
    assert host_advantage_scale("United States", "Canada", default_policy) == 0.5
    assert host_advantage_scale("Canada", "Mexico", default_policy) == 0.5
    assert host_advantage_scale("Mexico", "United States", default_policy) == 0.5


def test_neutral_when_non_host_plays_in_host_country(default_policy):
    """Argentina in USA → γ = 0 (campo neutro effettivo)."""
    assert host_advantage_scale("Argentina", "United States", default_policy) == 0.0
    assert host_advantage_scale("Brazil", "Canada", default_policy) == 0.0
    assert host_advantage_scale("Italy", "Mexico", default_policy) == 0.0


def test_neutral_when_match_in_non_host_country(default_policy):
    """USA in Italia → γ = 0 (non è host country)."""
    assert host_advantage_scale("United States", "Italy", default_policy) == 0.0
    assert host_advantage_scale("Argentina", "Italy", default_policy) == 0.0


def test_venue_none_returns_default_in_home(default_policy):
    """`venue_country=None` → default 1.0 (la decisione passa a `is_neutral` del caller)."""
    assert host_advantage_scale("Argentina", None, default_policy) == 1.0


def test_custom_policy_uses_configured_scales():
    """Verifica che gli scale del config siano effettivamente usati (non hardcoded)."""
    custom = HostAdvantage2026(
        host_teams=["United States", "Canada", "Mexico"],
        gamma_full_at_home=0.9,
        gamma_reduced_co_host=0.3,
        gamma_neutral=0.05,
    )
    assert host_advantage_scale("United States", "United States", custom) == 0.9
    assert host_advantage_scale("United States", "Canada", custom) == 0.3
    assert host_advantage_scale("Argentina", "United States", custom) == 0.05


# ---------------------------------------------------------------------------
# expected_goals: priorità dei parametri
# ---------------------------------------------------------------------------

def test_expected_goals_explicit_scale_overrides_everything(default_policy):
    """`home_advantage_scale` esplicito ha priorità su `venue_country` e `is_neutral`."""
    model = _model(gamma=0.3)
    # Specifico scale=0.4 esplicito; venue_country e is_neutral verrebbero ignorati
    lam, mu = expected_goals(
        model, "Argentina", "Mexico",
        is_neutral=True,                    # ignorato
        venue_country="Mexico",              # ignorato
        host_policy=default_policy,
        home_advantage_scale=0.4,
    )
    # λ = exp(α_arg + β_mex + γ·0.4) = exp(0 + 0 + 0.12)
    import math
    assert lam == pytest.approx(math.exp(0.3 * 0.4))


def test_expected_goals_venue_country_overrides_is_neutral(default_policy):
    """`venue_country` ha priorità su `is_neutral` quando entrambi sono passati."""
    model = _model(gamma=0.3)
    # is_neutral=True direbbe γ=0, ma venue=United States in casa di United States → γ pieno
    lam, _ = expected_goals(
        model, "United States", "Argentina",
        is_neutral=True,                     # questo verrebbe ignorato
        venue_country="United States",
        host_policy=default_policy,
    )
    import math
    assert lam == pytest.approx(math.exp(0.3))  # γ pieno applicato


def test_expected_goals_is_neutral_when_no_venue(default_policy):
    """Senza `venue_country` né `home_advantage_scale`, `is_neutral` decide."""
    model = _model(gamma=0.3)
    lam_home, _ = expected_goals(model, "Argentina", "Brazil", is_neutral=False)
    lam_neut, _ = expected_goals(model, "Argentina", "Brazil", is_neutral=True)
    import math
    assert lam_home == pytest.approx(math.exp(0.3))
    assert lam_neut == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# match_outcome_90 con venue_country (host policy 2026)
# ---------------------------------------------------------------------------

def test_match_outcome_90_with_venue_in_own_country(default_policy):
    """USA in USA: stesso risultato di USA in casa con γ pieno."""
    model = _model(gamma=0.3)
    out_venue = match_outcome_90(
        model, "United States", "Argentina",
        venue_country="United States", host_policy=default_policy,
    )
    out_home = match_outcome_90(model, "United States", "Argentina", is_neutral=False)
    assert out_venue.p_home_win == pytest.approx(out_home.p_home_win)


def test_match_outcome_90_with_venue_co_host_has_intermediate_advantage(default_policy):
    """USA in Canada: vantaggio intermedio (0.5·γ) → tra in casa USA e neutro."""
    model = _model(gamma=0.3)
    out_full = match_outcome_90(model, "United States", "Argentina", is_neutral=False)
    out_co = match_outcome_90(
        model, "United States", "Argentina",
        venue_country="Canada", host_policy=default_policy,
    )
    out_neut = match_outcome_90(model, "United States", "Argentina", is_neutral=True)
    # P(home) cresce monotonicamente con lo scale
    assert out_neut.p_home_win < out_co.p_home_win < out_full.p_home_win


def test_match_outcome_90_with_venue_neutral_for_non_host(default_policy):
    """Argentina in USA: γ=0, equivalente a campo neutro."""
    model = _model(gamma=0.3)
    out_venue = match_outcome_90(
        model, "Argentina", "Brazil",
        venue_country="United States", host_policy=default_policy,
    )
    out_neut = match_outcome_90(model, "Argentina", "Brazil", is_neutral=True)
    assert out_venue.p_home_win == pytest.approx(out_neut.p_home_win)
    # E `is_neutral` effettivo deve essere True (scale risultato = 0)
    assert out_venue.is_neutral is True


# ---------------------------------------------------------------------------
# advance_probability con venue_country
# ---------------------------------------------------------------------------

def test_advance_probability_with_venue_country(default_policy):
    """L'integrazione della host policy si propaga alla cascata 90'/ET/rigori."""
    model = _model(gamma=0.3)
    # Mexico in casa (Mexico) vs Argentina
    out_home = advance_probability(
        model, "Mexico", "Argentina",
        venue_country="Mexico", host_policy=default_policy,
    )
    # Mexico in Canada vs Argentina
    out_co = advance_probability(
        model, "Mexico", "Argentina",
        venue_country="Canada", host_policy=default_policy,
    )
    # Mexico in Italia (non-host) vs Argentina → effettivamente neutro
    out_neut = advance_probability(
        model, "Mexico", "Argentina",
        venue_country="Italy", host_policy=default_policy,
    )
    assert out_neut.p_home_advance < out_co.p_home_advance < out_home.p_home_advance
    assert out_neut.is_neutral is True
    assert out_home.is_neutral is False


def test_advance_probability_from_config_propagates_venue():
    cfg = load_config()
    model = _model(gamma=0.3)
    out_venue = advance_probability_from_config(
        model, "United States", "Argentina", cfg, venue_country="United States",
    )
    out_neut = advance_probability_from_config(
        model, "United States", "Argentina", cfg, is_neutral=True,
    )
    assert out_venue.p_home_advance > out_neut.p_home_advance


# ---------------------------------------------------------------------------
# Sanity: la policy resta coerente quando lo scale è 0 (non-host venue)
# ---------------------------------------------------------------------------

def test_neutral_via_venue_matches_explicit_neutral(default_policy):
    """Forzare γ=0 via `venue_country` non-host == `is_neutral=True`."""
    model = _model(gamma=0.3)
    out_venue = match_outcome_90(
        model, "Argentina", "Brazil",
        venue_country="Italy", host_policy=default_policy,
    )
    out_neut = match_outcome_90(model, "Argentina", "Brazil", is_neutral=True)
    assert out_venue.p_home_win == pytest.approx(out_neut.p_home_win)
    assert out_venue.p_draw == pytest.approx(out_neut.p_draw)
    assert out_venue.p_away_win == pytest.approx(out_neut.p_away_win)
