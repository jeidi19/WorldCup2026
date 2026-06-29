"""Politiche di normalizzazione e filtro per i nomi delle nazionali (Issue #2).

Il dataset `martj42/international-football-results-from-1872-to-2017` contiene 336 entità
distinte: federazioni FIFA attuali, federazioni storiche estinte, e alcune squadre
regionali/satira che non sono né riconosciute internazionalmente né coerenti col perimetro
del modello (predizione del Mondiale 2026, che è una competizione FIFA).

Tre insiemi disgiunti:

- `NON_FIFA_DENY_LIST` — entità non-FIFA o regionali/amatoriali. Le partite che le coinvolgono
  vengono RIMOSSE dal dataset processato (sono prevalentemente rumore o tornei satira).

- `EXTINCT_TEAMS` — entità storiche scomparse (Germany DR, Yugoslavia, Czechoslovakia, ...).
  Le partite vengono TENUTE: sono segnale utile per stimare le squadre superstiti (es.
  Italia vs Yugoslavia 1990 dice qualcosa sulla forza dell'Italia in quel momento). Le
  entità estinte avranno parametri α_i, β_i latenti ma NON saranno predette al 2026 perché
  non sono in tabellone.

- `NAME_ALIASES` — mapping verso forma canonica per refusi/varianti. Per ora vuoto: il
  dataset martj42 è già abbastanza normalizzato (Burma → Myanmar, Korea Republic ↔ South
  Korea sono già fissati alla forma odierna usata dal CSV). Da aggiornare se in futuro
  emergono casi.

Riferimento DoD #1 di Issue #2: "Nessun nome squadra ambiguo/duplicato non risolto (lista
mappature in un file versionato)" — questo file è quel registro versionato.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Deny-list: partite che le coinvolgono vengono DROPPATE.
# -----------------------------------------------------------------------------
NON_FIFA_DENY_LIST: frozenset[str] = frozenset(
    {
        # Micronazioni / satira
        "Sealand",
        "Padania",
        "Yorkshire",
        "Sark",
        # Regionali / sub-statali
        "Catalonia",
        "Basque Country",
        "Iraqi Kurdistan",
        "Kurdistan",
        "Zanzibar",
        "Parishes of Jersey",
        # Territori non-FIFA
        "Tibet",
        "Greenland",
        "Vatican City",
        "Niue",
        "Tuvalu",
        "Jersey",
        "Guernsey",
        "Monaco",
        # Diaspore / squadre etniche non riconosciute
        "United Koreans in Japan",
        "Sapmi",
    }
)

# -----------------------------------------------------------------------------
# Entità estinte: mantenute come segnale, NON predette al 2026.
# -----------------------------------------------------------------------------
EXTINCT_TEAMS: frozenset[str] = frozenset(
    {
        "German DR",         # 1952–1990, poi assorbita nella Germania riunificata
        "Yugoslavia",        # disgregata 1992 → Serbia, Croatia, Bosnia, Slovenia, Macedonia, Montenegro
        "Czechoslovakia",    # disgregata 1993 → Czech Republic, Slovakia
        "North Vietnam",     # unificato in Vietnam nel 1976
        "Vietnam Republic",  # unificato in Vietnam nel 1976
    }
)

# -----------------------------------------------------------------------------
# Alias di normalizzazione (refusi, varianti). Vuoto al momento.
# -----------------------------------------------------------------------------
NAME_ALIASES: dict[str, str] = {}


def normalize_team_name(name: str) -> str:
    """Restituisce la forma canonica del nome di una squadra (o il nome se non aliasato)."""
    return NAME_ALIASES.get(name, name)


def is_in_deny_list(name: str) -> bool:
    """True se la squadra va esclusa dal dataset processato."""
    return name in NON_FIFA_DENY_LIST


def is_extinct(name: str) -> bool:
    """True se la squadra è estinta (mantenuta come segnale, ma non predetta al 2026)."""
    return name in EXTINCT_TEAMS


__all__ = [
    "NON_FIFA_DENY_LIST",
    "EXTINCT_TEAMS",
    "NAME_ALIASES",
    "normalize_team_name",
    "is_in_deny_list",
    "is_extinct",
]
