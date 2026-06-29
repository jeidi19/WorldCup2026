"""Politiche di classificazione delle competizioni (Issue #3).

Ogni partita viene assegnata a un **bucket** che mappa al moltiplicatore di peso
configurato in `config.competition_weights`. I bucket sono fissati dalla sezione 1
di `docs/CLAUDE.md`:

| Bucket                 | Peso  | Esempi                                          |
|------------------------|-------|-------------------------------------------------|
| `mondiali`             | 1.0   | FIFA World Cup, Confederations Cup, Finalissima |
| `finali_continentali`  | 1.0   | UEFA Euro, Copa América, AFCON, Asian Cup, Gold |
| `qualificazioni`       | 0.8   | tutto `... qualification` (pattern fallback)    |
| `nations_league`       | 0.8   | UEFA / CONCACAF Nations League                  |
| `sub_continentali`     | 0.6   | CECAFA, COSAFA, Gulf Cup, AFF, SAFF, Arab Cup   |
| `amichevole`           | 0.4   | Friendly, Merdeka, King's Cup, BHC, Nordic      |
| (unmapped → default)   | 0.6   | tornei rari / non riconosciuti                  |

Inoltre `TOURNAMENT_DROP_LIST` enumera i tornei **multi-sport / U-23** (Asian
Games, SEA Games, Island Games, Pacific Games, Pan American Games, ...) le cui
partite vengono RIMOSSE dal training set: schierano formazioni B/U-23 e
contaminerebbero la stima delle forze delle nazionali A.

Riferimento: DoD di Issue #3 — "Tornei non mappati → default conservativo (es.
0.6) e loggare i nomi non mappati per revisione".
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Costanti dei bucket. I valori sono ID stringa stabili che il config.yaml usa
# come chiavi in `competition_weights`.
# -----------------------------------------------------------------------------
BUCKET_AMICHEVOLE = "amichevole"
BUCKET_QUALIFICAZIONI = "qualificazioni"
BUCKET_NATIONS_LEAGUE = "nations_league"
BUCKET_SUB_CONTINENTALI = "sub_continentali"
BUCKET_FINALI_CONTINENTALI = "finali_continentali"
BUCKET_MONDIALI = "mondiali"

ALL_BUCKETS: frozenset[str] = frozenset(
    {
        BUCKET_AMICHEVOLE,
        BUCKET_QUALIFICAZIONI,
        BUCKET_NATIONS_LEAGUE,
        BUCKET_SUB_CONTINENTALI,
        BUCKET_FINALI_CONTINENTALI,
        BUCKET_MONDIALI,
    }
)

# -----------------------------------------------------------------------------
# Tornei multi-sport / U-23: le partite vengono DROPPATE prima del fit.
# Ragione: schierano formazioni B / Under-23, non rappresentano la forza A.
# -----------------------------------------------------------------------------
TOURNAMENT_DROP_LIST: frozenset[str] = frozenset(
    {
        # Olimpiadi: U-23 dal 1992 in poi (3 fuori quota). Storicamente già amateur.
        "Olympic Games",
        # Multi-sport "Games"
        "Asian Games",
        "Southeast Asian Games",
        "South Pacific Games",
        "Pacific Games",
        "Pacific Mini Games",
        "Island Games",
        "Indian Ocean Island Games",
        "Pan American Games",
        "Mediterranean Games",
        "Central American Games",
        "Central American and Caribbean Games",
        "Bolivarian Games",
        "South American Games",
        "All-African Games",
        "All-Africa Games",
        "African Games",
        "Southeast Asian Peninsular Games",
        "South Asian Games",
        "African Friendship Games",
        "Lusophony Games",
        "Lusophony Cup",
        "Universiade",
        "Military World Games",
        "ALBA Games",
        # Tornei non-FIFA (alternativi): contaminano i rating delle squadre FIFA
        "CONIFA World Football Cup",
        "CONIFA European Football Cup",
        "Viva World Cup",
        # Multi-sport storici (presenti nel dataset)
        "South Pacific Mini Games",
        "Far Eastern Championship Games",
        "East Asian Games",
        "Inter-Allied Games",      # post-WWI military multi-sport (1919)
        "GaNEFo",                  # Games of New Emerging Forces (1963, anti-Olympic)
        "Inter Games",
    }
)

# -----------------------------------------------------------------------------
# Mapping ESPLICITO torneo → bucket. Copre la lista dei top tornei del dataset
# martj42 (per partite cumulate). Per i tornei minori e gli alias non listati,
# i pattern fallback (qualification, Nations League) decidono; il resto cade
# nel `default_unmapped` (loggato).
# -----------------------------------------------------------------------------
TOURNAMENT_BUCKETS: dict[str, str] = {
    # ---- Mondiali (1.0): tornei FIFA top tier ----
    "FIFA World Cup": BUCKET_MONDIALI,
    "Confederations Cup": BUCKET_MONDIALI,
    "Finalissima": BUCKET_MONDIALI,
    "Artemio Franchi Trophy": BUCKET_MONDIALI,  # antenato della Finalissima
    "Intercontinental Cup": BUCKET_MONDIALI,
    # ---- Finali continentali (1.0): top tornei di confederazione ----
    "UEFA Euro": BUCKET_FINALI_CONTINENTALI,
    "Copa América": BUCKET_FINALI_CONTINENTALI,
    "African Cup of Nations": BUCKET_FINALI_CONTINENTALI,
    "AFC Asian Cup": BUCKET_FINALI_CONTINENTALI,
    "Gold Cup": BUCKET_FINALI_CONTINENTALI,
    "CONCACAF Championship": BUCKET_FINALI_CONTINENTALI,  # vecchia denominazione del Gold Cup
    "CONCACAF Gold Cup": BUCKET_FINALI_CONTINENTALI,
    "Oceania Nations Cup": BUCKET_FINALI_CONTINENTALI,
    "OFC Nations Cup": BUCKET_FINALI_CONTINENTALI,
    # ---- Nations League (0.8) ----
    "UEFA Nations League": BUCKET_NATIONS_LEAGUE,
    "CONCACAF Nations League": BUCKET_NATIONS_LEAGUE,
    # ---- Sub-continentali / zonali (0.6) ----
    "CECAFA Cup": BUCKET_SUB_CONTINENTALI,
    "COSAFA Cup": BUCKET_SUB_CONTINENTALI,
    "Amílcar Cabral Cup": BUCKET_SUB_CONTINENTALI,
    "WAFU Cup": BUCKET_SUB_CONTINENTALI,
    "WAFF Championship": BUCKET_SUB_CONTINENTALI,
    "Gulf Cup": BUCKET_SUB_CONTINENTALI,
    "AFF Championship": BUCKET_SUB_CONTINENTALI,
    "SAFF Cup": BUCKET_SUB_CONTINENTALI,
    "EAFF Championship": BUCKET_SUB_CONTINENTALI,
    "UNCAF Cup": BUCKET_SUB_CONTINENTALI,
    "CFU Caribbean Cup": BUCKET_SUB_CONTINENTALI,
    "Arab Cup": BUCKET_SUB_CONTINENTALI,
    "Arab Nations Cup": BUCKET_SUB_CONTINENTALI,
    "Pan Arab Games": BUCKET_SUB_CONTINENTALI,
    "CCCF Championship": BUCKET_SUB_CONTINENTALI,
    "Baltic Cup": BUCKET_SUB_CONTINENTALI,
    "Balkan Cup": BUCKET_SUB_CONTINENTALI,
    "AFC Challenge Cup": BUCKET_SUB_CONTINENTALI,
    "UDEAC Cup": BUCKET_SUB_CONTINENTALI,
    "UNIFFAC Cup": BUCKET_SUB_CONTINENTALI,
    "Melanesia Cup": BUCKET_SUB_CONTINENTALI,
    "Polynesia Cup": BUCKET_SUB_CONTINENTALI,
    "West African Cup": BUCKET_SUB_CONTINENTALI,
    "Windward Islands Tournament": BUCKET_SUB_CONTINENTALI,
    "Coupe de l'Outre-Mer": BUCKET_SUB_CONTINENTALI,
    "Tournament of Three Nations": BUCKET_SUB_CONTINENTALI,
    "Inter Games Football Tournament": BUCKET_SUB_CONTINENTALI,
    "Pan American Championship": BUCKET_SUB_CONTINENTALI,
    "ASEAN Championship": BUCKET_SUB_CONTINENTALI,
    "MSG Prime Minister's Cup": BUCKET_SUB_CONTINENTALI,
    "Dynasty Cup": BUCKET_SUB_CONTINENTALI,
    "AFC Solidarity Cup": BUCKET_SUB_CONTINENTALI,
    "Nile Basin Tournament": BUCKET_SUB_CONTINENTALI,
    # ---- Amichevoli (0.4): include amichevoli a inviti di prestigio ----
    "Friendly": BUCKET_AMICHEVOLE,
    "Merdeka Tournament": BUCKET_AMICHEVOLE,
    "British Home Championship": BUCKET_AMICHEVOLE,
    "Nordic Championship": BUCKET_AMICHEVOLE,
    "King's Cup": BUCKET_AMICHEVOLE,
    "Korea Cup": BUCKET_AMICHEVOLE,
    "Central European International Cup": BUCKET_AMICHEVOLE,
    "Lunar New Year Cup": BUCKET_AMICHEVOLE,
    "Cyprus International Tournament": BUCKET_AMICHEVOLE,
    "USA Cup": BUCKET_AMICHEVOLE,
    "China Cup": BUCKET_AMICHEVOLE,
    "Carlsberg Cup": BUCKET_AMICHEVOLE,
    "Atlantic Cup": BUCKET_AMICHEVOLE,
    "Tournoi de France": BUCKET_AMICHEVOLE,
    "Pacific Cup": BUCKET_AMICHEVOLE,
    "Dunhill Cup": BUCKET_AMICHEVOLE,
    "Rous Cup": BUCKET_AMICHEVOLE,
    "Mubarak Cup": BUCKET_AMICHEVOLE,
    "Kirin Cup": BUCKET_AMICHEVOLE,
    "Tianjin TEDA Cup": BUCKET_AMICHEVOLE,
    "Nehru Cup": BUCKET_AMICHEVOLE,
    "VFF Cup": BUCKET_AMICHEVOLE,
    "Millennium Cup": BUCKET_AMICHEVOLE,
    "Simba Tournament": BUCKET_AMICHEVOLE,
    "Mahinda Rajapaksa Cup": BUCKET_AMICHEVOLE,
    "Philippine Peace Cup": BUCKET_AMICHEVOLE,
    "Copa Lipton": BUCKET_AMICHEVOLE,
    "Copa Newton": BUCKET_AMICHEVOLE,
    "Copa Premier Centenario": BUCKET_AMICHEVOLE,
    "Copa del Atlántico": BUCKET_AMICHEVOLE,
    "Copa Confraternidad Centroamericana": BUCKET_AMICHEVOLE,
    "FIFA Series": BUCKET_AMICHEVOLE,
    "Indonesia Tournament": BUCKET_AMICHEVOLE,
    "Vietnam Independence Cup": BUCKET_AMICHEVOLE,
    "Palestine Cup": BUCKET_AMICHEVOLE,
    "Malta International Tournament": BUCKET_AMICHEVOLE,
    "Cyprus Tournament": BUCKET_AMICHEVOLE,
    "Three Nations Cup": BUCKET_AMICHEVOLE,
    "Brazil Independence Cup": BUCKET_AMICHEVOLE,
    "CONCACAF Series": BUCKET_AMICHEVOLE,
    "United Arab Emirates Friendship Tournament": BUCKET_AMICHEVOLE,
    "Copa Chevallier Boutell": BUCKET_AMICHEVOLE,
    "Merlion Cup": BUCKET_AMICHEVOLE,
    "Copa Roca": BUCKET_AMICHEVOLE,
    "Copa Paz del Chaco": BUCKET_AMICHEVOLE,
    "Prime Minister's Cup": BUCKET_AMICHEVOLE,
    "Kirin Challenge Cup": BUCKET_AMICHEVOLE,
    "Soccer Ashes": BUCKET_AMICHEVOLE,
    "ABCS Tournament": BUCKET_AMICHEVOLE,
    "Copa del Pacífico": BUCKET_AMICHEVOLE,
    "Copa Rio Branco": BUCKET_AMICHEVOLE,
    "Jordan International Tournament": BUCKET_AMICHEVOLE,
    "Copa Oswaldo Cruz": BUCKET_AMICHEVOLE,
    "Copa Carlos Dittborn": BUCKET_AMICHEVOLE,
    "Copa Juan Pinto Durán": BUCKET_AMICHEVOLE,
    "Copa Premio Honor Uruguayo": BUCKET_AMICHEVOLE,
    "Copa Premio Honor Argentino": BUCKET_AMICHEVOLE,
    "Copa Artigas": BUCKET_AMICHEVOLE,
    "Kuneitra Cup": BUCKET_AMICHEVOLE,
    "Trans-Tasman Cup": BUCKET_AMICHEVOLE,
    "Miami Cup": BUCKET_AMICHEVOLE,
    "King Hassan II Tournament": BUCKET_AMICHEVOLE,
    "Unity Cup": BUCKET_AMICHEVOLE,
    "Tournament Burkina Faso": BUCKET_AMICHEVOLE,
    "SKN Football Festival": BUCKET_AMICHEVOLE,
}


def _matches_qualification_pattern(name: str) -> bool:
    """True se il nome termina per 'qualification' (case-sensitive, come nel dataset)."""
    return name.endswith(" qualification") or name.endswith(" Qualification")


def _matches_nations_league_pattern(name: str) -> bool:
    """True se il nome contiene 'Nations League' (e non è una qualification)."""
    return "Nations League" in name and not _matches_qualification_pattern(name)


def classify_tournament(name: str) -> str | None:
    """Ritorna il bucket di appartenenza o `None` se il torneo non è classificabile.

    Ordine di risoluzione:
    1. Mapping esplicito in `TOURNAMENT_BUCKETS`.
    2. Pattern fallback: `... qualification` → `qualificazioni`.
    3. Pattern fallback: contiene `Nations League` → `nations_league`.
    4. Altrimenti `None` (verrà loggato e applicato il `default_unmapped`).
    """
    bucket = TOURNAMENT_BUCKETS.get(name)
    if bucket is not None:
        return bucket
    if _matches_qualification_pattern(name):
        return BUCKET_QUALIFICAZIONI
    if _matches_nations_league_pattern(name):
        return BUCKET_NATIONS_LEAGUE
    return None


def should_drop_tournament(name: str) -> bool:
    """True se le partite di quel torneo vanno droppate (multi-sport / U-23)."""
    return name in TOURNAMENT_DROP_LIST


__all__ = [
    "BUCKET_AMICHEVOLE",
    "BUCKET_QUALIFICAZIONI",
    "BUCKET_NATIONS_LEAGUE",
    "BUCKET_SUB_CONTINENTALI",
    "BUCKET_FINALI_CONTINENTALI",
    "BUCKET_MONDIALI",
    "ALL_BUCKETS",
    "TOURNAMENT_BUCKETS",
    "TOURNAMENT_DROP_LIST",
    "classify_tournament",
    "should_drop_tournament",
]
