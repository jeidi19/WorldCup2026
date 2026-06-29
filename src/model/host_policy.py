"""Policy del vantaggio casa per il Mondiale 2026 tri-ospitante (Issue #8).

Nessun Mondiale è mai stato giocato con tre paesi ospitanti contemporaneamente: la
policy qui è una **scelta di giudizio non validabile** (è esplicita in `config.yaml`,
sezione `host_advantage_2026`).

Regola:

- Se la squadra di casa coincide con il paese in cui si gioca → γ pieno.
- Se la squadra di casa è uno degli host ma gioca in un altro dei tre paesi ospitanti
  (es. USA gioca a Toronto) → γ ridotto (default 0.5).
- In tutti gli altri casi → γ = 0 (campo neutro).

`gamma_full_at_home`, `gamma_reduced_co_host`, `gamma_neutral` sono moltiplicatori
applicati al γ già stimato dal fit (Issue #5). Il γ globale è asimmetrico: si applica
solo alla `home`, mai alla `away` (cfr. parametrizzazione DC in `dixon_coles.py`).

Convenzione importante per il chiamante: nel knockout 2026 non esiste un "home" di
sorteggio. Per usare correttamente questa policy, la convenzione adottata è "il
caller passa come `home_team` la squadra che gioca nel proprio paese", se ce n'è
una; altrimenti la scelta home/away è indifferente perché lo scale risulterà 0
(campo neutro effettivo).
"""
from __future__ import annotations

from src.config import HostAdvantage2026


def host_advantage_scale(
    home_team: str,
    venue_country: str | None,
    policy: HostAdvantage2026,
) -> float:
    """Restituisce il moltiplicatore di γ da applicare alla `home_team`.

    Parametri:
        home_team:      nome FIFA della squadra di casa (come in `model.teams`).
        venue_country:  nome FIFA del paese ospitante della singola partita; `None`
                        per "non specificato" (caller userà comunque `is_neutral`).
        policy:         blocco `host_advantage_2026` del config.

    Ritorna `gamma_full_at_home`, `gamma_reduced_co_host` o `gamma_neutral`.
    """
    if venue_country is None:
        # Niente policy host: il caller userà `is_neutral` per decidere
        # se applicare γ pieno (False → 1.0) o γ = 0 (True → 0.0).
        # Conserviamo per chiarezza il default "in casa".
        return 1.0

    if home_team == venue_country:
        return policy.gamma_full_at_home

    host_teams = set(policy.host_teams)
    if home_team in host_teams and venue_country in host_teams:
        return policy.gamma_reduced_co_host

    return policy.gamma_neutral


__all__ = ["host_advantage_scale"]
