"""Indicizzazione delle squadre e rappresentazione vettoriale delle partite.

Il modello Dixon-Coles ha un parametro α e un β per ogni squadra. Lavoriamo con
indici interi `0..n_teams-1` invece che con nomi (stringhe) per:
- velocità: indexing in numpy array è O(1) e vettorizzabile;
- determinismo: ordering alfabetico stabile tra rebuild.

`TeamIndexer` è la mappa bidirezionale nome ↔ indice.
`MatchData` raggruppa gli array che la NLL legge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


class TeamIndexer:
    """Mappa bidirezionale `nome ↔ indice` con ordinamento deterministico (alfabetico)."""

    __slots__ = ("_teams", "_to_idx")

    def __init__(self, teams: Iterable[str]):
        # Deduplicate + sort: garantisce stabilità tra rebuild
        unique_sorted = sorted(set(teams))
        if not unique_sorted:
            raise ValueError("TeamIndexer richiede almeno una squadra")
        self._teams: tuple[str, ...] = tuple(unique_sorted)
        self._to_idx: dict[str, int] = {name: i for i, name in enumerate(self._teams)}

    def __len__(self) -> int:
        return len(self._teams)

    def __contains__(self, name: str) -> bool:
        return name in self._to_idx

    @property
    def n_teams(self) -> int:
        return len(self._teams)

    @property
    def teams(self) -> tuple[str, ...]:
        return self._teams

    def to_idx(self, name: str) -> int:
        return self._to_idx[name]

    def to_idx_array(self, names: Sequence[str] | pd.Series | np.ndarray) -> np.ndarray:
        return np.fromiter(
            (self._to_idx[str(n)] for n in names),
            dtype=np.int64,
            count=len(names),
        )

    def name(self, idx: int) -> str:
        return self._teams[idx]

    @classmethod
    def from_match_dataframe(cls, df: pd.DataFrame) -> "TeamIndexer":
        """Costruisce l'indexer dall'unione delle colonne `home_team` e `away_team`."""
        return cls(set(df["home_team"]).union(df["away_team"]))


@dataclass(frozen=True)
class MatchData:
    """Rappresentazione vettoriale delle partite usata dalla NLL.

    Tutti gli array hanno la stessa lunghezza `n_matches`. `home_advantage` è 1.0 quando
    la squadra di casa gioca effettivamente in casa (NOT neutral) e 0.0 altrimenti;
    moltiplica γ nel modello.
    """

    home_idx: np.ndarray            # int64, (n_matches,)
    away_idx: np.ndarray            # int64
    home_goals: np.ndarray          # int64
    away_goals: np.ndarray          # int64
    home_advantage: np.ndarray      # float64 (0.0 o 1.0)
    n_teams: int

    def __post_init__(self) -> None:
        n = len(self.home_idx)
        for name, arr in (
            ("away_idx", self.away_idx),
            ("home_goals", self.home_goals),
            ("away_goals", self.away_goals),
            ("home_advantage", self.home_advantage),
        ):
            if len(arr) != n:
                raise ValueError(
                    f"MatchData: lunghezza incoerente per '{name}' ({len(arr)} vs {n})"
                )

    @property
    def n_matches(self) -> int:
        return len(self.home_idx)


def prepare_match_data(df: pd.DataFrame, indexer: TeamIndexer) -> MatchData:
    """Converte un DataFrame di partite in `MatchData`.

    Richiede le colonne `home_team`, `away_team`, `home_score`, `away_score`, `neutral`.
    """
    required = {"home_team", "away_team", "home_score", "away_score", "neutral"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne richieste mancanti in df: {sorted(missing)}")
    return MatchData(
        home_idx=indexer.to_idx_array(df["home_team"]),
        away_idx=indexer.to_idx_array(df["away_team"]),
        home_goals=df["home_score"].to_numpy(dtype=np.int64),
        away_goals=df["away_score"].to_numpy(dtype=np.int64),
        home_advantage=(~df["neutral"].to_numpy(dtype=bool)).astype(np.float64),
        n_teams=indexer.n_teams,
    )


__all__ = ["TeamIndexer", "MatchData", "prepare_match_data"]
