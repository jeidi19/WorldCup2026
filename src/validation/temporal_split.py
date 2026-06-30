"""Split temporale del dataset (Issue #9).

Il piano vieta esplicitamente split casuali: il modello deve essere validato SOLO su
partite future rispetto al training (cfr. Principio #3 in `docs/CLAUDE.md`). Lo split
qui implementato:

- Train: `date <= cutoff_date`
- Test:  `date >  cutoff_date`

`TemporalSplit.__post_init__` esegue automaticamente l'anti-leakage check (DoD #1):
nessuna partita di test entra mai nel training.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Split temporale tra train e test rispetto a `cutoff_date`."""

    cutoff_date: pd.Timestamp
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    @property
    def n_train(self) -> int:
        return len(self.train_df)

    @property
    def n_test(self) -> int:
        return len(self.test_df)

    def __post_init__(self) -> None:
        # DoD #1: anti-leakage automatico
        if self.n_train == 0:
            raise ValueError(f"Train set vuoto per cutoff_date={self.cutoff_date}")
        if self.n_test == 0:
            raise ValueError(f"Test set vuoto per cutoff_date={self.cutoff_date}")
        train_max = pd.Timestamp(self.train_df["date"].max())
        test_min = pd.Timestamp(self.test_df["date"].min())
        if train_max > self.cutoff_date:
            raise AssertionError(
                f"Leakage rilevato: train ha partita del {train_max.date()} > "
                f"cutoff {self.cutoff_date.date()}"
            )
        if test_min <= self.cutoff_date:
            raise AssertionError(
                f"Leakage rilevato: test ha partita del {test_min.date()} <= "
                f"cutoff {self.cutoff_date.date()}"
            )


def temporal_split(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
) -> TemporalSplit:
    """Splitta `df` in (train, test) rispetto a `cutoff_date`.

    Train include le partite di `cutoff_date`, test le esclude (cutoff inclusivo a sinistra).
    Le righe sono ordinate cronologicamente.
    """
    if "date" not in df.columns:
        raise ValueError("df deve contenere la colonna 'date'")
    cutoff = pd.Timestamp(cutoff_date)
    if df["date"].dtype != "datetime64[us]" and df["date"].dtype != "datetime64[ns]":
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    train = df.loc[df["date"] <= cutoff].copy()
    test = df.loc[df["date"] > cutoff].copy()
    train = train.sort_values("date").reset_index(drop=True)
    test = test.sort_values("date").reset_index(drop=True)
    return TemporalSplit(cutoff_date=cutoff, train_df=train, test_df=test)


__all__ = ["TemporalSplit", "temporal_split"]
