"""Fit del modello Dixon-Coles via L-BFGS-B (Issue #5).

`fit_model(data, weights, config)` minimizza `dixon_coles_nll` su un vettore di
parametri lungo `2·n_teams + 2`:

- alpha_i (attacco), beta_i (difesa): liberi (-inf, +inf), centrati dalla penalty di
  identificabilità in NLL;
- gamma > 0: vantaggio casa;
- rho ∈ [-0.2, 0.2]: correzione Dixon-Coles.

Bounds rilassati leggermente all'interno dell'intervallo per evitare numerical issues
sui bordi. Dopo il fit, eseguiamo una rifinitura cosmetica: shift di `mean(alpha)` su
alpha (negativa) e beta (positiva) per centrare esattamente gli `alpha`; questa
trasformazione lascia invariata la NLL data (è esattamente l'invarianza del modello).

Output: `DixonColesModel` (frozen dataclass) con i rating, metadata di fit, e
`save`/`load` JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import AppConfig, load_config
from src.model.dixon_coles import (
    decode_params,
    dixon_coles_nll,
    encode_params,
    initial_params,
    n_params,
)
from src.model.indexing import MatchData, TeamIndexer, prepare_match_data

logger = logging.getLogger(__name__)

# Bounds rilassati per evitare bordo esatto (cfr. piano: rho in [-0.2, 0.2], gamma > 0)
_GAMMA_LOWER = 1e-6
_RHO_BOUND_MARGIN = 1e-3  # rho in [-0.2+eps, 0.2-eps]


# ---------------------------------------------------------------------------
# Modello fittato
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DixonColesModel:
    """Risultato del fit Dixon-Coles. Serializzabile in JSON."""

    teams: tuple[str, ...]
    alpha: tuple[float, ...]     # len = n_teams
    beta: tuple[float, ...]
    gamma: float
    rho: float

    n_matches_train: int
    reference_date: str | None       # ISO yyyy-mm-dd (None se non specificata)
    fitted_at: str                   # ISO timestamp UTC
    final_nll: float
    n_iterations: int
    converged: bool
    optimization_message: str

    half_life_years: float
    identifiability_penalty_strength: float
    tau_floor: float

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_params(self) -> int:
        return n_params(self.n_teams)

    def alpha_of(self, team: str) -> float:
        return self.alpha[self._idx(team)]

    def beta_of(self, team: str) -> float:
        return self.beta[self._idx(team)]

    def strength_of(self, team: str) -> float:
        """Proxy di 'forza complessiva' = alpha − beta (attacco meno difesa-debole)."""
        idx = self._idx(team)
        return self.alpha[idx] - self.beta[idx]

    def _idx(self, team: str) -> int:
        try:
            return self.teams.index(team)
        except ValueError:
            raise KeyError(f"Squadra non presente nel modello: {team!r}") from None

    def to_dataframe(self) -> pd.DataFrame:
        """Tabella per squadra: alpha, beta, strength = alpha − beta."""
        alpha_arr = np.asarray(self.alpha)
        beta_arr = np.asarray(self.beta)
        return pd.DataFrame(
            {
                "team": self.teams,
                "alpha": alpha_arr,
                "beta": beta_arr,
                "strength": alpha_arr - beta_arr,
            }
        ).sort_values("strength", ascending=False).reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            # Override tuple-as-list per JSON
            "teams": list(self.teams),
            "alpha": list(self.alpha),
            "beta": list(self.beta),
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "DixonColesModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            teams=tuple(data["teams"]),
            alpha=tuple(data["alpha"]),
            beta=tuple(data["beta"]),
            gamma=float(data["gamma"]),
            rho=float(data["rho"]),
            n_matches_train=int(data["n_matches_train"]),
            reference_date=data.get("reference_date"),
            fitted_at=data["fitted_at"],
            final_nll=float(data["final_nll"]),
            n_iterations=int(data["n_iterations"]),
            converged=bool(data["converged"]),
            optimization_message=data["optimization_message"],
            half_life_years=float(data["half_life_years"]),
            identifiability_penalty_strength=float(data["identifiability_penalty_strength"]),
            tau_floor=float(data["tau_floor"]),
        )


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def make_bounds(n_teams: int, rho_bounds: tuple[float, float]) -> list[tuple[float | None, float | None]]:
    """Bounds in formato L-BFGS-B: alpha/beta liberi, gamma>0, rho nei limiti."""
    rho_lo, rho_hi = rho_bounds
    rho_lo += _RHO_BOUND_MARGIN
    rho_hi -= _RHO_BOUND_MARGIN
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * (2 * n_teams)
    bounds.append((_GAMMA_LOWER, None))   # gamma
    bounds.append((rho_lo, rho_hi))       # rho
    return bounds


# ---------------------------------------------------------------------------
# Centratura post-hoc
# ---------------------------------------------------------------------------

def center_alpha_beta(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Applica la trasformazione invariante α += c, β −= c per ottenere mean(α) = 0.

    Lascia inalterati i log_lam e log_mu del modello (è l'invarianza identificabile).
    Cosmetica: rende i parametri immediatamente interpretabili (α positivo = attacco
    sopra media; β positivo = difesa sotto media).
    """
    c_alpha = float(alpha.mean())
    return alpha - c_alpha, beta + c_alpha


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_model(
    data: MatchData,
    weights: np.ndarray,
    config: AppConfig,
    indexer: TeamIndexer,
    *,
    reference_date: pd.Timestamp | str | None = None,
    initial_gamma: float = 0.3,
    initial_rho: float = -0.05,
    max_iter: int = 500,
    max_fun: int = 500_000,
    tolerance: float = 1e-7,
) -> DixonColesModel:
    """Minimizza la NLL Dixon-Coles via L-BFGS-B (gradient numerico).

    `data`, `weights`: input training set (vedi `prepare_match_data`).
    `config`: parametri (`identifiability_penalty_strength`, `tau_floor`,
              `goals.rho_bounds`, `time_decay.half_life_years` per i metadata).
    `indexer`: serve per i nomi delle squadre nel modello restituito.
    """
    if data.n_teams != indexer.n_teams:
        raise ValueError(
            f"Incongruenza n_teams: data={data.n_teams} vs indexer={indexer.n_teams}"
        )

    p0 = initial_params(indexer.n_teams, gamma_init=initial_gamma, rho_init=initial_rho)
    bounds = make_bounds(indexer.n_teams, tuple(config.goals.rho_bounds))

    pen = config.dixon_coles.identifiability_penalty_strength
    tau_floor = config.dixon_coles.tau_floor

    def objective(p: np.ndarray) -> float:
        return dixon_coles_nll(
            p, data, weights,
            identifiability_penalty_strength=pen,
            tau_floor=tau_floor,
        )

    logger.info(
        "Fit Dixon-Coles: %d squadre, %d partite, %d parametri",
        indexer.n_teams, data.n_matches, n_params(indexer.n_teams),
    )

    result = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        jac=None,                # finite-diff
        bounds=bounds,
        options={
            "maxiter": max_iter,
            "maxfun": max_fun,
            "ftol": tolerance,
            "gtol": tolerance,
        },
    )

    alpha, beta, gamma, rho = decode_params(result.x, indexer.n_teams)
    alpha, beta = center_alpha_beta(alpha, beta)

    # NLL ricomputata sui parametri centrati (deve coincidere con result.fun a meno di penalty)
    final_nll = float(objective(encode_params(alpha, beta, gamma, rho)))

    ref_iso: str | None
    if reference_date is None:
        ref_iso = None
    else:
        ref_iso = str(pd.Timestamp(reference_date).date())

    model = DixonColesModel(
        teams=tuple(indexer.teams),
        alpha=tuple(float(x) for x in alpha),
        beta=tuple(float(x) for x in beta),
        gamma=float(gamma),
        rho=float(rho),
        n_matches_train=data.n_matches,
        reference_date=ref_iso,
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        final_nll=final_nll,
        n_iterations=int(result.nit),
        converged=bool(result.success),
        optimization_message=str(result.message),
        half_life_years=float(config.time_decay.half_life_years),
        identifiability_penalty_strength=float(pen),
        tau_floor=float(tau_floor),
    )

    logger.info(
        "Fit completato: %d iter, NLL=%.4f, gamma=%.4f, rho=%.4f, converged=%s",
        result.nit, final_nll, gamma, rho, result.success,
    )
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit del modello Dixon-Coles (Issue #5).")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Parquet pesato (default: data/processed/matches_weighted.parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path JSON del modello (default: data/models/dixon_coles.json).",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default=None,
        help="Reference_date dei pesi usata (solo metadata).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Stampa i top N rating dopo il fit (default: 15).",
    )
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-fun", type=int, default=500_000,
                        help="Budget di valutazioni di NLL (finite-diff: ~n_params per iter).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    config = load_config()
    repo_root = Path(__file__).resolve().parents[2]
    input_path = (args.input or (repo_root / config.paths.data_processed / "matches_weighted.parquet")).resolve()
    output_path = (args.output or (repo_root / config.paths.models / "dixon_coles.json")).resolve()

    logger.info("Carico %s", input_path)
    df = pd.read_parquet(input_path)
    indexer = TeamIndexer.from_match_dataframe(df)
    data = prepare_match_data(df, indexer)
    weights = df["weight"].to_numpy(dtype=np.float64)

    model = fit_model(
        data, weights, config, indexer,
        reference_date=args.reference_date,
        max_iter=args.max_iter,
        max_fun=args.max_fun,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    logger.info("Salvato modello in %s", output_path)

    if args.top > 0:
        ratings = model.to_dataframe().head(args.top)
        logger.info("Top %d rating (strength = alpha - beta):", args.top)
        for _, row in ratings.iterrows():
            logger.info(
                "  %3d. %-30s  alpha=%+.3f  beta=%+.3f  strength=%+.3f",
                int(row.name) + 1, row["team"], row["alpha"], row["beta"], row["strength"],
            )


if __name__ == "__main__":
    main()
