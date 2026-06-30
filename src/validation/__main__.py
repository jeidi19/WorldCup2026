"""CLI per la grid search di emivita (Issue #9).

Esempi:
    python -m src.validation --cutoff 2022-12-31
    python -m src.validation --cutoff 2022-12-31 --half-lives 1.5 2.0 2.5
    python -m src.validation --cutoff 2022-12-31 --output data/validation/run1.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.validation.grid_search import grid_search_half_life


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cutoff", type=str, default="2022-12-31",
        help="Cutoff date YYYY-MM-DD (default: 2022-12-31).",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Parquet clean (default: data/processed/matches_clean.parquet).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path JSON dei risultati (default: data/validation/grid_xi_<cutoff>.json).",
    )
    parser.add_argument(
        "--half-lives", type=float, nargs="+", default=None,
        help="Override del grid di emivita (default: config.time_decay.half_life_years_grid).",
    )
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-fun", type=int, default=500_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    config = load_config()
    repo = Path(__file__).resolve().parents[2]

    input_path = (args.input or (repo / "data" / "processed" / "matches_clean.parquet")).resolve()
    cutoff_iso = args.cutoff
    output_path = (
        args.output
        or (repo / "data" / "validation" / f"grid_xi_{cutoff_iso}.json")
    ).resolve()

    df = pd.read_parquet(input_path)
    result = grid_search_half_life(
        df, cutoff_iso, config,
        half_life_grid=args.half_lives,
        max_iter=args.max_iter,
        max_fun=args.max_fun,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cutoff_date": result.cutoff_date,
        "n_train": result.n_train,
        "n_test_evaluated": result.n_test_evaluated,
        "best_half_life_years": result.best_half_life,
        "best_log_loss": result.best_log_loss,
        "grid_rows": result.half_life_results.to_dict(orient="records"),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result.half_life_results.to_csv(
        output_path.with_suffix(".csv"), index=False
    )
    print(f"\nBest half_life = {result.best_half_life:.2f} anni "
          f"(log_loss={result.best_log_loss:.4f})")
    print(f"Salvato in {output_path}")


if __name__ == "__main__":
    main()
