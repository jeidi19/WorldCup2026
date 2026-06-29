"""Caricamento e validazione della configurazione del progetto.

Il file `config.yaml` alla radice del repo contiene i parametri di modello, pesi,
mercato e host policy. Qui li tipizziamo con Pydantic v2 in modo che eventuali
errori di configurazione (chiavi sconosciute, valori fuori range) emergano subito,
non a runtime nel mezzo del fit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field

_StrictModel = ConfigDict(extra="forbid")

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


class PerimeterConfig(BaseModel):
    model_config = _StrictModel
    scope: str = "knockout_only"


class GoalsConfig(BaseModel):
    model_config = _StrictModel
    distribution: str = "poisson_dixon_coles"
    rho_bounds: tuple[float, float] = (-0.2, 0.2)


class CompetitionWeights(BaseModel):
    model_config = _StrictModel
    amichevole: PositiveFloat = 0.4
    qualificazioni: PositiveFloat = 0.8
    nations_league: PositiveFloat = 0.8
    finali_continentali: PositiveFloat = 1.0
    mondiali: PositiveFloat = 1.0
    default_unmapped: PositiveFloat = 0.6


class ExtraTime(BaseModel):
    model_config = _StrictModel
    lambda_factor: PositiveFloat = 1.0 / 3.0
    mu_factor: PositiveFloat = 1.0 / 3.0


class PenaltyShootout(BaseModel):
    model_config = _StrictModel
    base_prob_winner: Probability = 0.50
    edge_to_favorite: Annotated[float, Field(ge=0.0, le=0.5)] = 0.0


class Kelly(BaseModel):
    model_config = _StrictModel
    fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25


class Outputs(BaseModel):
    model_config = _StrictModel
    enable_output_a_calibrated: bool = True
    enable_output_b_pure_model: bool = True


class TimeDecay(BaseModel):
    model_config = _StrictModel
    half_life_years: PositiveFloat = 2.0
    half_life_years_grid: list[PositiveFloat] = Field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    )


class MarketBlending(BaseModel):
    model_config = _StrictModel
    w_default: Probability = 0.7
    w_grid: list[Probability] = Field(default_factory=lambda: [0.3, 0.5, 0.7, 0.85, 0.95])


class Betting(BaseModel):
    model_config = _StrictModel
    min_edge_threshold: Probability = 0.03
    min_edge_threshold_grid: list[Probability] = Field(
        default_factory=lambda: [0.02, 0.03, 0.04, 0.05]
    )


class HostAdvantage2026(BaseModel):
    model_config = _StrictModel
    host_countries: list[str] = Field(default_factory=lambda: ["USA", "Canada", "Mexico"])
    gamma_full_at_home: float = 1.0
    gamma_reduced_co_host: float = 0.5
    gamma_neutral: float = 0.0


class Paths(BaseModel):
    model_config = _StrictModel
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    models: Path = Path("data/models")
    notebooks: Path = Path("notebooks")


class AppConfig(BaseModel):
    """Configurazione completa del progetto."""

    model_config = _StrictModel
    perimeter: PerimeterConfig = Field(default_factory=PerimeterConfig)
    goals: GoalsConfig = Field(default_factory=GoalsConfig)
    competition_weights: CompetitionWeights = Field(default_factory=CompetitionWeights)
    include_friendlies: bool = True
    extra_time: ExtraTime = Field(default_factory=ExtraTime)
    penalty_shootout: PenaltyShootout = Field(default_factory=PenaltyShootout)
    kelly: Kelly = Field(default_factory=Kelly)
    outputs: Outputs = Field(default_factory=Outputs)
    time_decay: TimeDecay = Field(default_factory=TimeDecay)
    market_blending: MarketBlending = Field(default_factory=MarketBlending)
    betting: Betting = Field(default_factory=Betting)
    host_advantage_2026: HostAdvantage2026 = Field(default_factory=HostAdvantage2026)
    paths: Paths = Field(default_factory=Paths)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path | str | None = None) -> AppConfig:
    """Carica e valida la configurazione da YAML.

    Se `path` è None usa `config.yaml` alla radice del repo.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)


__all__ = ["AppConfig", "load_config", "DEFAULT_CONFIG_PATH"]
