"""
Shared project configuration for data generation, training, inference, and
diagnostics.

The important fix here is that the observation grid is stored explicitly and,
by default, starts strictly after t = 0. That keeps the inverse problem honest:
the model must infer the initial condition from later dynamics rather than from
a noisy copy of the initial state itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from observation import make_observation_times
from threebody import default_masses


DEFAULT_ARTIFACT_DIR = Path("checkpoints_report")
DEFAULT_CONFIG_PATH = DEFAULT_ARTIFACT_DIR / "config.json"
DEFAULT_DATA_PATH = DEFAULT_ARTIFACT_DIR / "training_data.npz"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "npe_threebody.keras"


@dataclass
class ProjectConfig:
    masses: list[float]
    T: float = 5.0
    K: int = 20
    sigma_x: float = 0.05
    sigma_v: float = 0.02
    pos_scale: float = 1.0
    vel_scale: float = 0.5
    eps: float = 1e-3
    rtol: float = 1e-7
    atol: float = 1e-9
    n_samples: int = 20000
    include_t0: bool = False
    observation_times: list[float] | None = None

    def __post_init__(self):
        self.masses = [float(m) for m in self.masses]
        if self.K <= 0:
            raise ValueError("K must be positive.")
        if self.T <= 0:
            raise ValueError("T must be positive.")

        if self.observation_times is None:
            self.observation_times = make_observation_times(
                self.T, self.K, include_t0=self.include_t0
            ).tolist()
        else:
            self.observation_times = [float(t) for t in self.observation_times]

        if len(self.observation_times) != self.K:
            raise ValueError(
                f"Expected {self.K} observation times, got {len(self.observation_times)}."
            )

        t_obs = np.asarray(self.observation_times, dtype=float)
        if not np.all(np.diff(t_obs) > 0):
            raise ValueError("Observation times must be strictly increasing.")
        if t_obs[-1] > self.T + 1e-12:
            raise ValueError("Observation times must satisfy t_K <= T.")
        if not self.include_t0 and t_obs[0] <= 0.0:
            raise ValueError(
                "Observation times must start strictly after t=0 for this project."
            )

    @property
    def t_obs(self) -> np.ndarray:
        return np.asarray(self.observation_times, dtype=float)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return cls(**json.load(f))


def default_config() -> ProjectConfig:
    return ProjectConfig(masses=default_masses((1, 2, 3)).tolist())
