"""
Shared helpers for posterior inference and diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from observation import simulate_observation
from priors import sample_prior_bounded
from project_config import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH, ProjectConfig


def load_approximator(model_path: str | Path = DEFAULT_MODEL_PATH):
    try:
        import keras
    except ImportError as exc:
        raise ImportError(
            "Keras is required to load the trained BayesFlow approximator."
        ) from exc
    return keras.saving.load_model(model_path)


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    return ProjectConfig.from_json(config_path)


def draw_reference_case(cfg: ProjectConfig, rng: np.random.Generator):
    masses = np.asarray(cfg.masses, dtype=float)
    thetas, _ = sample_prior_bounded(
        rng,
        n=1,
        masses=masses,
        T=cfg.T,
        pos_scale=cfg.pos_scale,
        vel_scale=cfg.vel_scale,
        eps=cfg.eps,
    )
    theta = thetas[0]
    y, clean = simulate_observation(
        theta,
        masses,
        cfg.T,
        cfg.K,
        cfg.sigma_x,
        cfg.sigma_v,
        rng,
        eps=cfg.eps,
        t_obs=cfg.t_obs,
        return_clean=True,
    )
    return theta, y, clean


def simulate_from_theta(theta, cfg: ProjectConfig, rng: np.random.Generator, return_clean=False):
    masses = np.asarray(cfg.masses, dtype=float)
    return simulate_observation(
        theta,
        masses,
        cfg.T,
        cfg.K,
        cfg.sigma_x,
        cfg.sigma_v,
        rng,
        eps=cfg.eps,
        t_obs=cfg.t_obs,
        return_clean=return_clean,
    )


def sample_posterior_theta(approximator, y, num_samples: int) -> np.ndarray:
    y_batch = np.asarray(y, dtype=np.float32)[None, ...]
    posterior = approximator.sample(num_samples=num_samples, conditions={"y": y_batch})
    theta_samples = np.asarray(posterior["theta"])
    if theta_samples.ndim != 3 or theta_samples.shape[0] != 1:
        raise ValueError(
            f"Expected posterior['theta'] shape (1, S, 8), got {theta_samples.shape}."
        )
    return theta_samples[0]


def posterior_mean_and_intervals(theta_samples: np.ndarray, level: float = 0.9):
    alpha = 0.5 * (1.0 - level)
    lower = np.quantile(theta_samples, alpha, axis=0)
    upper = np.quantile(theta_samples, 1.0 - alpha, axis=0)
    mean = theta_samples.mean(axis=0)
    return mean, lower, upper


def sbc_ranks(theta_true: np.ndarray, theta_samples: np.ndarray) -> np.ndarray:
    return np.sum(theta_samples < theta_true[None, :], axis=0)
