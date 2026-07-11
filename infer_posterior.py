"""
Infer the posterior for either a synthetic test observation or a user-supplied
observation array and save a quick diagnostic plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from posterior_utils import (
    draw_reference_case,
    load_approximator,
    load_config,
    posterior_mean_and_intervals,
    sample_posterior_theta,
)
from project_config import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH


THETA_LABELS = ["x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--observation-npy", type=str, default=None,
                        help="Optional path to a .npy or .npz file containing y with shape (K, 12).")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=str, default="results_inference")
    return parser.parse_args()


def load_observation(path: str):
    observation_path = Path(path)
    if observation_path.suffix == ".npz":
        data = np.load(observation_path)
        if "y" not in data:
            raise ValueError(f"{observation_path} does not contain a 'y' array.")
        return np.asarray(data["y"])
    return np.asarray(np.load(observation_path))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    approximator = load_approximator(args.model)
    rng = np.random.default_rng(args.seed)

    theta_true = None
    if args.observation_npy:
        y = load_observation(args.observation_npy)
    else:
        theta_true, y, clean = draw_reference_case(cfg, rng)
        np.savez(out_dir / "synthetic_case.npz", theta_true=theta_true, y=y, clean=clean)

    if y.shape != (cfg.K, 12):
        raise ValueError(f"Expected y shape {(cfg.K, 12)}, got {y.shape}.")

    theta_samples = sample_posterior_theta(approximator, y, args.num_samples)
    mean, lower, upper = posterior_mean_and_intervals(theta_samples, level=0.9)

    np.savez(out_dir / "posterior_samples.npz", theta=theta_samples, y=y)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        ax.hist(theta_samples[:, j], bins=40, color="#1f77b4", alpha=0.85)
        ax.axvline(mean[j], color="black", lw=1.5, label="posterior mean")
        ax.axvspan(lower[j], upper[j], color="#ffcc80", alpha=0.5, label="90% CI")
        if theta_true is not None:
            ax.axvline(theta_true[j], color="#d62728", lw=1.5, ls="--", label="true theta")
        ax.set_title(THETA_LABELS[j])
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Posterior over initial-condition parameters")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out_dir / "posterior_summary.png", dpi=120)

    print("Posterior mean:")
    print(mean)
    print("\nPosterior 90% intervals:")
    print(np.stack([lower, upper], axis=1))
    if theta_true is not None:
        print("\nTrue theta:")
        print(theta_true)
        print("\nAbsolute posterior-mean error:")
        print(np.abs(mean - theta_true))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
