"""
Run the compulsory posterior diagnostics:
  - held-out recovery
  - simulation-based calibration (SBC)
  - posterior predictive checks (PPC)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

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
    sbc_ranks,
    simulate_from_theta,
)
from project_config import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH


THETA_LABELS = ["x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--num-cases", type=int, default=40)
    parser.add_argument("--num-posterior-samples", type=int, default=500)
    parser.add_argument("--num-ppc-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=str, default="results_diagnostics")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    approximator = load_approximator(args.model)
    rng = np.random.default_rng(args.seed)

    truths = []
    means = []
    lowers = []
    uppers = []
    ranks = []
    ppc_coverages = []

    for case_idx in range(args.num_cases):
        theta_true, y_obs, _clean = draw_reference_case(cfg, rng)
        theta_samples = sample_posterior_theta(
            approximator, y_obs, args.num_posterior_samples
        )
        mean, lower, upper = posterior_mean_and_intervals(theta_samples, level=0.9)

        truths.append(theta_true)
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)
        ranks.append(sbc_ranks(theta_true, theta_samples))

        n_ppc = min(args.num_ppc_draws, len(theta_samples))
        chosen = rng.choice(len(theta_samples), size=n_ppc, replace=False)
        predictive = []
        for idx in chosen:
            y_rep = simulate_from_theta(theta_samples[idx], cfg, rng, return_clean=False)
            if y_rep is not None:
                predictive.append(y_rep)
        predictive = np.asarray(predictive)
        if len(predictive) == 0:
            raise RuntimeError("No valid posterior-predictive draws were generated.")

        pred_lower = np.quantile(predictive, 0.05, axis=0)
        pred_upper = np.quantile(predictive, 0.95, axis=0)
        ppc_coverages.append(np.mean((y_obs >= pred_lower) & (y_obs <= pred_upper)))

        print(
            f"case {case_idx + 1:03d}/{args.num_cases}: "
            f"mean abs err={np.mean(np.abs(mean - theta_true)):.4f}, "
            f"ppc coverage={ppc_coverages[-1]:.3f}"
        )

    truths = np.asarray(truths)
    means = np.asarray(means)
    lowers = np.asarray(lowers)
    uppers = np.asarray(uppers)
    ranks = np.asarray(ranks)
    ppc_coverages = np.asarray(ppc_coverages)

    mae_per_dim = np.mean(np.abs(means - truths), axis=0)
    rmse_per_dim = np.sqrt(np.mean((means - truths) ** 2, axis=0))
    marginal_coverage = np.mean((truths >= lowers) & (truths <= uppers), axis=0)

    metrics = {
        "num_cases": int(args.num_cases),
        "num_posterior_samples": int(args.num_posterior_samples),
        "num_ppc_draws": int(args.num_ppc_draws),
        "mae_per_dim": mae_per_dim.tolist(),
        "rmse_per_dim": rmse_per_dim.tolist(),
        "marginal_90pct_coverage_per_dim": marginal_coverage.tolist(),
        "mean_ppc_componentwise_90pct_coverage": float(np.mean(ppc_coverages)),
        "std_ppc_componentwise_90pct_coverage": float(np.std(ppc_coverages)),
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.savez(
        out_dir / "diagnostics_arrays.npz",
        truths=truths,
        means=means,
        lowers=lowers,
        uppers=uppers,
        ranks=ranks,
        ppc_coverages=ppc_coverages,
    )

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        ax.hist(
            ranks[:, j],
            bins=np.arange(args.num_posterior_samples + 2) - 0.5,
            color="#1f77b4",
            alpha=0.85,
        )
        ax.set_title(f"SBC rank: {THETA_LABELS[j]}")
        ax.set_xlabel("rank")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "sbc_rank_histograms.png", dpi=120)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        ax.scatter(truths[:, j], means[:, j], s=22, alpha=0.8, color="#d62728")
        lo = min(truths[:, j].min(), means[:, j].min())
        hi = max(truths[:, j].max(), means[:, j].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"Recovery: {THETA_LABELS[j]}")
        ax.set_xlabel("true")
        ax.set_ylabel("posterior mean")
    plt.tight_layout()
    plt.savefig(out_dir / "recovery_scatter.png", dpi=120)

    plt.figure(figsize=(6, 4))
    plt.hist(ppc_coverages, bins=15, color="#2ca02c", alpha=0.85)
    plt.axvline(np.mean(ppc_coverages), color="black", lw=1.5, ls="--")
    plt.xlabel("componentwise PPC interval coverage")
    plt.ylabel("count")
    plt.title("Posterior predictive coverage across held-out cases")
    plt.tight_layout()
    plt.savefig(out_dir / "ppc_coverage_histogram.png", dpi=120)

    print("\nRecovery / SBC / PPC summary")
    for label, mae, rmse, cov in zip(THETA_LABELS, mae_per_dim, rmse_per_dim, marginal_coverage):
        print(f"{label:>3s}: MAE={mae:.4f}, RMSE={rmse:.4f}, 90% coverage={cov:.3f}")
    print(f"\nMean PPC componentwise 90% coverage: {np.mean(ppc_coverages):.3f}")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
