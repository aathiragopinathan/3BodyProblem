"""
Pre-generate a large, fixed (theta, y) training dataset for the strict
post-t=0 observation setup.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from observation import add_noise, flatten_observation
from priors import is_bounded, theta_to_state
from project_config import DEFAULT_ARTIFACT_DIR, ProjectConfig, default_config
from threebody import simulate


def _draw_one(seed, config_dict):
    """Draw one bounded theta, simulate, and add noise in a worker process."""
    cfg = ProjectConfig(**config_dict)
    rng = np.random.default_rng(seed)
    masses = np.asarray(cfg.masses, dtype=float)
    for _ in range(200):
        theta = np.concatenate([
            rng.uniform(-cfg.pos_scale, cfg.pos_scale, size=4),
            rng.uniform(-cfg.vel_scale, cfg.vel_scale, size=4),
        ])
        pos, vel = theta_to_state(theta, masses)
        bounded, regime = is_bounded(
            pos,
            vel,
            masses,
            T=cfg.T,
            eps=cfg.eps,
            rtol=cfg.rtol,
            atol=cfg.atol,
            n_check=20,
        )
        if not bounded:
            continue

        result = simulate(
            pos,
            vel,
            masses,
            (0.0, cfg.T),
            cfg.t_obs,
            eps=cfg.eps,
            rtol=cfg.rtol,
            atol=cfg.atol,
            check_regime=False,
        )
        if len(result["t"]) < cfg.K:
            continue

        pos_noisy, vel_noisy = add_noise(
            result["pos"], result["vel"], cfg.sigma_x, cfg.sigma_v, rng
        )
        y = flatten_observation(pos_noisy, vel_noisy)
        return theta.astype(np.float32), y.astype(np.float32)
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional existing JSON config to reuse.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR),
                        help="Directory to write config and dataset into.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override the number of training pairs to generate.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of worker processes.")
    return parser.parse_args()


def _collect_results_serial(cfg: ProjectConfig):
    thetas, ys = [], []
    seeds = np.random.SeedSequence().spawn(cfg.n_samples)
    config_dict = cfg.to_dict()
    for i, seed in enumerate(seeds, start=1):
        result = _draw_one(seed, config_dict)
        if result is not None:
            theta, y = result
            thetas.append(theta)
            ys.append(y)
        if i % 2000 == 0:
            print(f"  ...{i}/{cfg.n_samples} done, {len(thetas)} usable")
    return np.asarray(thetas), np.asarray(ys)


def _collect_results_parallel(cfg: ProjectConfig, workers: int):
    thetas, ys = [], []
    seeds = np.random.SeedSequence().spawn(cfg.n_samples)
    config_dict = cfg.to_dict()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_draw_one, seed, config_dict) for seed in seeds]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                theta, y = result
                thetas.append(theta)
                ys.append(y)

            if i % 2000 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (cfg.n_samples - i) / rate
                print(
                    f"  ...{i}/{cfg.n_samples} done, {len(thetas)} usable, "
                    f"{elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining"
                )
    return np.asarray(thetas), np.asarray(ys)


def main():
    args = parse_args()
    cfg = ProjectConfig.from_json(args.config) if args.config else default_config()
    if args.n_samples is not None:
        cfg.n_samples = int(args.n_samples)
    output_dir = args.output_dir

    print(
        f"Generating {cfg.n_samples} strict post-t0 (theta, y) pairs "
        f"using {args.workers} worker processes..."
    )
    print(
        f"Observation grid starts at t={cfg.t_obs[0]:.3f} and ends at t={cfg.t_obs[-1]:.3f}."
    )

    t0 = time.time()
    if args.workers is None or args.workers <= 1:
        print("Using serial dataset generation.")
        thetas, ys = _collect_results_serial(cfg)
    else:
        try:
            thetas, ys = _collect_results_parallel(cfg, args.workers)
        except PermissionError:
            print("Parallel workers are not permitted here; falling back to serial generation.")
            thetas, ys = _collect_results_serial(cfg)

    print(f"\nDone. {len(thetas)} usable pairs in {time.time() - t0:.1f}s")
    print(f"thetas shape: {thetas.shape}, ys shape: {ys.shape}")

    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "training_data.npz"), theta=thetas, y=ys)

    cfg.n_samples = int(len(thetas))
    cfg.save_json(os.path.join(output_dir, "config.json"))

    print(f"\nSaved {output_dir}/training_data.npz and {output_dir}/config.json")
    print("Next: run train_npe.py -- it will detect this file and train offline.")


if __name__ == "__main__":
    main()
