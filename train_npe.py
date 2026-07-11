"""
Workstream C: BayesFlow NPE training setup for the strict post-t=0 dataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import bayesflow as bf
import numpy as np

from observation import add_noise, flatten_observation
from priors import is_bounded, theta_to_state
from project_config import DEFAULT_ARTIFACT_DIR, ProjectConfig, default_config
from threebody import simulate


def build_workflow(cfg: ProjectConfig):
    masses = np.asarray(cfg.masses, dtype=float)
    t_eval = cfg.t_obs

    def prior(max_tries=200):
        for _ in range(max_tries):
            theta = np.concatenate([
                np.random.uniform(-cfg.pos_scale, cfg.pos_scale, size=4),
                np.random.uniform(-cfg.vel_scale, cfg.vel_scale, size=4),
            ]).astype(np.float64)
            pos, vel = theta_to_state(theta, masses)
            bounded, regime = is_bounded(pos, vel, masses, T=cfg.T, eps=cfg.eps)
            if bounded:
                return {"theta": theta.astype(np.float32)}
        raise RuntimeError(
            f"Could not find a bounded theta in {max_tries} tries; "
            "pos_scale/vel_scale may be too aggressive."
        )

    def likelihood(theta):
        theta64 = np.asarray(theta, dtype=np.float64)
        pos0, vel0 = theta_to_state(theta64, masses)
        result = simulate(
            pos0, vel0, masses, (0.0, cfg.T), t_eval, eps=cfg.eps, check_regime=False
        )
        if len(result["t"]) < cfg.K:
            raise RuntimeError(
                "Simulated trajectory shorter than K even though prior() "
                "checked boundedness. Check config consistency."
            )

        pos_noisy, vel_noisy = add_noise(
            result["pos"], result["vel"], cfg.sigma_x, cfg.sigma_v, np.random.default_rng()
        )
        y = flatten_observation(pos_noisy, vel_noisy)
        return {"y": y.astype(np.float32)}

    simulator = bf.make_simulator([prior, likelihood])
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .rename("theta", "inference_variables")
        .rename("y", "summary_variables")
    )

    summary_network = bf.networks.TimeSeriesNetwork()
    inference_network = bf.networks.CouplingFlow()

    return bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
        standardize=["inference_variables", "summary_variables"],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the JSON config generated alongside the dataset.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a pre-generated training_data.npz file.")
    parser.add_argument("--model-out", type=str, default=None,
                        help="Path to save the trained BayesFlow approximator.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--online-batches-per-epoch", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dir = Path(DEFAULT_ARTIFACT_DIR)
    config_path = Path(args.config) if args.config else artifact_dir / "config.json"
    data_path = Path(args.data) if args.data else artifact_dir / "training_data.npz"
    model_path = Path(args.model_out) if args.model_out else artifact_dir / "npe_threebody.keras"

    cfg = ProjectConfig.from_json(config_path) if config_path.exists() else default_config()
    workflow = build_workflow(cfg)

    if data_path.exists():
        print(f"Loading pre-generated dataset from {data_path} ...")
        data = np.load(data_path)
        training_data = {"theta": data["theta"], "y": data["y"]}
        print({k: v.shape for k, v in training_data.items()})

        print("\nTraining OFFLINE (reusing the fixed dataset across many epochs)...")
        workflow.fit_offline(training_data, epochs=args.epochs, batch_size=args.batch_size)

        print("\nSaving trained approximator ...")
        model_path.parent.mkdir(exist_ok=True)
        workflow.approximator.save(filepath=model_path)
        print(f"Saved to {model_path}")
        print(f"Config: {config_path}")
        print(f"Dataset: {data_path}")
    else:
        print(f"No offline dataset found at {data_path}.")
        print("Run `python3 generate_dataset.py` first for real training.")
        print("Falling back to a short ONLINE sanity check only:\n")
        batch = workflow.simulator.sample(4)
        print({k: np.asarray(v).shape for k, v in batch.items()})
        workflow.fit_online(
            epochs=3,
            batch_size=32,
            num_batches_per_epoch=args.online_batches_per_epoch,
        )
        print("\nThis was just a sanity check, not real training.")


if __name__ == "__main__":
    main()
