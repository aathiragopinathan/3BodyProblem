# Three-Body NPE - Handoff Notes

The current final pipeline now lives under `checkpoints_report/`. This version keeps the key inverse-problem fix from the earlier draft, namely that observations start strictly after `t = 0`, and it upgrades the corrected smoke run to a stronger 20k-sample training run plus larger held-out diagnostics.

## Setup

```
pip install bayesflow tensorflow
pip install "keras<3.13"
```

Two gotchas we hit, save yourself the time:

1. If you're on Apple silicon and get an MPS/torch error even after installing tensorflow, your shell might have `KERAS_BACKEND` set to torch from a previous attempt. Force it at the top of any script:
```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```
2. `pip install bayesflow` grabs the newest Keras by default, which currently has a bug with BayesFlow's networks (`units=None` error). The `keras<3.13` pin above avoids it.

## Files

**threebody.py** - the physics engine. Integrates the 3-body equations (DOP853), handles gravitational softening, detects collisions/ejections, checks energy conservation. Everything else depends on this.

**priors.py** - defines theta as the 8 free parameters (positions/velocities of bodies 1 and 2; body 3 is derived from center-of-mass and momentum constraints). Also does rejection sampling for bounded initial conditions, and estimates the Lyapunov timescale.

**observation.py** - adds observation noise to a clean trajectory to produce y. `simulate_observation()` is the full theta -> y pipeline.

**project_config.py** - single source of truth for masses, noise scales, solver tolerances, and the exact observation-time grid used everywhere.

**generate_dataset.py / train_npe.py** - regenerate the strict post-`t=0` dataset and retrain the BayesFlow model. These now default to `checkpoints_report/`.

**infer_posterior.py** - load the trained model, generate or read one observation, sample the posterior, and save a summary figure.

**evaluate_posterior.py** - runs the compulsory diagnostics: held-out recovery, SBC rank histograms, and posterior predictive checks.

**checkpoints_report/npe_threebody.keras** - the current stronger corrected network. Load it with:
```python
from posterior_utils import load_approximator
approximator = load_approximator("checkpoints_report/npe_threebody.keras")
```

**checkpoints_report/training_data.npz** - the corrected 20k-sample `(theta, y)` dataset used for the stronger report-scale training run.

**checkpoints_report/config.json** - T, K, masses, noise scales, prior ranges, and the exact `observation_times`. Any new observation used for inference or diagnostics must match this file.

**checkpoints_v2/** - corrected but smaller smoke-run artifacts from the earlier 5k-sample retraining pass. Useful as a benchmark, but not the main final result anymore.

**checkpoints/** - legacy artifacts from the earlier version that included `t = 0` in the observation grid. Keep them only for reference, not for the final project story.

**results_diagnostics_report/** - report-scale held-out recovery, SBC, and PPC outputs for the stronger model.

**results_inference_report/** - saved posterior examples for the stronger model, including a harder held-out case.

## Getting posterior samples

Once the model's loaded, BayesFlow's interface is:
```python
posterior = approximator.sample(num_samples=1000, conditions={"y": your_y_array})
```
`your_y_array` needs a batch dimension, so shape `(1, 20, 12)` for one observation. Output is a dict, `posterior["theta"]` will be shape `(1, 1000, 8)`.

Build test observations the same way training data was made: draw a theta with `priors.py`, simulate with `threebody.py`, add noise with `observation.py`, using the settings from `checkpoints_report/config.json`.

Convenience commands:
```bash
python3 infer_posterior.py --config checkpoints_report/config.json --model checkpoints_report/npe_threebody.keras
python3 evaluate_posterior.py --config checkpoints_report/config.json --model checkpoints_report/npe_threebody.keras
```

## Current status

The compulsory implementation pipeline is complete, and the report-scale follow-up run is complete too:
- corrected dataset regenerated with strict post-`t=0` observations
- stronger 20k-sample dataset and trained model saved in `checkpoints_report/`
- larger held-out diagnostics saved in `results_diagnostics_report/`
- harder-case posterior example saved in `results_inference_report/`

What is still optional:
- extend the whole pipeline from 2D to 3D
- push experiments beyond the current report-scale run if you want even stronger final figures

For the full professor-vs-done checklist and the scientific interpretation, see `PROJECT_STATUS_REPORT.md`.

Ping me if anything about the simulator/data side is unclear, happy to walk through it.
