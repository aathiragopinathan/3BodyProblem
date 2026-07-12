# Three-Body NPE - Handoff Notes

Hey team, here's where things stand. Simulator and data pipeline are done, network is trained. This doc should get you set up and moving on diagnostics/inference without having to reverse-engineer everything.

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

**generate_dataset.py / train_npe.py** - how the trained model was actually produced. You probably won't rerun these, but they're the record of exactly what settings were used (prior ranges, noise scales, network architecture, training budget). Useful if you need to retrain or extend later.

**checkpoints/npe_threebody.keras** - the trained network. Load it with:
```python
import keras
approximator = keras.saving.load_model("checkpoints/npe_threebody.keras")
```

**checkpoints/training_data.npz** - the (theta, y) pairs used for training. Has `theta` and `y` arrays if you want to inspect or reuse them instead of regenerating.

**checkpoints/config.json** - T, K, masses, noise scales, prior ranges. Important: any new observation you build for testing/diagnostics has to use these exact same settings, otherwise the model's output is meaningless. Treat this as the source of truth.

## Getting posterior samples

Once the model's loaded, BayesFlow's interface is:
```python
posterior = approximator.sample(num_samples=1000, conditions={"y": your_y_array})
```
`your_y_array` needs a batch dimension, so shape `(1, 20, 12)` for one observation. Output is a dict, `posterior["theta"]` will be shape `(1, 1000, 8)`.

Build test observations the same way training data was made: draw a theta with `priors.py`, simulate with `threebody.py`, add noise with `observation.py`, using the settings from `config.json`.

## What's next

This is basically Workstream D at this point:
- SBC: run this on many simulated test cases, check the rank histograms are flat (calibration check)
- Recovery: does the posterior mean land near the true theta on held-out simulated data
- Posterior predictive checks
- The interesting part: characterizing when/why the posterior comes out wide or multimodal, and making the case that's the correct answer given the chaos, not a failure

Ping me if anything about the simulator/data side is unclear, happy to walk through it.

## Current status

The core project pipeline is now complete. The simulator is set up in the 2D center-of-mass frame with distinct fixed masses, gravitational softening, bounded-prior rejection, and strict post-`t = 0` observations so that the model is not given the initial state directly. A corrected training dataset was generated from this setup and used to train a BayesFlow neural posterior estimator to infer the initial conditions from noisy trajectory observations.

The evaluation pipeline is also in place. We ran held-out posterior diagnostics using recovery plots, SBC rank histograms, and posterior predictive checks. In the stronger final run, the model was trained on a corrected 20k-sample dataset and evaluated on 50 held-out cases with 1000 posterior samples per case. Overall, the results are reasonably well calibrated, and the harder cases show the expected behavior for a chaotic system: the posterior can remain broad, and that is a scientifically meaningful result rather than a failure of the method.

## Folders

- `checkpoints/` - legacy training artifacts from the older pipeline that still included `t = 0` in the observation grid, so these are kept only for reference.
- `checkpoints_v2/` - corrected training artifacts from the first fixed retraining pass after removing the `t = 0` leakage.
- `checkpoints_report/` - final stronger training artifacts from the larger corrected report-scale run, and this is the main folder to use now.
- `checkpoints_v2_smoke/` - small smoke-test artifacts used to quickly verify that the corrected pipeline runs end to end.
- `checkpoints_v2_smoke_parallel/` - smoke-test artifacts generated through the parallel dataset-generation path.
- `results_inference_v2/` - posterior inference outputs produced with the corrected smaller `v2` model.
- `results_diagnostics_v2/` - held-out diagnostic outputs from the corrected smaller `v2` model.
- `results_inference_report/` - posterior inference outputs produced with the stronger final report-scale model.
- `results_diagnostics_report/` - final larger diagnostic outputs from the stronger report-scale model.