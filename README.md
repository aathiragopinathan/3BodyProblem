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