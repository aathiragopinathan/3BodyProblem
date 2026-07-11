# Project Status Report

## Final status

- Compulsory implementation pipeline: complete.
- Report-scale experimentation requested after the implementation pass: complete.
- Optional extension to 3D: not implemented.

## Report-scale work completed

- Regenerated a corrected dataset with strict post-`t=0` observations and saved it in `checkpoints_report/training_data.npz`.
- Used the stronger corrected configuration in `checkpoints_report/config.json`.
- Masses are `1:2:3`, normalized to total mass `1`.
- The observation horizon is `T = 5.0`.
- The trajectory is observed at `K = 20` times.
- The observation times are `0.25, 0.50, ..., 5.00`.
- Separate noise scales are used: `sigma_x = 0.05`, `sigma_v = 0.02`.
- Retrained the BayesFlow neural posterior estimator on `20,000` corrected samples and saved the model in `checkpoints_report/npe_threebody.keras`.
- Ran a larger held-out diagnostic sweep and saved it in `results_diagnostics_report/`.
- The diagnostic sweep used `50` held-out cases.
- The diagnostic sweep used `1000` posterior samples per case.
- The diagnostic sweep used `100` posterior-predictive draws per case.
- Saved a representative harder held-out posterior example in `results_inference_report/case033/`.

## Key results

- Report-scale PPC result: mean componentwise 90% coverage `0.9228`.
- Report-scale marginal 90% coverage for `x1`: `1.00`.
- Report-scale marginal 90% coverage for `y1`: `0.94`.
- Report-scale marginal 90% coverage for `x2`: `0.96`.
- Report-scale marginal 90% coverage for `y2`: `0.98`.
- Report-scale marginal 90% coverage for `vx1`: `0.98`.
- Report-scale marginal 90% coverage for `vy1`: `0.92`.
- Report-scale marginal 90% coverage for `vx2`: `0.96`.
- Report-scale marginal 90% coverage for `vy2`: `0.94`.
- Compared with the earlier 10-case smoke run, the stronger 20k model improved both posterior accuracy and PPC calibration.
- The harder held-out example in `results_inference_report/case033/` is mainly broad in the velocity coordinates, especially `vx1` and `vy1`, which is consistent with uncertainty amplification in sensitive regimes.

## Scientific interpretation

- In this project, a sharp posterior is not the goal by itself. The system is chaotic, so many nearby initial conditions can produce trajectories that remain hard to distinguish once noise is added.
- Because of that, broad posteriors are scientifically correct whenever the observation does not uniquely identify one initial state. Overconfident narrow posteriors would be worse than wide ones.
- The report-scale diagnostics support that interpretation: the 90% intervals are mostly near nominal coverage, and the PPC coverage is close to the target level overall.
- The harder cases in the held-out sweep do not mean the method failed. They show the expected behavior of a chaotic inverse problem: some draws are much less identifiable than others.
- In the saved harder example, the posterior is mostly wide and non-Gaussian rather than sharply collapsed. That is a valid result. Multimodality is allowed by the method and may appear in even more ambiguous cases, but even when the posterior is simply wide, that width is meaningful scientific information.

## Professor expectations vs current status

- `Use the 2D planar 3-body problem first`: done.
- `Optional extension to 3D`: not done, still optional.
- `Nondimensionalize with G = 1 and total mass normalized`: done.
- `Use distinct fixed labeled masses`: done with normalized `1:2:3`.
- `Work in the center-of-mass frame with zero total momentum`: done.
- `Infer the initial conditions at t = 0 from noisy trajectory observations`: done.
- `Use BayesFlow with an amortized neural posterior estimator`: done.
- `Jointly train summary and posterior networks`: done with BayesFlow `TimeSeriesNetwork` and `CouplingFlow`.
- `Standardize network inputs`: done in the BayesFlow workflow.
- `Integrate with an adaptive high-accuracy solver`: done with `solve_ivp(..., method="DOP853")`.
- `Add gravitational softening`: done with `eps = 1e-3`.
- `Restrict the prior to bounded trajectories and reject collisions/ejections`: done.
- `Use energy conservation as a simulator trust check`: done.
- `Observe the trajectory at intermediate times, not only the endpoint`: done with `K = 20`.
- `Ensure observation times fall after t = 0 and within the informative timescale`: done with the strict post-`t=0` grid.
- `Estimate the Lyapunov timescale empirically`: done in `priors.py` and `lyapunov_sweep.py`.
- `Add separate Gaussian noise scales for positions and velocities`: done.
- `Keep body labels consistent`: done.
- `Accept wide or multimodal posteriors as the correct result in chaotic regimes`: done in both the pipeline design and the final interpretation.

## What remains pending

- No compulsory implementation items are pending anymore.
- No compulsory report-scale experimentation items are pending anymore.
- Optional item remaining: 3D extension.
- Optional item remaining: even larger training and evaluation sweeps if you want stronger figures for a final report or presentation.
