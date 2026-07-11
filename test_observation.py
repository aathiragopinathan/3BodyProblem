import numpy as np
from threebody import default_masses
from priors import sample_prior_bounded
from observation import (make_observation_times, add_noise, flatten_observation,
                          simulate_observation, generate_training_pairs)

m = default_masses((1, 2, 3))
rng = np.random.default_rng(42)

print("=== TEST 1: noise has the claimed statistical properties ===")
# Take a clean trajectory, add noise MANY times, check recovered sigmas
thetas, _ = sample_prior_bounded(rng, n=1, masses=m, T=5.0)
theta = thetas[0]
y_clean_pair = simulate_observation(theta, m, T=5.0, K=20, sigma_x=0.0, sigma_v=0.0,
                                     rng=rng, return_clean=True)
y_zero_noise, clean = y_clean_pair
print(f"  with sigma=0, noisy == clean? {np.allclose(y_zero_noise, clean)}")

SX, SV = 0.05, 0.02
n_rep = 4000
pos_res, vel_res = [], []
for _ in range(n_rep):
    y = simulate_observation(theta, m, T=5.0, K=20, sigma_x=SX, sigma_v=SV, rng=rng)
    resid = y - clean
    pos_res.append(resid[:, :6])   # first 6 cols = positions
    vel_res.append(resid[:, 6:])   # last 6 cols = velocities
pos_res = np.array(pos_res); vel_res = np.array(vel_res)

print(f"  position residual: mean={pos_res.mean():+.5f} (want 0), std={pos_res.std():.5f} (want {SX})")
print(f"  velocity residual: mean={vel_res.mean():+.5f} (want 0), std={vel_res.std():.5f} (want {SV})")

print("\n=== TEST 2: noise is independent across timesteps (no correlation) ===")
# correlation between residual at t_k and t_{k+1}, for one coordinate
c = np.corrcoef(pos_res[:, 0, 0], pos_res[:, 1, 0])[0, 1]
print(f"  corr(residual at t_0, residual at t_1) = {c:+.4f} (want ~0)")

print("\n=== TEST 3: shapes are right ===")
y = simulate_observation(theta, m, T=5.0, K=20, sigma_x=SX, sigma_v=SV, rng=rng)
print(f"  y shape for K=20: {y.shape}  (want (20, 12))")
t_obs = make_observation_times(5.0, 20)
print(f"  observation times: {t_obs[0]:.2f} ... {t_obs[-1]:.2f}, count={len(t_obs)}")
print(f"  starts after t=0? {t_obs[0] > 0.0}")

print("\n=== TEST 4: full training-pair generation ===")
rng2 = np.random.default_rng(7)
thetas, ys = generate_training_pairs(n=25, masses=m, T=5.0, K=20,
                                      sigma_x=SX, sigma_v=SV, rng=rng2, verbose=True)
print(f"  thetas shape: {thetas.shape}  (want (25, 8))")
print(f"  ys shape:     {ys.shape}      (want (25, 20, 12))")
print(f"  any NaNs? thetas={np.isnan(thetas).any()}, ys={np.isnan(ys).any()}")

print("\n=== TEST 5: does noise actually matter? (signal vs noise scale) ===")
sig = np.std(clean[:, :6])
print(f"  clean position spread: {sig:.4f}")
print(f"  position noise sigma : {SX:.4f}   (noise/signal = {SX/sig:.1%})")
sigv = np.std(clean[:, 6:])
print(f"  clean velocity spread: {sigv:.4f}")
print(f"  velocity noise sigma : {SV:.4f}   (noise/signal = {SV/sigv:.1%})")
