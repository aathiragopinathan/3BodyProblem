import numpy as np
from threebody import default_masses, energy, simulate
from priors import (theta_to_state, state_to_theta, sample_prior_raw,
                     sample_prior_bounded, estimate_lyapunov_time)

m = default_masses((1, 2, 3))

print("=== TEST: theta <-> state conversion respects COM & momentum exactly ===")
rng = np.random.default_rng(1)
theta = sample_prior_raw(rng, 1, pos_scale=1.0, vel_scale=0.5)[0]
pos, vel = theta_to_state(theta, m)
com = (m[:, None] * pos).sum(axis=0)
p_tot = (m[:, None] * vel).sum(axis=0)
print(f"  COM position (should be ~0): {com}")
print(f"  total momentum (should be ~0): {p_tot}")
theta_back = state_to_theta(pos, vel)
print(f"  round-trip theta match: {np.allclose(theta, theta_back)}")

print("\n=== TEST: rejection sampling acceptance rate (pos_scale=1.0, vel_scale=0.5) ===")
rng = np.random.default_rng(2)
thetas, stats = sample_prior_bounded(rng, n=30, masses=m, T=5.0,
                                      pos_scale=1.0, vel_scale=0.5, verbose=True)
print(f"  accepted shape: {thetas.shape}")

print("\n=== TEST: Lyapunov timescale, figure-eight (near-periodic, should NOT look exponential) ===")
G_fig = 3.0
m_eq = np.array([1., 1., 1.]) / 3.0
p1 = [-0.97000436,  0.24308753]; p2 = [0.97000436, -0.24308753]; p3 = [0., 0.]
v3 = np.array([-0.93240737, -0.86473146]); v1 = -v3/2; v2 = -v3/2
theta_fig = state_to_theta(np.array([p1, p2, p3]), np.array([v1, v2, v3]))
# NOTE: figure-eight uses equal masses + G=3, not our project's default masses/G.
# We can't use theta_to_state's mass-based body-3 reconstruction here since it
# assumes m -> default; just pass the explicit state directly for this special case.
from threebody import simulate as sim_direct
t_eval = np.linspace(0, 6.3259, 200)
pos1, vel1 = np.array([p1, p2, p3]), np.array([v1, v2, v3])
rng2 = np.random.default_rng(0)
direction = rng2.normal(size=6); direction /= np.linalg.norm(direction)
pert = 1e-8 * direction.reshape(3, 2)
r1 = sim_direct(pos1, vel1, m_eq, (0, 6.3259), t_eval, G=G_fig, eps=0.0, check_regime=False)
r2 = sim_direct(pos1 + pert*0 + np.vstack([pert]), vel1, m_eq, (0, 6.3259), t_eval, G=G_fig, eps=0.0, check_regime=False)
delta = np.linalg.norm(r1["pos"].reshape(len(r1["t"]),-1) - r2["pos"].reshape(len(r2["t"]),-1), axis=1)
print(f"  figure-eight divergence: starts at {delta[0]:.2e}, ends at {delta[-1]:.2e} (growth factor {delta[-1]/max(delta[0],1e-300):.2f})")

print("\n=== TEST: Lyapunov timescale, generic bounded chaotic case (project masses) ===")
if len(thetas) > 0:
    result = estimate_lyapunov_time(thetas[0], m, T=8.0, n_points=200)
    print(f"  lyapunov_rate: {result['lyapunov_rate']:.4f}")
    print(f"  lyapunov_time: {result['lyapunov_time']:.4f}")
    print(f"  fit_r2:        {result['fit_r2']:.4f}")
    print(f"  fit_ok:        {result['fit_ok']}")
else:
    print("  no accepted thetas to test with -- acceptance rate too low, need to tune scales")
