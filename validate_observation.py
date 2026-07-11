import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threebody import default_masses, simulate
from priors import sample_prior_bounded, theta_to_state
from observation import simulate_observation, make_observation_times

m = default_masses((1, 2, 3))
rng = np.random.default_rng(3)
thetas, _ = sample_prior_bounded(rng, n=1, masses=m, T=5.0)
theta = thetas[0]

T, K = 5.0, 20
SX, SV = 0.05, 0.02
y, clean = simulate_observation(theta, m, T, K, SX, SV, rng, return_clean=True)
t_obs = make_observation_times(T, K)

# dense underlying trajectory for reference
pos0, vel0 = theta_to_state(theta, m)
dense = simulate(pos0, vel0, m, (0, T), np.linspace(0, T, 500), check_regime=False)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
colors = ["#1f77b4", "#d62728", "#2ca02c"]

# Panel 1: trajectory in space + noisy observations
for b in range(3):
    ax[0].plot(dense["pos"][:, b, 0], dense["pos"][:, b, 1], color=colors[b],
               lw=1, alpha=0.6, label=f"body {b+1} (true path)")
    ax[0].scatter(y[:, 2*b], y[:, 2*b+1], color=colors[b], s=28,
                  edgecolor="k", linewidth=0.4, zorder=3)
ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
ax[0].set_title("What the network sees:\nnoisy observations (dots) vs true path (lines)")
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")

# Panel 2: x-coordinate of each body vs time, clean vs noisy
for b in range(3):
    ax[1].plot(t_obs, clean[:, 2*b], color=colors[b], lw=1.5, label=f"body {b+1} clean")
    ax[1].scatter(t_obs, y[:, 2*b], color=colors[b], s=20, alpha=0.8)
ax[1].set_xlabel("time"); ax[1].set_ylabel("x position")
ax[1].set_title(f"Position: clean vs noisy (σx={SX})"); ax[1].legend(fontsize=8)

# Panel 3: velocity, clean vs noisy
for b in range(3):
    ax[2].plot(t_obs, clean[:, 6+2*b], color=colors[b], lw=1.5, label=f"body {b+1} clean")
    ax[2].scatter(t_obs, y[:, 6+2*b], color=colors[b], s=20, alpha=0.8)
ax[2].set_xlabel("time"); ax[2].set_ylabel("x velocity")
ax[2].set_title(f"Velocity: clean vs noisy (σv={SV})"); ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("validation_observation.png", dpi=110)
print("saved validation_observation.png")
print(f"y shape = {y.shape}  ->  {y.size} numbers per observation (= K x 12 = {K}x12)")
