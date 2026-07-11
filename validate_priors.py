import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threebody import default_masses, simulate
from priors import theta_to_state, sample_prior_bounded, estimate_lyapunov_time

m = default_masses((1, 2, 3))

# Chaotic case: use an accepted bounded theta
rng = np.random.default_rng(2)
thetas, stats = sample_prior_bounded(rng, n=5, masses=m, T=8.0,
                                      pos_scale=1.0, vel_scale=0.5)
theta_chaotic = thetas[0]
res_chaotic = estimate_lyapunov_time(theta_chaotic, m, T=8.0, n_points=200)

# Figure-eight case (periodic, near-zero chaos) for contrast
G_fig = 3.0
m_eq = np.array([1., 1., 1.]) / 3.0
p1 = [-0.97000436, 0.24308753]; p2 = [0.97000436, -0.24308753]; p3 = [0., 0.]
v3 = np.array([-0.93240737, -0.86473146]); v1 = -v3/2; v2 = -v3/2
pos0, vel0 = np.array([p1, p2, p3]), np.array([v1, v2, v3])
rng2 = np.random.default_rng(0)
direction = rng2.normal(size=6); direction /= np.linalg.norm(direction)
pert = (1e-8 * direction).reshape(3, 2)
t_eval = np.linspace(0, 8, 200)
r1 = simulate(pos0, vel0, m_eq, (0, 8), t_eval, G=G_fig, eps=0.0, check_regime=False)
r2 = simulate(pos0 + pert, vel0, m_eq, (0, 8), t_eval, G=G_fig, eps=0.0, check_regime=False)
delta_fig = np.linalg.norm(
    r1["pos"].reshape(len(r1["t"]), -1) - r2["pos"].reshape(len(r2["t"]), -1), axis=1)
delta_fig = np.clip(delta_fig, 1e-300, None)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(res_chaotic["t"], res_chaotic["log_delta"], label="chaotic case (project masses)", color="#d62728")
ax[0].plot(r1["t"], np.log(delta_fig), label="figure-eight (periodic)", color="#1f77b4")
n_fit = int(0.6 * len(res_chaotic["t"]))
fit_line = res_chaotic["lyapunov_rate"] * res_chaotic["t"][:n_fit] + \
    (res_chaotic["log_delta"][:n_fit] - res_chaotic["lyapunov_rate"] * res_chaotic["t"][:n_fit]).mean()
ax[0].plot(res_chaotic["t"][:n_fit], fit_line, "k--", lw=1, label=f"fit: λ={res_chaotic['lyapunov_rate']:.2f}")
ax[0].set_xlabel("time"); ax[0].set_ylabel("ln‖Δstate‖")
ax[0].set_title("Divergence of nearby trajectories")
ax[0].legend()

# Example accepted trajectory
pos_c, vel_c = theta_to_state(theta_chaotic, m)
t_eval2 = np.linspace(0, 8, 400)
traj = simulate(pos_c, vel_c, m, (0, 8), t_eval2, check_regime=False)
colors = ["#1f77b4", "#d62728", "#2ca02c"]
for b in range(3):
    ax[1].plot(traj["pos"][:, b, 0], traj["pos"][:, b, 1], color=colors[b], lw=1, label=f"body {b+1}")
ax[1].set_aspect("equal"); ax[1].set_title("Example accepted (bounded) trajectory")
ax[1].legend(); ax[1].set_xlabel("x"); ax[1].set_ylabel("y")

plt.tight_layout()
plt.savefig("validation_priors.png", dpi=110)
print("saved validation_priors.png")
print(f"\nLyapunov time for this sample: {res_chaotic['lyapunov_time']:.3f}")
print(f"Suggested T (a few / lambda): ~{2/res_chaotic['lyapunov_rate']:.2f} to {4/res_chaotic['lyapunov_rate']:.2f}")
