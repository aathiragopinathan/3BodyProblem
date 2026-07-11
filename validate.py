import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threebody import simulate, default_masses, to_com_frame, energy

# ---------------------------------------------------------------
# TEST 1: figure-eight (equal masses) -> validates the dynamics
# ---------------------------------------------------------------
m_eq = np.array([1.0, 1.0, 1.0]) / 3.0     # equal, normalized to sum 1
# Standard figure-eight ICs are given for G=1, m=1 each. We rescale masses to
# sum to 1, so also rescale G*m: with m_i = 1/3 instead of 1, accelerations
# shrink by 3x. Easiest: keep the classic ICs but set G so that G*m = 1.
# classic uses G=1, m=1 => G*m=1. With m=1/3 we need G=3.
G_fig = 3.0
p1 = [-0.97000436,  0.24308753]
p2 = [ 0.97000436, -0.24308753]
p3 = [ 0.0,          0.0]
v3 = np.array([-0.93240737, -0.86473146])
v1 = -v3 / 2
v2 = -v3 / 2
pos0 = np.array([p1, p2, p3])
vel0 = np.array([v1, v2, v3])

T_period = 6.3259               # period of the classic figure-eight
t_eval = np.linspace(0, T_period, 600)
r_fig = simulate(pos0, vel0, m_eq, (0, T_period), t_eval,
                 G=G_fig, eps=0.0, enforce_com=True)

print("TEST 1 (figure-eight, equal masses)")
print(f"  integrator success : {r_fig['success']}")
print(f"  max |dE/E|         : {r_fig['max_rel_dE']:.3e}")
# closure: does it return near the start after one period?
start = np.concatenate([r_fig['pos'][0].ravel(), r_fig['vel'][0].ravel()])
end   = np.concatenate([r_fig['pos'][-1].ravel(), r_fig['vel'][-1].ravel()])
print(f"  orbit closure err  : {np.linalg.norm(end-start):.3e}")

# ---------------------------------------------------------------
# TEST 2: project setup -> distinct masses 1:2:3, softening, COM
# ---------------------------------------------------------------
m = default_masses((1, 2, 3))
print("\nTEST 2 (project setup: masses 1:2:3, softening on)")
print(f"  masses (sum={m.sum():.3f}): {m}")

rng = np.random.default_rng(0)
pos0b = rng.uniform(-1, 1, size=(3, 2))
vel0b = rng.uniform(-0.5, 0.5, size=(3, 2))

# verify COM reduction really zeroes COM position & momentum
pc, vc = to_com_frame(pos0b, vel0b, m)
print(f"  COM position after reduction : {np.abs((m[:,None]*pc).sum(0)).max():.2e}")
print(f"  total momentum after reduction: {np.abs((m[:,None]*vc).sum(0)).max():.2e}")

T = 8.0
t_eval2 = np.linspace(0, T, 500)
r = simulate(pos0b, vel0b, m, (0, T), t_eval2, eps=1e-3)
print(f"  integrator success : {r['success']}")
print(f"  E0                 : {r['E0']:.6f}")
print(f"  max |dE/E|         : {r['max_rel_dE']:.3e}")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 5))

# figure-eight
colors = ["#1f77b4", "#d62728", "#2ca02c"]
for b in range(3):
    ax[0].plot(r_fig["pos"][:, b, 0], r_fig["pos"][:, b, 1],
               color=colors[b], lw=1.5, label=f"body {b+1}")
ax[0].set_title("TEST 1: figure-eight (dynamics check)")
ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")

# project trajectory
for b in range(3):
    ax[1].plot(r["pos"][:, b, 0], r["pos"][:, b, 1],
               color=colors[b], lw=1.2, label=f"body {b+1} (m={m[b]:.2f})")
    ax[1].plot(r["pos"][0, b, 0], r["pos"][0, b, 1], "o", color=colors[b])
ax[1].set_title("TEST 2: distinct masses 1:2:3")
ax[1].set_aspect("equal"); ax[1].legend(); ax[1].set_xlabel("x"); ax[1].set_ylabel("y")

# energy drift
ax[2].plot(r_fig["t"], np.abs(r_fig["E_series"]-r_fig["E0"])/abs(r_fig["E0"]),
           label="figure-eight")
ax[2].plot(r["t"], np.abs(r["E_series"]-r["E0"])/abs(r["E0"]),
           label="project setup")
ax[2].set_yscale("log"); ax[2].set_title("Energy conservation self-check")
ax[2].set_xlabel("time"); ax[2].set_ylabel("|E(t) - E0| / |E0|"); ax[2].legend()

plt.tight_layout()
plt.savefig("validation.png", dpi=110)
print("\nsaved validation.png")
