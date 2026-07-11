import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threebody import default_masses
from priors import sample_prior_bounded, estimate_lyapunov_time

m = default_masses((1, 2, 3))
rng = np.random.default_rng(123)

N = 60
T_PROBE = 8.0   # generous window just for measuring lambda itself
print(f"Drawing {N} bounded initial conditions and estimating Lyapunov time for each...")
thetas, stats = sample_prior_bounded(rng, n=N, masses=m, T=T_PROBE,
                                      pos_scale=1.0, vel_scale=0.5, verbose=True)

rates, times, r2s, ok_flags = [], [], [], []
for theta in thetas:
    res = estimate_lyapunov_time(theta, m, T=T_PROBE, n_points=200)
    rates.append(res["lyapunov_rate"])
    times.append(res["lyapunov_time"])
    r2s.append(res["fit_r2"])
    ok_flags.append(res["fit_ok"])

rates = np.array(rates); times = np.array(times); r2s = np.array(r2s)
ok_flags = np.array(ok_flags)

print(f"\n{ok_flags.sum()}/{N} samples had a clean exponential-divergence fit (fit_ok=True)")
good_times = times[ok_flags & np.isfinite(times)]
print(f"\nLyapunov TIME statistics (fit_ok samples only, n={len(good_times)}):")
print(f"  min    : {good_times.min():.3f}")
print(f"  25%ile : {np.percentile(good_times, 25):.3f}")
print(f"  median : {np.median(good_times):.3f}")
print(f"  75%ile : {np.percentile(good_times, 75):.3f}")
print(f"  max    : {good_times.max():.3f}")

median_lyap = np.median(good_times)
print(f"\nSuggested T (a few Lyapunov times, using median={median_lyap:.3f}):")
for mult in [2, 3, 4, 5]:
    print(f"  {mult} x lyapunov_time = {mult*median_lyap:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].hist(good_times, bins=15, color="#1f77b4", edgecolor="k", alpha=0.8)
ax[0].axvline(median_lyap, color="red", ls="--", label=f"median = {median_lyap:.2f}")
ax[0].set_xlabel("Lyapunov time (1/λ)"); ax[0].set_ylabel("count")
ax[0].set_title("Distribution of Lyapunov timescales\nacross accepted prior samples")
ax[0].legend()

ax[1].scatter(r2s, times, c=ok_flags, cmap="coolwarm_r", edgecolor="k", s=40)
ax[1].set_xlabel("fit R² (exponential-fit quality)")
ax[1].set_ylabel("Lyapunov time")
ax[1].set_title("Fit quality vs timescale\n(blue = accepted as clean exponential fit)")

plt.tight_layout()
plt.savefig("lyapunov_sweep.png", dpi=110)
print("\nsaved lyapunov_sweep.png")
