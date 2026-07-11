"""
Workstream B: priors, rejection sampling, and Lyapunov timescale estimation.

Builds on threebody.py (the simulator core). This module is responsible for:
  1. Defining theta as the 8 truly free parameters (r1, r2, v1, v2), with
     body 3's position/velocity DERIVED from the COM-at-origin and
     zero-total-momentum constraints -- so we never sample or waste network
     capacity on the 4 dependent dimensions.
  2. Rejection sampling: draw theta from the prior, keep only initial
     conditions that stay bounded (no collision, no ejection) over [0, T].
  3. Empirically estimating the Lyapunov timescale, to inform how the
     observation window [0, T] and observation times t_1..t_K should be
     chosen (per the project brief: "several observation times should fall
     within the first Lyapunov time or so").

theta layout (length 8):
    [ x1, y1, x2, y2, vx1, vy1, vx2, vy2 ]
"""

import numpy as np
from threebody import simulate, is_bounded, default_masses, N_BODIES, DIM


# ---------------------------------------------------------------------------
# theta <-> full (pos, vel) state conversion
# ---------------------------------------------------------------------------
def theta_to_state(theta, masses):
    """
    Expand the 8 free parameters into the full 3-body (pos, vel) state,
    deriving body 3 from the COM-at-origin and zero-momentum constraints.

    theta   : (8,) = [x1,y1, x2,y2, vx1,vy1, vx2,vy2]
    masses  : (3,)

    returns : pos (3,2), vel (3,2)   -- exactly COM-centered, zero total
              momentum, by construction (not just approximately, since we
              solve for body 3 directly rather than post-hoc shifting).
    """
    theta = np.asarray(theta, dtype=float)
    r1, r2 = theta[0:2], theta[2:4]
    v1, v2 = theta[4:6], theta[6:8]
    m1, m2, m3 = masses

    # COM at origin:      m1 r1 + m2 r2 + m3 r3 = 0        => r3 = -(m1 r1 + m2 r2)/m3
    # total momentum = 0: m1 v1 + m2 v2 + m3 v3 = 0        => v3 = -(m1 v1 + m2 v2)/m3
    r3 = -(m1 * r1 + m2 * r2) / m3
    v3 = -(m1 * v1 + m2 * v2) / m3

    pos = np.array([r1, r2, r3])
    vel = np.array([v1, v2, v3])
    return pos, vel


def state_to_theta(pos, vel):
    """Inverse of theta_to_state: drop body 3 (it's fully determined by 1,2)."""
    return np.concatenate([pos[0], pos[1], vel[0], vel[1]])


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------
def sample_prior_raw(rng, n, pos_scale=1.0, vel_scale=0.5):
    """
    Draw n raw theta samples (before bounded-regime rejection).

    Uniform box priors, in nondimensional units (characteristic length = 1):
        x1,y1,x2,y2       ~ Uniform(-pos_scale, +pos_scale)
        vx1,vy1,vx2,vy2   ~ Uniform(-vel_scale, +vel_scale)

    These are placeholder ranges -- pos_scale=1 matches the characteristic
    length unit; vel_scale=0.5 is a starting guess for "roughly bound-orbit"
    speeds and should be tuned once acceptance rates are measured below.
    """
    pos_part = rng.uniform(-pos_scale, pos_scale, size=(n, 4))
    vel_part = rng.uniform(-vel_scale, vel_scale, size=(n, 4))
    return np.concatenate([pos_part, vel_part], axis=1)


# ---------------------------------------------------------------------------
# Rejection sampling
# ---------------------------------------------------------------------------
def sample_prior_bounded(rng, n, masses, T, pos_scale=1.0, vel_scale=0.5,
                          eps=1e-3, r_min=1e-2, r_max=50.0, n_check=50,
                          max_attempts_factor=50, verbose=False):
    """
    Draw n theta samples that are bounded (no collision/ejection) over [0,T].

    Uses simple rejection sampling: repeatedly draw batches from the raw
    prior, keep only the bounded ones, until n accepted samples are
    collected (or we give up after max_attempts_factor * n total draws).

    Returns
    -------
    thetas   : (n, 8) accepted parameter draws
    stats    : dict with acceptance rate and regime breakdown, useful for
               tuning pos_scale/vel_scale/T as a group
    """
    accepted = []
    regime_counts = {"bounded": 0, "collision": 0, "ejection": 0}
    n_tried = 0
    max_tries = max_attempts_factor * n
    batch = max(64, n)  # draw in batches for efficiency

    while len(accepted) < n and n_tried < max_tries:
        candidates = sample_prior_raw(rng, batch, pos_scale, vel_scale)
        for theta in candidates:
            if len(accepted) >= n or n_tried >= max_tries:
                break
            n_tried += 1
            pos, vel = theta_to_state(theta, masses)
            bounded, regime = is_bounded(pos, vel, masses, T, eps=eps,
                                         r_min=r_min, r_max=r_max, n_check=n_check)
            regime_counts[regime] += 1
            if bounded:
                accepted.append(theta)

    stats = {
        "n_accepted": len(accepted),
        "n_tried": n_tried,
        "acceptance_rate": len(accepted) / n_tried if n_tried else float("nan"),
        "regime_counts": regime_counts,
    }
    if verbose:
        print(f"[sample_prior_bounded] accepted {len(accepted)}/{n_tried} "
              f"({stats['acceptance_rate']:.1%}) | regimes: {regime_counts}")

    return np.array(accepted), stats


# ---------------------------------------------------------------------------
# Lyapunov timescale estimation
# ---------------------------------------------------------------------------
def estimate_lyapunov_time(theta, masses, T, n_points=200, perturbation=1e-8,
                            eps=1e-3, fit_fraction=0.15):
    """
    Empirically estimate the finite-time Lyapunov exponent / e-folding time
    for a given initial condition, by evolving two nearby trajectories and
    measuring their exponential divergence rate.

    Method:
      1. Take theta, and a slightly perturbed twin theta + delta
         (delta is a small random direction of size `perturbation`).
      2. Integrate both forward over [0, T], recording the full state at
         n_points times.
      3. Compute Delta(t) = || state_1(t) - state_2(t) || at each time.
      4. Fit  ln(Delta(t)) ~ lambda * t + const  via linear regression over
         the *early* portion of the trajectory (fit_fraction of the window).
         Divergence in the 3-body problem grows in BURSTS tied to close
         encounters between bodies (Heggie 1991, Sec 6.2), not as a smooth
         exponential -- a long fit window gets dragged around by these
         spikes and degrades the fit even when the underlying growth is
         genuinely exponential on average. Empirically, a short window
         (~15% of T) gives much cleaner fits than a long one (see
         lyapunov_sweep.py).
      5. Lyapunov time (e-folding time) = 1 / lambda.

    Returns a dict with:
        t, log_delta      : arrays for plotting/diagnosis
        lyapunov_rate      : lambda (growth rate)
        lyapunov_time      : 1/lambda (the timescale to use for setting T
                              and the observation grid: "a few / lambda")
        fit_ok             : whether growth looked roughly exponential
                              (positive slope with reasonable fit quality)
    """
    rng = np.random.default_rng(0)
    theta = np.asarray(theta, dtype=float)
    direction = rng.normal(size=8)
    direction /= np.linalg.norm(direction)
    theta_pert = theta + perturbation * direction

    pos1, vel1 = theta_to_state(theta, masses)
    pos2, vel2 = theta_to_state(theta_pert, masses)

    t_eval = np.linspace(0, T, n_points)
    r1 = simulate(pos1, vel1, masses, (0, T), t_eval, eps=eps, check_regime=False)
    r2 = simulate(pos2, vel2, masses, (0, T), t_eval, eps=eps, check_regime=False)

    # Use the shorter of the two (in case either solver stopped early)
    k = min(len(r1["t"]), len(r2["t"]))
    t = r1["t"][:k]
    state1 = np.concatenate([r1["pos"][:k].reshape(k, -1), r1["vel"][:k].reshape(k, -1)], axis=1)
    state2 = np.concatenate([r2["pos"][:k].reshape(k, -1), r2["vel"][:k].reshape(k, -1)], axis=1)
    delta = np.linalg.norm(state1 - state2, axis=1)
    delta = np.clip(delta, 1e-300, None)  # avoid log(0)
    log_delta = np.log(delta)

    # Fit slope over the early portion (before saturation)
    n_fit = max(5, int(fit_fraction * k))
    A = np.vstack([t[:n_fit], np.ones(n_fit)]).T
    slope, intercept = np.linalg.lstsq(A, log_delta[:n_fit], rcond=None)[0]

    # crude R^2 of the fit, to flag periodic/non-chaotic cases where the
    # "exponential growth" story doesn't really apply
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((log_delta[:n_fit] - pred) ** 2)
    ss_tot = np.sum((log_delta[:n_fit] - log_delta[:n_fit].mean()) ** 2)
    r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    lyapunov_rate = slope
    lyapunov_time = 1.0 / slope if slope > 0 else np.inf

    return {
        "t": t,
        "log_delta": log_delta,
        "lyapunov_rate": lyapunov_rate,
        "lyapunov_time": lyapunov_time,
        "fit_r2": r2_fit,
        "fit_ok": (slope > 0) and (r2_fit > 0.8),
    }
