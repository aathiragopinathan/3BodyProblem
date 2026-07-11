"""
Workstream B (cont.): observation model and noise.

Completes the data-generation pipeline:

    theta ~ prior  -->  simulate (clean trajectory)  -->  add noise  -->  y

The observation model records the full state (positions AND velocities of all
three bodies) at K fixed observation times t_1 < ... < t_K <= T, then corrupts
each recorded number with independent Gaussian noise.

Separate noise scales for position and velocity, because they are different
physical quantities measured by different means with different precision:

    x_hat = x + sigma_x * eps ,   eps  ~ N(0, I)
    v_hat = v + sigma_v * eps',   eps' ~ N(0, I)

applied i.i.d. per body, per coordinate, per timestep.

y layout: (K, 3, 2, 2) collapsed to (K, 12) as
    [ x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3 ]  per timestep
matching the simulator's internal state layout.
"""

import numpy as np
from threebody import simulate, default_masses
from priors import theta_to_state


def make_observation_times(T, K, include_t0=False):
    """
    K observation times on (0, T] by default.

    The brief asks that several observation times fall within roughly the
    first Lyapunov time, "where the information actually is". So T itself
    should be chosen as a small multiple of the Lyapunov time (see
    estimate_lyapunov_time), and then a uniform grid over [0,T] naturally
    places several points inside that informative window.

    By default we exclude t = 0 so the inverse problem does not leak the
    initial condition back to the network as part of the observation.
    """
    if K <= 0:
        raise ValueError("K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if include_t0:
        return np.linspace(0.0, T, K)
    return np.linspace(0.0, T, K + 1)[1:]


def add_noise(pos, vel, sigma_x, sigma_v, rng):
    """
    Corrupt a clean trajectory with i.i.d. Gaussian observation noise.

    pos, vel : (K, 3, 2) clean states at the observation times
    returns  : (K, 3, 2), (K, 3, 2) noisy states
    """
    pos_noisy = pos + sigma_x * rng.normal(size=pos.shape)
    vel_noisy = vel + sigma_v * rng.normal(size=vel.shape)
    return pos_noisy, vel_noisy


def flatten_observation(pos, vel):
    """
    Pack (K,3,2) positions + (K,3,2) velocities into (K, 12), matching the
    simulator's state layout: positions first, then velocities, per timestep.
    """
    K = pos.shape[0]
    return np.concatenate([pos.reshape(K, -1), vel.reshape(K, -1)], axis=1)


def simulate_observation(theta, masses, T, K, sigma_x, sigma_v, rng,
                          eps=1e-3, t_obs=None, return_clean=False):
    """
    Full forward model for ONE theta:  theta -> y

    This is the function BayesFlow's simulator wrapper will ultimately call.

    Returns
    -------
    y : (K, 12) noisy observation, or None if the trajectory was not bounded
        (caller should reject and redraw -- though if theta came from
        sample_prior_bounded with the same T, this should not happen)
    clean : (K, 12) noise-free trajectory (only if return_clean=True)
    """
    pos0, vel0 = theta_to_state(theta, masses)
    if t_obs is None:
        t_eval = make_observation_times(T, K)
    else:
        t_eval = np.asarray(t_obs, dtype=float)
        if t_eval.shape != (K,):
            raise ValueError(f"Expected t_obs shape ({K},), got {t_eval.shape}.")
        if not np.all(np.diff(t_eval) > 0):
            raise ValueError("Observation times must be strictly increasing.")
        if t_eval[0] <= 0.0:
            raise ValueError("Observation times must start strictly after t=0.")
        if t_eval[-1] > T + 1e-12:
            raise ValueError("Observation times must satisfy t_K <= T.")

    r = simulate(pos0, vel0, masses, (0, T), t_eval, eps=eps, check_regime=True)

    if not r["bounded"]:
        return (None, None) if return_clean else None

    pos_noisy, vel_noisy = add_noise(r["pos"], r["vel"], sigma_x, sigma_v, rng)
    y = flatten_observation(pos_noisy, vel_noisy)

    if return_clean:
        return y, flatten_observation(r["pos"], r["vel"])
    return y


def generate_training_pairs(n, masses, T, K, sigma_x, sigma_v, rng,
                             pos_scale=1.0, vel_scale=0.5, eps=1e-3,
                             t_obs=None, verbose=False):
    """
    Generate n (theta, y) training pairs: the dataset the NPE network learns
    from. Draws bounded thetas from the prior, simulates each, adds noise.

    Returns
    -------
    thetas : (n, 8)
    ys     : (n, K, 12)
    """
    from priors import sample_prior_bounded

    thetas, stats = sample_prior_bounded(
        rng, n, masses, T, pos_scale=pos_scale, vel_scale=vel_scale, eps=eps
    )
    if verbose:
        print(f"[generate_training_pairs] prior acceptance: "
              f"{stats['acceptance_rate']:.1%}, regimes: {stats['regime_counts']}")

    ys = []
    kept = []
    for theta in thetas:
        y = simulate_observation(
            theta, masses, T, K, sigma_x, sigma_v, rng, eps=eps, t_obs=t_obs
        )
        if y is not None:
            ys.append(y)
            kept.append(theta)

    return np.array(kept), np.array(ys)
