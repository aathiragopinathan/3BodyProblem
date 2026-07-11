"""
Workstream A: Three-body simulator core.

Physics engine that maps initial conditions (theta) -> trajectory (pre-noise).
Everything is in nondimensional units: G = 1, total mass = 1, characteristic
length = 1. Works in 2D planar. Masses are distinct, fixed, and labeled to
break permutation symmetry.

Design notes
------------
State vector layout (flat, length 12):
    [ x1, y1, x2, y2, x3, y3,  vx1, vy1, vx2, vy2, vx3, vy3 ]
i.e. positions (3,2) flattened, then velocities (3,2) flattened.

Softening: the force uses (r^2 + eps^2)^(-3/2). The energy check MUST use the
matching softened potential  -G m_i m_j / sqrt(r^2 + eps^2), otherwise energy
is not conserved and the self-check becomes meaningless.
"""

import numpy as np
from scipy.integrate import solve_ivp

N_BODIES = 3
DIM = 2


# ---------------------------------------------------------------------------
# Masses
# ---------------------------------------------------------------------------
def default_masses(ratio=(1.0, 2.0, 3.0)):
    """Distinct, fixed, labeled masses, normalized so total mass = 1."""
    m = np.asarray(ratio, dtype=float)
    return m / m.sum()


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
def accelerations(pos, masses, G=1.0, eps=0.0):
    """
    Softened gravitational accelerations.

    pos     : (N, 2) positions
    masses  : (N,)   masses
    returns : (N, 2) accelerations

    a_i = G * sum_{j != i} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
    """
    diff = pos[None, :, :] - pos[:, None, :]      # diff[i,j] = r_j - r_i, (N,N,2)
    r2 = np.sum(diff**2, axis=-1)                 # (N,N)
    denom = r2 + eps**2                           # (N,N)
    np.fill_diagonal(denom, 1.0)                  # avoid 0**-1.5 on diagonal...
    inv_r3 = denom ** (-1.5)                      # (N,N)
    np.fill_diagonal(inv_r3, 0.0)                 # ...self-interaction zeroed here
    acc = G * np.einsum("j,ijk,ij->ik", masses, diff, inv_r3)
    return acc


def rhs(t, state, masses, G, eps):
    """Right-hand side for solve_ivp. state is flat length 12."""
    pos = state[: N_BODIES * DIM].reshape(N_BODIES, DIM)
    vel = state[N_BODIES * DIM :].reshape(N_BODIES, DIM)
    acc = accelerations(pos, masses, G=G, eps=eps)
    return np.concatenate([vel.ravel(), acc.ravel()])


# ---------------------------------------------------------------------------
# Center-of-mass frame
# ---------------------------------------------------------------------------
def to_com_frame(pos, vel, masses):
    """Shift to COM frame: COM at origin, total momentum zero."""
    M = masses.sum()
    r_com = (masses[:, None] * pos).sum(axis=0) / M
    v_com = (masses[:, None] * vel).sum(axis=0) / M
    return pos - r_com, vel - v_com


# ---------------------------------------------------------------------------
# Energy (softened, consistent with the force law)
# ---------------------------------------------------------------------------
def energy(pos, vel, masses, G=1.0, eps=0.0):
    """Total energy T + W with the softened potential matching the force."""
    T = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
    W = 0.0
    for i in range(N_BODIES):
        for j in range(i + 1, N_BODIES):
            r = np.sqrt(np.sum((pos[i] - pos[j]) ** 2) + eps**2)
            W -= G * masses[i] * masses[j] / r
    return T + W


# ---------------------------------------------------------------------------
# Bounded-regime detection: collision and ejection events
# ---------------------------------------------------------------------------
def _make_collision_event(masses, G, eps, r_min):
    """
    Event function: fires when the closest pairwise distance drops to r_min.
    Terminal (stops integration) and only triggers going downward (approach).
    """
    def event(t, state, *_args):
        pos = state[: N_BODIES * DIM].reshape(N_BODIES, DIM)
        min_r = np.inf
        for i in range(N_BODIES):
            for j in range(i + 1, N_BODIES):
                r = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                min_r = min(min_r, r)
        return min_r - r_min

    event.terminal = True
    event.direction = -1.0
    return event


def _make_ejection_event(masses, G, eps, r_max):
    """
    Event function: fires when any body's distance from the COM exceeds
    r_max. Terminal and only triggers going upward (receding).
    """
    def event(t, state, *_args):
        pos = state[: N_BODIES * DIM].reshape(N_BODIES, DIM)
        M = masses.sum()
        r_com = (masses[:, None] * pos).sum(axis=0) / M
        max_dist = np.max(np.sqrt(np.sum((pos - r_com) ** 2, axis=1)))
        return r_max - max_dist

    event.terminal = True
    event.direction = -1.0
    return event



def is_bounded(init_pos, init_vel, masses, T, G=1.0, eps=1e-3,
               r_min=1e-2, r_max=50.0, n_check=50, rtol=1e-9, atol=1e-11):
    """
    Convenience check for rejection sampling (Workstream B): does this IC
    stay bounded (no collision, no ejection) over [0, T]?

    Uses a coarser t_eval grid by default since only the boolean outcome is
    needed here, not a full high-resolution trajectory -- keep n_check small
    for a fast prior-sampling loop. The event mechanism checks continuously
    (not just at t_eval points), so fast collisions between grid points are
    still caught.

    Returns (bounded: bool, regime: str)
    """
    t_eval = np.linspace(0, T, n_check)
    r = simulate(init_pos, init_vel, masses, (0, T), t_eval,
                 G=G, eps=eps, rtol=rtol, atol=atol,
                 r_min=r_min, r_max=r_max, check_regime=True)
    return r["bounded"], r["regime"]


def simulate(
    init_pos,
    init_vel,
    masses,
    t_span,
    t_eval,
    G=1.0,
    eps=1e-3,
    rtol=1e-10,
    atol=1e-12,
    enforce_com=True,
    r_min=1e-2,
    r_max=50.0,
    check_regime=True,
):
    """
    Integrate the three-body system and report energy conservation and
    bounded-regime status.

    init_pos : (3,2) initial positions
    init_vel : (3,2) initial velocities
    masses   : (3,)  masses (should sum to 1 for nondim units)
    t_span   : (t0, tf)
    t_eval   : array of observation times where the state is recorded
    r_min    : collision threshold (min pairwise distance) -- integration
               stops early if breached
    r_max    : ejection threshold (max distance of any body from COM) --
               integration stops early if breached
    check_regime : if True, attach collision/ejection event detection

    Returns a dict with:
        t, pos, vel          : trajectory at recorded times (may be shorter
                                than t_eval if integration was stopped early)
        E0, Ef, E_series      : energy trace
        max_rel_dE            : max_t |E(t) - E0| / |E0|   (trustworthiness)
        success               : integrator success flag
        bounded               : True iff no collision/ejection occurred and
                                 t_eval was fully covered
        regime                : one of "bounded", "collision", "ejection"
        stop_time             : time of event, if any, else t_span[1]
    """
    init_pos = np.asarray(init_pos, dtype=float).reshape(N_BODIES, DIM)
    init_vel = np.asarray(init_vel, dtype=float).reshape(N_BODIES, DIM)

    if enforce_com:
        init_pos, init_vel = to_com_frame(init_pos, init_vel, masses)

    state0 = np.concatenate([init_pos.ravel(), init_vel.ravel()])
    E0 = energy(init_pos, init_vel, masses, G=G, eps=eps)

    events = None
    if check_regime:
        events = [
            _make_collision_event(masses, G, eps, r_min),
            _make_ejection_event(masses, G, eps, r_max),
        ]

    sol = solve_ivp(
        rhs,
        t_span,
        state0,
        method="DOP853",
        t_eval=t_eval,
        args=(masses, G, eps),
        rtol=rtol,
        atol=atol,
        dense_output=False,
        events=events,
    )

    states = sol.y.T                              # (K, 12), K <= len(t_eval)
    pos = states[:, : N_BODIES * DIM].reshape(-1, N_BODIES, DIM)
    vel = states[:, N_BODIES * DIM :].reshape(-1, N_BODIES, DIM)

    # Energy drift across all recorded steps (the trustworthiness self-check)
    if len(sol.t) > 0:
        E_series = np.array(
            [energy(pos[k], vel[k], masses, G=G, eps=eps) for k in range(len(sol.t))]
        )
        max_rel_dE = np.max(np.abs(E_series - E0) / np.abs(E0))
    else:
        E_series = np.array([])
        max_rel_dE = np.nan

    # Classify regime
    regime = "bounded"
    stop_time = t_span[1]
    if check_regime:
        collided = len(sol.t_events[0]) > 0
        ejected = len(sol.t_events[1]) > 0
        if collided:
            regime = "collision"
            stop_time = sol.t_events[0][0]
        elif ejected:
            regime = "ejection"
            stop_time = sol.t_events[1][0]
    # Also treat "integration didn't cover the full requested window" as
    # not bounded, even without an explicit event (e.g. solver gave up).
    covered_full_window = len(sol.t) > 0 and sol.t[-1] >= t_eval[-1] - 1e-9
    bounded = (regime == "bounded") and sol.success and covered_full_window

    return {
        "t": sol.t,
        "pos": pos,
        "vel": vel,
        "E0": E0,
        "Ef": E_series[-1] if len(E_series) else np.nan,
        "E_series": E_series,
        "max_rel_dE": max_rel_dE,
        "success": sol.success,
        "bounded": bounded,
        "regime": regime,
        "stop_time": stop_time,
    }
