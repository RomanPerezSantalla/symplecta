"""Reference problems and small utilities shared by the test suite.

Each problem is a dict of ``solve_symplectic_ivp`` keyword arguments, so a
test can splat it and override whatever it needs.
"""

import numpy as np

# --- harmonic oscillator, H = (p^2 + q^2)/2 -----------------------------
# Exact solution q(t) = cos t, p(t) = -sin t from (1, 0). H is quadratic,
# which matters for implicit midpoint: quadratic invariants are preserved
# exactly, so energy should be flat rather than merely bounded.
SHO = {
    "force": lambda t, q: -q,
    "potential": lambda t, q: 0.5 * q @ q,
}
SHO_Q0 = np.array([1.0])
SHO_P0 = np.array([0.0])


def sho_exact(t):
    return np.array([np.cos(t)]), np.array([-np.sin(t)])


# --- driven oscillator, force depends on t ------------------------------
# q'' = -q + A cos(wt). A method that evaluates its force at the wrong time
# within a step loses an order here and *nowhere else*: every autonomous
# problem is blind to that mistake.
_A, _W = 0.3, 0.7
_C = _A / (1 - _W**2)
DRIVEN = {"force": lambda t, q: -q + _A * np.cos(_W * t)}


def driven_exact(t):
    return np.array([(1 - _C) * np.cos(t) + _C * np.cos(_W * t)])


# --- non-separable, H = p^2 (1 + q^2)/2 + q^2/2 --------------------------
# The (1 + q^2) factor is a configuration-dependent mass, so H does not
# split as T(p) + V(q) and only the general interface can take it.
NONSEP = {
    "dHdq": lambda t, q, p: p**2 * q + q,
    "dHdp": lambda t, q, p: p * (1 + q**2),
    "hamiltonian": lambda t, q, p: 0.5 * np.sum(p**2 * (1 + q**2))
    + 0.5 * np.sum(q**2),
}
NONSEP_H = {"hamiltonian": NONSEP["hamiltonian"]}
NONSEP_Q0 = np.array([0.7])
NONSEP_P0 = np.array([0.4])


# --- bead on a rigid rod ------------------------------------------------
# Separable H with one holonomic constraint |q|^2 = L^2, unit mass.
GRAV, ROD = 9.81, 1.0
PENDULUM = {
    "force": lambda t, q: np.array([0.0, -GRAV]),
    "potential": lambda t, q: GRAV * q[1],
    "constraints": {
        "g": lambda q: np.array([q @ q - ROD**2]),
        "jac": lambda q: 2 * q[None, :],
    },
}

# The same physics as an unconstrained one-degree-of-freedom problem in the
# angle. Integrating both and comparing catches errors that a method cannot
# see when checked only against a finer run of itself.
PENDULUM_ANGLE = {
    "force": lambda t, th: np.array([-GRAV * ROD * np.sin(th[0])]),
    "mass": ROD**2,
    "potential": lambda t, th: -GRAV * ROD * np.cos(th[0]),
}


def pendulum_state(degrees=50.0):
    """Cartesian (q0, p0) for a bead released from rest at `degrees`."""
    th = np.deg2rad(degrees)
    return np.array([ROD * np.sin(th), -ROD * np.cos(th)]), np.zeros(2)


def angle_state(degrees=50.0):
    return np.array([np.deg2rad(degrees)]), np.zeros(1)


def angle_to_cartesian(th, p_th):
    """Map the angle formulation's state onto the Cartesian one."""
    th_dot = p_th[0] / ROD**2
    return (
        np.array([ROD * np.sin(th[0]), -ROD * np.cos(th[0])]),
        ROD * th_dot * np.array([np.cos(th[0]), np.sin(th[0])]),
    )


# --- utilities ----------------------------------------------------------


def observed_order(step_sizes, errors):
    """Least-squares slope of log(error) against log(h).

    More robust than a single error ratio: one unlucky step size cannot
    swing the answer on its own.
    """
    log_h = np.log(np.asarray(step_sizes, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    return float(np.polyfit(log_h, log_e, 1)[0])


def energy_band(result):
    """Peak-to-peak spread of the reported energy."""
    return float(result.energy.max() - result.energy.min())
