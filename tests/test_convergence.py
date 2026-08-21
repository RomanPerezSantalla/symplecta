"""Every method should converge at its advertised order.

`order` is a class attribute, so the expected slope comes from the method
itself rather than being restated here.
"""

import numpy as np
import pytest

from helpers import (
    DRIVEN,
    NONSEP,
    NONSEP_P0,
    NONSEP_Q0,
    PENDULUM,
    PENDULUM_ANGLE,
    SHO,
    SHO_P0,
    SHO_Q0,
    angle_state,
    angle_to_cartesian,
    driven_exact,
    observed_order,
    pendulum_state,
    sho_exact,
)
from symplecta import solve_symplectic_ivp as solve
from symplecta.methods import METHODS

EXPLICIT = ["symplectic_euler", "verlet", "yoshida4"]
# high-order methods reach round-off sooner, so they get coarser grids
STEPS = {1: (0.02, 0.01, 0.005), 2: (0.02, 0.01, 0.005), 4: (0.1, 0.05, 0.025)}


def _final_error(method, problem, q0, p0, tf, h, exact, **kw):
    r = solve((0.0, tf), q0, p0, method=method, h=h, **problem, **kw)
    assert r.success
    return float(np.linalg.norm(r.q[:, -1] - exact))


@pytest.mark.parametrize("name", EXPLICIT)
def test_explicit_methods_converge_at_their_order(name):
    order = METHODS[name].order
    hs = STEPS[order]
    exact, _ = sho_exact(1.0)
    errors = [
        _final_error(name, SHO, SHO_Q0, SHO_P0, 1.0, h, exact) for h in hs
    ]
    assert observed_order(hs, errors) == pytest.approx(order, abs=0.25)


@pytest.mark.parametrize("name", ["verlet", "yoshida4"])
def test_order_survives_a_time_dependent_force(name):
    """Regression: force evaluated at the wrong time within a step.

    Verlet's second kick belongs at t + h and Yoshida's three kicks at their
    accumulated substep times. Getting either wrong is invisible on an
    autonomous problem and drops the method to first order here.
    """
    order = METHODS[name].order
    hs = STEPS[order]
    errors = [
        _final_error(name, DRIVEN, SHO_Q0, SHO_P0, 5.0, h, driven_exact(5.0))
        for h in hs
    ]
    assert observed_order(hs, errors) == pytest.approx(order, abs=0.25)


def test_implicit_midpoint_converges_at_second_order():
    hs = (0.04, 0.02, 0.01)
    reference = solve(
        (0.0, 2.0), NONSEP_Q0, NONSEP_P0, method="implicit_midpoint",
        h=1e-4, **NONSEP,
    )
    errors = [
        _final_error(
            "implicit_midpoint", NONSEP, NONSEP_Q0, NONSEP_P0, 2.0, h,
            reference.q[:, -1],
        )
        for h in hs
    ]
    assert observed_order(hs, errors) == pytest.approx(2, abs=0.25)


@pytest.mark.parametrize("name", ["shake", "rattle"])
def test_constrained_methods_converge_against_independent_physics(name):
    """Reference is the angle formulation, not a finer run of the same method.

    A constrained method that is wrong but self-consistent would still show
    a clean slope against itself; it cannot also agree with a completely
    different parameterisation of the same pendulum.
    """
    tf = 2.0
    th0, pth0 = angle_state()
    fine = solve((0.0, tf), th0, pth0, method="yoshida4", h=1e-4,
                 **PENDULUM_ANGLE)
    exact, _ = angle_to_cartesian(fine.q[:, -1], fine.p[:, -1])

    q0, p0 = pendulum_state()
    hs = (0.02, 0.01, 0.005)
    errors = [
        _final_error(name, PENDULUM, q0, p0, tf, h, exact, tol=1e-12)
        for h in hs
    ]
    assert observed_order(hs, errors) == pytest.approx(2, abs=0.25)


def test_every_method_in_the_registry_is_covered():
    """Fail loudly when a new method is added without a convergence test."""
    covered = set(EXPLICIT) | {"implicit_midpoint", "shake", "rattle"}
    assert covered == set(METHODS)
