"""SHAKE and RATTLE: what they hold, and where they differ."""

import numpy as np
import pytest

from helpers import PENDULUM, ROD, energy_band, pendulum_state
from symplecta import Constraints, SeparableSystem
from symplecta import solve_symplectic_ivp as solve
from symplecta.methods import Rattle, Shake

CON = Constraints(
    g=lambda q: np.array([q @ q - ROD**2]),
    jac=lambda q: 2 * q[None, :],
)


def _violations(result):
    """max |g(q)| along a trajectory."""
    return max(
        abs(result.q[:, k] @ result.q[:, k] - ROD**2)
        for k in range(result.t.size)
    )


def _hidden(result, system):
    """max |G(q) M^-1 p| along a trajectory - the velocity-level constraint."""
    return max(
        abs(
            (system.constraints.jac(result.q[:, k])
             @ system.velocity(result.p[:, k]))[0]
        )
        for k in range(result.t.size)
    )


@pytest.mark.parametrize("name", ["shake", "rattle"])
def test_position_constraint_holds_without_drifting(name):
    q0, p0 = pendulum_state()
    r = solve((0.0, 200.0), q0, p0, method=name, h=0.005, tol=1e-11, **PENDULUM)
    assert r.success
    assert _violations(r) < 1e-9


@pytest.mark.parametrize("name", ["shake", "rattle"])
def test_constrained_methods_conserve_energy(name):
    q0, p0 = pendulum_state()
    r = solve((0.0, 100.0), q0, p0, method=name, h=0.005, tol=1e-11, **PENDULUM)
    assert energy_band(r) < 1e-3
    assert abs(r.energy[-1] - r.energy[0]) <= energy_band(r)


def test_rattle_enforces_the_hidden_constraint_and_shake_does_not():
    """The one test that tells the two methods apart.

    RATTLE's velocity stage is a direct linear solve, so its residual sits
    at round-off rather than at `tol`. SHAKE never imposes the condition at
    all and leaves it O(h) - here the two differ by ten orders of magnitude.
    """
    q0, p0 = pendulum_state()
    system = SeparableSystem(force=PENDULUM["force"], constraints=CON)
    kw = dict(h=0.01, tol=1e-11, **PENDULUM)

    shake = solve((0.0, 20.0), q0, p0, method="shake", **kw)
    rattle = solve((0.0, 20.0), q0, p0, method="rattle", **kw)

    assert _hidden(rattle, system) < 1e-13
    assert _hidden(shake, system) > 1e-3
    assert _hidden(rattle, system) < _hidden(shake, system) / 1e8


def test_shake_newton_variants_agree():
    """'frozen' and 'exact' solve the same equation by different routes."""
    q0, p0 = pendulum_state()
    kw = dict(method="shake", h=0.01, tol=1e-11, **PENDULUM)
    frozen = solve((0.0, 20.0), q0, p0, newton="frozen", **kw)
    exact = solve((0.0, 20.0), q0, p0, newton="exact", **kw)
    assert np.allclose(frozen.q, exact.q, atol=1e-9)
    assert np.allclose(frozen.p, exact.p, atol=1e-9)


def test_exact_newton_converges_where_frozen_does_not():
    """At coarse steps the frozen Jacobian stops contracting."""
    q0, p0 = pendulum_state()
    kw = dict(method="shake", h=0.5, tol=1e-10, **PENDULUM)
    assert not solve((0.0, 20.0), q0, p0, newton="frozen", **kw).success
    assert solve((0.0, 20.0), q0, p0, newton="exact", **kw).success


def test_redundant_constraints_are_reported_as_rank_deficiency():
    """cho_factor failing on G M^-1 G.T means the constraints are dependent."""
    q0, p0 = pendulum_state()
    duplicated = {
        "g": lambda q: np.array([q @ q - 1.0, 2 * (q @ q - 1.0)]),
        "jac": lambda q: np.vstack([2 * q, 4 * q]),
    }
    with pytest.raises(ValueError, match="rank deficient"):
        solve((0.0, 1.0), q0, p0, method="shake", h=0.01,
              force=PENDULUM["force"], constraints=duplicated)


def test_initial_position_must_lie_on_the_manifold():
    with pytest.raises(ValueError, match=r"q0 does not satisfy"):
        solve((0.0, 1.0), [2.0, 0.0], [0.0, 0.0], method="shake", h=0.01,
              force=PENDULUM["force"], constraints=PENDULUM["constraints"])


def test_non_tangent_initial_momentum_warns_but_runs():
    q0, _ = pendulum_state()
    outward = q0 / np.linalg.norm(q0)          # straight along the rod
    with pytest.warns(RuntimeWarning, match="not tangent"):
        r = solve((0.0, 1.0), q0, outward, method="rattle", h=0.01,
                  force=PENDULUM["force"],
                  constraints=PENDULUM["constraints"])
    assert r.success


@pytest.mark.parametrize("cls", [Shake, Rattle])
def test_constrained_step_guards_on_a_bare_system(cls):
    """Calling `step` directly, bypassing the driver's checks."""
    bare = SeparableSystem(force=PENDULUM["force"])
    q0, p0 = pendulum_state()
    with pytest.raises(ValueError, match=cls.__name__):
        cls().step(0.0, q0, p0, 0.01, bare)


def test_rattle_is_a_shake():
    """Stage one is SHAKE verbatim, which is why the options are inherited."""
    assert issubclass(Rattle, Shake)
    assert Rattle(tol=1e-8, max_iter=7, newton="exact").newton == "exact"
