"""`solve_symplectic_ivp`: the grid, the result, and what it refuses."""

import numpy as np
import pytest

from helpers import (
    NONSEP,
    NONSEP_H,
    NONSEP_P0,
    NONSEP_Q0,
    PENDULUM,
    SHO,
    SHO_P0,
    SHO_Q0,
    pendulum_state,
    sho_exact,
)
from symplecta import SymplecticResult
from symplecta import solve_symplectic_ivp as solve
from symplecta.methods import METHODS, VelocityVerlet

# --- result shape and layout --------------------------------------------


def test_result_uses_scipys_array_layout():
    """(n_dof, n_points), so res.q[i] is one coordinate's trajectory."""
    q0, p0 = np.array([1.0, 2.0, 3.0]), np.zeros(3)
    r = solve((0.0, 1.0), q0, p0, force=lambda t, q: -q, n_steps=10)
    assert isinstance(r, SymplecticResult)
    assert r.t.shape == (11,)
    assert r.q.shape == (3, 11)
    assert r.p.shape == (3, 11)
    assert r.y.shape == (6, 11)
    assert np.allclose(r.y[:3], r.q)
    assert np.allclose(r.y[3:], r.p)
    assert np.allclose(r.q[:, 0], q0)


def test_result_reports_the_method_and_evaluation_count():
    r = solve((0.0, 1.0), SHO_Q0, SHO_P0, method="verlet", n_steps=50, **SHO)
    assert r.method == "verlet"
    assert r.nfev == 100          # velocity Verlet: two force calls per step
    assert r.success and "reached the end" in r.message


# --- the fixed grid -----------------------------------------------------


def test_n_steps_gives_exactly_that_many_steps():
    r = solve((0.0, 1.0), SHO_Q0, SHO_P0, n_steps=200, **SHO)
    assert r.t.size == 201
    assert r.t[-1] == 1.0


def test_h_is_an_upper_bound_and_the_grid_lands_on_tf():
    with pytest.warns(RuntimeWarning, match="h reduced"):
        r = solve((0.0, 1.0), SHO_Q0, SHO_P0, h=0.08, **SHO)
    assert r.t.size - 1 == 13                 # ceil(1 / 0.08)
    assert r.t[-1] == pytest.approx(1.0)
    assert np.diff(r.t).max() < 0.08 + 1e-12  # never exceeds what was asked


def test_an_exactly_dividing_h_is_not_adjusted():
    r = solve((0.0, 1.0), SHO_Q0, SHO_P0, h=0.01, **SHO)
    assert r.t.size - 1 == 100


def test_backward_integration_retraces_a_forward_run():
    """h is a magnitude; direction comes from t_span."""
    q0, p0 = np.array([1.0]), np.array([0.3])
    fw = solve((0.0, 10.0), q0, p0, method="verlet", h=0.001, **SHO)
    bw = solve((10.0, 0.0), fw.q[:, -1], fw.p[:, -1], method="verlet",
               h=0.001, **SHO)
    assert bw.t[-1] == pytest.approx(0.0)
    assert bw.q[0, -1] == pytest.approx(q0[0], abs=1e-12)
    assert bw.p[0, -1] == pytest.approx(p0[0], abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one of `h` or `n_steps`"),
        ({"h": 0.1, "n_steps": 10}, "exactly one of `h` or `n_steps`"),
        ({"h": 0.0}, "non-zero"),
        ({"n_steps": 0}, "at least 1"),
    ],
)
def test_bad_grid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        solve((0.0, 1.0), SHO_Q0, SHO_P0, **kwargs, **SHO)


def test_empty_t_span():
    with pytest.raises(ValueError, match="empty"):
        solve((1.0, 1.0), SHO_Q0, SHO_P0, h=0.1, **SHO)


def test_mismatched_initial_conditions():
    with pytest.raises(ValueError, match="same shape"):
        solve((0.0, 1.0), [1.0, 2.0], [0.0], h=0.1, force=lambda t, q: -q)


# --- t_eval -------------------------------------------------------------


def test_t_eval_interpolates_the_fixed_grid():
    times = np.linspace(0.0, 10.0, 7)
    r = solve((0.0, 10.0), SHO_Q0, SHO_P0, method="yoshida4", h=0.001,
              t_eval=times, **SHO)
    assert np.allclose(r.t, times)
    assert r.q.shape == (1, times.size)
    assert np.allclose(r.q[0], np.cos(times), atol=1e-8)
    assert r.energy.shape == times.shape


def test_t_eval_outside_the_interval_is_rejected():
    with pytest.raises(ValueError, match="outside the integrated interval"):
        solve((0.0, 1.0), SHO_Q0, SHO_P0, h=0.01, t_eval=[0.5, 2.0], **SHO)


# --- choosing a method --------------------------------------------------


def test_a_method_instance_may_be_passed_directly():
    r = solve((0.0, 1.0), SHO_Q0, SHO_P0, method=VelocityVerlet(), h=0.01,
              **SHO)
    assert r.success
    assert r.method == "VelocityVerlet"


def test_options_are_forwarded_to_a_named_method():
    q0, p0 = pendulum_state()
    r = solve((0.0, 1.0), q0, p0, method="shake", h=0.01, tol=1e-8,
              max_iter=10, **PENDULUM)
    assert r.success


def test_options_alongside_an_instance_are_refused():
    with pytest.raises(TypeError, match="only accepted when `method` is a name"):
        solve((0.0, 1.0), SHO_Q0, SHO_P0, method=VelocityVerlet(), h=0.01,
              tol=1e-8, **SHO)


def test_unknown_method_lists_the_valid_ones():
    with pytest.raises(ValueError, match="unknown method"):
        solve((0.0, 1.0), SHO_Q0, SHO_P0, method="banana", h=0.01, **SHO)


def test_a_named_method_that_does_not_take_the_option():
    with pytest.raises(TypeError, match="does not accept"):
        solve((0.0, 1.0), SHO_Q0, SHO_P0, method="verlet", h=0.01,
              tol=1e-8, **SHO)


# --- refusing combinations that would be silently wrong -----------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "no Hamiltonian given"),
        ({"force": lambda t, q: -q, **NONSEP}, "give either"),
        ({**NONSEP_H, "mass": 2.0}, "`mass` belongs to the separable form"),
        ({**NONSEP_H, "potential": lambda t, q: 0.0},
         "`potential` belongs to the separable form"),
        ({**NONSEP_H, "constraints": PENDULUM["constraints"]},
         "only supported for separable"),
    ],
)
def test_incoherent_hamiltonian_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        solve((0.0, 1.0), SHO_Q0, SHO_P0, h=0.01, **kwargs)


def test_explicit_method_on_a_general_system_is_refused():
    """It would run, and return a trajectory that is neither symplectic
    nor convergent - the failure mode this check exists to prevent."""
    with pytest.raises(ValueError, match="consumes a separable system"):
        solve((0.0, 1.0), NONSEP_Q0, NONSEP_P0, method="verlet", h=0.01,
              **NONSEP)


def test_general_method_on_a_separable_system_is_refused():
    with pytest.raises(ValueError, match="consumes a general system"):
        solve((0.0, 1.0), SHO_Q0, SHO_P0, method="implicit_midpoint", h=0.01,
              **SHO)


def test_constrained_system_through_an_unconstrained_method_is_refused():
    q0, p0 = pendulum_state()
    with pytest.raises(ValueError, match="carries constraints"):
        solve((0.0, 1.0), q0, p0, method="verlet", h=0.01, **PENDULUM)


def test_constrained_method_without_constraints_is_refused():
    with pytest.raises(ValueError, match="no constraints were given"):
        solve((0.0, 1.0), SHO_Q0, SHO_P0, method="rattle", h=0.01, **SHO)


def test_constraints_may_be_a_mapping_or_a_Constraints():
    q0, p0 = pendulum_state()
    r = solve((0.0, 1.0), q0, p0, method="shake", h=0.01, **PENDULUM)
    assert r.success


def test_a_malformed_constraints_mapping_says_what_it_needs():
    q0, p0 = pendulum_state()
    with pytest.raises(ValueError, match="'g' and 'jac'"):
        solve((0.0, 1.0), q0, p0, method="shake", h=0.01,
              force=PENDULUM["force"], constraints={"g": lambda q: q})


# --- failure handling ---------------------------------------------------


def test_a_step_that_cannot_converge_truncates_instead_of_raising():
    """scipy-like: keep the partial trajectory and report where it stopped."""
    r = solve((0.0, 50.0), NONSEP_Q0, NONSEP_P0, method="implicit_midpoint",
              h=5.0, max_iter=3, **NONSEP)
    assert not r.success
    assert r.t.size < 11
    assert "did not converge" in r.message
    assert r.q.shape[1] == r.t.size          # arrays truncated in step
    assert r.p.shape[1] == r.t.size


def test_t_eval_after_a_failed_run_is_an_error_not_a_guess():
    with pytest.raises(RuntimeError, match="integration failed"):
        solve((0.0, 50.0), NONSEP_Q0, NONSEP_P0, method="implicit_midpoint",
              h=5.0, max_iter=3, t_eval=[0.0, 10.0], **NONSEP)


# --- every method reachable through the public API ----------------------


@pytest.mark.parametrize("name", sorted(METHODS))
def test_every_registered_method_runs_through_the_driver(name):
    if METHODS[name].handles_constraints:
        q0, p0 = pendulum_state()
        problem = PENDULUM
    elif METHODS[name].interface == "general":
        q0, p0, problem = NONSEP_Q0, NONSEP_P0, NONSEP
    else:
        q0, p0, problem = SHO_Q0, SHO_P0, SHO
    r = solve((0.0, 1.0), q0, p0, method=name, h=0.01, **problem)
    assert r.success
    assert r.t.size == 101
    assert np.all(np.isfinite(r.q)) and np.all(np.isfinite(r.p))


def test_the_documented_example_reproduces_the_exact_solution():
    r = solve((0.0, 10.0), SHO_Q0, SHO_P0, method="yoshida4", h=0.001, **SHO)
    exact_q, exact_p = sho_exact(10.0)
    assert r.q[0, -1] == pytest.approx(exact_q[0], abs=1e-10)
    assert r.p[0, -1] == pytest.approx(exact_p[0], abs=1e-10)
