"""Energy behaviour, which is the reason the library exists.

A non-symplectic method of the same order would pass every convergence test
in `test_convergence.py` and fail everything here.
"""

import numpy as np
import pytest

from helpers import (
    NONSEP,
    NONSEP_P0,
    NONSEP_Q0,
    SHO,
    SHO_P0,
    SHO_Q0,
    energy_band,
    observed_order,
)
from symplecta import solve_symplectic_ivp as solve
from symplecta.methods import METHODS

EXPLICIT = ["symplectic_euler", "verlet", "yoshida4"]


@pytest.mark.parametrize("name", EXPLICIT)
def test_energy_stays_bounded_over_a_long_run(name):
    """Bounded, not merely small: the end must sit inside the same band.

    Explicit Euler on this problem multiplies energy by (1 + h^2) every
    step, so it would fail by an enormous margin rather than marginally.
    """
    r = solve((0.0, 400.0), SHO_Q0, SHO_P0, method=name, h=0.02, **SHO)
    e0 = r.energy[0]
    band = energy_band(r)
    assert band < 0.05 * e0
    # no secular drift: the last value is not clinging to an extreme
    assert abs(r.energy[-1] - e0) <= band


@pytest.mark.parametrize("name", EXPLICIT)
def test_energy_band_shrinks_at_the_method_order(name):
    order = METHODS[name].order
    hs = (0.08, 0.04, 0.02)
    bands = [
        energy_band(solve((0.0, 60.0), SHO_Q0, SHO_P0, method=name, h=h, **SHO))
        for h in hs
    ]
    assert observed_order(hs, bands) == pytest.approx(order, abs=0.3)


def test_higher_order_methods_conserve_energy_better():
    kw = dict(h=0.02, **SHO)
    bands = {
        name: energy_band(solve((0.0, 60.0), SHO_Q0, SHO_P0, method=name, **kw))
        for name in EXPLICIT
    }
    assert bands["verlet"] < bands["symplectic_euler"] / 10
    assert bands["yoshida4"] < bands["verlet"] / 10


def test_implicit_midpoint_conserves_a_quadratic_invariant_exactly():
    """H = (p^2 + q^2)/2 is quadratic, and Gauss methods preserve those.

    The distinction from Verlet is qualitative, not a matter of degree:
    Verlet oscillates within an O(h^2) band, this should be flat to
    round-off.
    """
    h = 0.05
    r = solve(
        (0.0, 500.0), SHO_Q0, SHO_P0, method="implicit_midpoint", h=h,
        dHdq=lambda t, q, p: q,
        dHdp=lambda t, q, p: p,
        hamiltonian=lambda t, q, p: 0.5 * (p @ p + q @ q),
    )
    assert r.success
    assert energy_band(r) < 1e-9

    verlet = solve((0.0, 500.0), SHO_Q0, SHO_P0, method="verlet", h=h, **SHO)
    assert energy_band(r) < energy_band(verlet) / 1000


def test_energy_is_none_unless_asked_for():
    without = solve((0.0, 1.0), SHO_Q0, SHO_P0, h=0.01,
                    force=SHO["force"])
    assert without.energy is None

    with_it = solve((0.0, 1.0), SHO_Q0, SHO_P0, h=0.01, **SHO)
    assert with_it.energy is not None
    assert with_it.energy.shape == with_it.t.shape


def test_non_separable_energy_is_bounded():
    r = solve((0.0, 200.0), NONSEP_Q0, NONSEP_P0, method="implicit_midpoint",
              h=0.01, **NONSEP)
    assert r.success
    assert energy_band(r) < 1e-4
    assert abs(r.energy[-1] - r.energy[0]) <= energy_band(r)


def test_finite_difference_gradients_match_exact_ones():
    """The convenience path must be a drop-in, not a degraded mode."""
    common = {"method": "implicit_midpoint", "h": 0.01, "tol": 1e-10}
    exact = solve((0.0, 50.0), NONSEP_Q0, NONSEP_P0, **common, **NONSEP)
    approx = solve((0.0, 50.0), NONSEP_Q0, NONSEP_P0, **common,
                   hamiltonian=NONSEP["hamiltonian"])
    assert np.allclose(exact.q, approx.q, atol=1e-8)
    assert np.allclose(exact.p, approx.p, atol=1e-8)
