"""The system classes: mass handling, energy, and input validation."""

import numpy as np
import pytest

from helpers import NONSEP
from symplecta import GeneralSystem, SeparableSystem


def _force(t, q):
    return -q


# --- mass, in its three accepted forms ----------------------------------


def test_scalar_mass_divides():
    s = SeparableSystem(force=_force, mass=4.0)
    assert np.allclose(s.velocity(np.array([2.0, 8.0])), [0.5, 2.0])


def test_diagonal_mass_divides_elementwise():
    s = SeparableSystem(force=_force, mass=np.array([2.0, 4.0]))
    assert np.allclose(s.velocity(np.array([2.0, 8.0])), [1.0, 2.0])


def test_matrix_mass_solves_rather_than_inverting():
    """Off-diagonal mass means momentum is not parallel to velocity."""
    m = np.array([[3.0, 1.0], [1.0, 1.0]])
    s = SeparableSystem(force=_force, mass=m)
    p = np.array([1.0, 0.0])
    v = s.velocity(p)
    assert np.allclose(v, [0.5, -0.5])
    assert np.allclose(m @ v, p)          # it really solved M v = p
    assert v[1] != 0.0                    # a push in q0 alone moves q1 too


def test_the_three_mass_forms_agree_when_they_describe_the_same_matrix():
    p = np.array([3.0, 5.0])
    scalar = SeparableSystem(force=_force, mass=2.0).velocity(p)
    diagonal = SeparableSystem(force=_force, mass=np.full(2, 2.0)).velocity(p)
    matrix = SeparableSystem(force=_force, mass=2.0 * np.eye(2)).velocity(p)
    assert np.allclose(scalar, diagonal)
    assert np.allclose(scalar, matrix)


@pytest.mark.parametrize(
    ("mass", "match"),
    [
        (0.0, "positive"),
        (-1.0, "positive"),
        (np.array([1.0, -2.0]), "positive"),
        (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "square"),
        (np.array([[1.0, 2.0], [0.0, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
        (np.zeros((2, 2, 2)), "scalar, 1-D, or 2-D"),
    ],
)
def test_bad_mass_is_rejected_at_construction(mass, match):
    with pytest.raises(ValueError, match=match):
        SeparableSystem(force=_force, mass=mass)


def test_symmetry_is_checked_separately_from_definiteness():
    """cho_factor reads one triangle only, so it cannot catch asymmetry.

    Left to itself it would silently factor the symmetric matrix implied by
    whichever triangle it read, and integrate a different physical system.
    """
    with pytest.raises(ValueError, match="symmetric"):
        SeparableSystem(force=_force, mass=np.array([[2.0, 9.0], [0.0, 2.0]]))


# --- energy -------------------------------------------------------------


def test_separable_energy_is_kinetic_plus_potential():
    s = SeparableSystem(force=_force, mass=2.0,
                        potential=lambda t, q: 0.5 * q @ q)
    q, p = np.array([1.0]), np.array([2.0])
    assert s.energy(0.0, q, p) == pytest.approx(0.5 * 4.0 / 2.0 + 0.5)


def test_general_energy_is_the_hamiltonian_itself():
    g = GeneralSystem(**NONSEP)
    q, p = np.array([0.7]), np.array([0.4])
    assert g.energy(0.0, q, p) == pytest.approx(NONSEP["hamiltonian"](0.0, q, p))


@pytest.mark.parametrize(
    ("system", "match"),
    [
        (SeparableSystem(force=_force), "potential"),
        (GeneralSystem(dHdq=lambda t, q, p: q, dHdp=lambda t, q, p: p),
         "hamiltonian"),
    ],
)
def test_energy_without_the_needed_callable_names_it(system, match):
    with pytest.raises(ValueError, match=match):
        system.energy(0.0, np.array([1.0]), np.array([1.0]))


def test_tracks_energy_reports_availability():
    assert not SeparableSystem(force=_force).tracks_energy
    assert SeparableSystem(force=_force, potential=lambda t, q: 0.0).tracks_energy
    assert GeneralSystem(**NONSEP).tracks_energy
    assert not GeneralSystem(dHdq=lambda t, q, p: q,
                             dHdp=lambda t, q, p: p).tracks_energy


def test_is_constrained_is_answerable_for_both_system_kinds():
    """The driver asks this without knowing which kind it holds."""
    assert not SeparableSystem(force=_force).is_constrained
    assert not GeneralSystem(**NONSEP).is_constrained


# --- general system, partials and the finite-difference fallback --------


def test_partials_given_explicitly_are_used_as_given():
    g = GeneralSystem(**NONSEP)
    assert not g.numerical_gradients


def test_hamiltonian_alone_builds_the_partials():
    g = GeneralSystem(hamiltonian=NONSEP["hamiltonian"])
    assert g.numerical_gradients
    q, p = np.array([0.7, -1.2]), np.array([0.4, 0.9])
    assert np.allclose(g.dHdq(0.0, q, p), NONSEP["dHdq"](0.0, q, p), atol=1e-8)
    assert np.allclose(g.dHdp(0.0, q, p), NONSEP["dHdp"](0.0, q, p), atol=1e-8)


def test_central_differences_beat_the_forward_difference_floor():
    """Forward differences would cap near sqrt(eps) ~ 1.5e-8; central do better."""
    g = GeneralSystem(hamiltonian=lambda t, q, p: 0.5 * (p @ p + q @ q))
    q, p = np.array([0.5]), np.array([0.8])
    assert abs(g.dHdq(0.0, q, p)[0] - 0.5) < 1e-10
    assert abs(g.dHdp(0.0, q, p)[0] - 0.8) < 1e-10


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dHdq": lambda t, q, p: q}, "missing dHdp"),
        ({"dHdp": lambda t, q, p: p}, "missing dHdq"),
        ({}, "either both partials"),
        ({"hamiltonian": lambda t, q, p: 0.0, "fd_rel_step": 0.0},
         "fd_rel_step"),
    ],
)
def test_general_system_validates_its_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GeneralSystem(**kwargs)
