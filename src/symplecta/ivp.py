from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from symplecta.methods import METHODS


class _Counted:
    """
    Wrap a callable so the driver can report how often it was evaluated.
    """

    __slots__ = ("fn", "n")

    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, *args):
        self.n += 1
        return self.fn(*args)


def _as_1d(name, value):
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _resolve_grid(t0, tf, h, n_steps):
    """
    Settle the fixed grid, returning `(h, n_steps)`.
    """
    if (h is None) == (n_steps is None):
        raise ValueError(
            "give exactly one of `h` or `n_steps`"
        )
    span = tf - t0
    if span == 0:
        raise ValueError(f"t_span is empty: t0 == tf == {t0!r}")

    if n_steps is None:
        # h is a magnitude; the direction of travel comes from t_span, so a
        # backward run is t_span=(10, 0) with the same positive h
        h = abs(float(h))
        if h == 0:
            raise ValueError("h must be non-zero")
        # the epsilon keeps an exactly-divisible span off the next integer
        n_steps = max(int(np.ceil(abs(span) / h - 1e-12)), 1)
        actual = span / n_steps
        if abs(abs(actual) - h) > 1e-9 * h:
            warnings.warn(
                f"h reduced from {h:.6g} to {actual:.6g} so that {n_steps} "
                f"equal steps land exactly on tf={tf:.6g}",
                RuntimeWarning,
                stacklevel=3,
            )
        return actual, n_steps

    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    return span / n_steps, n_steps


def _resolve_method(method, options):
    """Turn a name or an instance into an integrator."""
    if isinstance(method, str):
        try:
            cls = METHODS[method]
        except KeyError:
            raise ValueError(
                f"unknown method {method!r}; choose from "
                f"{sorted(METHODS)} or pass an integrator instance"
            ) from None
        try:
            return cls(**options)
        except TypeError as exc:
            raise TypeError(
                f"{cls.__name__} does not accept {sorted(options)}: {exc}"
            ) from exc
    if options:
        raise TypeError(
            f"extra options {sorted(options)} are only accepted when `method` "
            f"is a name; configure the instance you passed instead"
        )
    for attr in ("interface", "step"):
        if not hasattr(method, attr):
            raise TypeError(
                f"a method instance must expose `{attr}`; {method!r} does not"
            )
    return method


def _build_system(force, mass, potential, dHdq, dHdp, hamiltonian, constraints):
    """
    Resolve the callable arguments into exactly one System.
    """
    separable = force is not None
    general = dHdq is not None or dHdp is not None or hamiltonian is not None

    if separable and general:
        raise ValueError(
            "give either `force` (separable H = T(p) + V(q)) or the general "
            "form (`dHdq` and `dHdp` or `hamiltonian`)"
        )
    if not separable and not general:
        raise ValueError(
            "no Hamiltonian given: pass `force` for a separable system, or "
            "`dHdq` and `dHdp` (or `hamiltonian` alone) for a general one"
        )

    if general:
        for label, value in (("mass", mass), ("potential", potential)):
            if value is not None:
                raise ValueError(
                    f"`{label}` belongs to the separable form and is not compatible with dHdq/dHdp/hamiltonian"
                )
        if constraints is not None:
            raise ValueError(
                "constraints are only supported for separable systems"
            )
        if dHdq is not None and dHdp is not None:
            counters = [_Counted(dHdq), _Counted(dHdp)]
            return GeneralSystem(counters[0], counters[1], hamiltonian), counters

        counted = _Counted(hamiltonian) if hamiltonian is not None else None
        return (
            GeneralSystem(dHdq=dHdq, dHdp=dHdp, hamiltonian=counted),
            [] if counted is None else [counted],
        )

    if constraints is not None and not isinstance(constraints, Constraints):
        try:
            constraints = Constraints(g=constraints["g"], jac=constraints["jac"])
        except (TypeError, KeyError) as exc:
            raise ValueError(
                "constraints must be a Constraints instance or a mapping with "
                "'g' and 'jac' keys"
            ) from exc
    counted = _Counted(force)
    system = SeparableSystem(
        force=counted,
        mass=1.0 if mass is None else mass,
        potential=potential,
        constraints=constraints,
    )
    return system, [counted]


def _check_compatibility(system, integrator):
    """
    Refuse combinations that would quietly integrate the wrong problem.
    """
    name = type(integrator).__name__
    wanted = integrator.interface
    kind = "separable" if isinstance(system, SeparableSystem) else "general"
    if wanted != kind:
        if wanted == "separable":
            detail = (
                "an explicit method would run on the general partials and "
                "return a trajectory that is neither symplectic nor convergent"
            )
        else:
            detail = "it assumes nothing about H and needs dHdq/dHdp"
        raise ValueError(
            f"{name} consumes a {wanted} system but a {kind} one was given: "
            f"{detail}"
        )

    handles = getattr(integrator, "handles_constraints", False)
    if system.is_constrained and not handles:
        raise ValueError(
            f"the system carries constraints but {name} does not apply them. Use method='rattle' or method='shake'"
        )
    if handles and not system.is_constrained:
        raise ValueError(
            f"{name} is a constrained method but no constraints were given. Give constraints={{'g': ..., 'jac': ...}}"
        )


def _check_initial_state(system, q0, p0):
    """
    A constrained run has to start on the manifold.
    """
    if not system.is_constrained:
        return
    violation = np.max(np.abs(system.constraints.g(q0)))
    if violation > 1e-8:
        raise ValueError(
            f"q0 does not satisfy the constraints: max|g(q0)| = "
            f"{violation:.3e}."
        )
    hidden = np.max(np.abs(system.constraints.jac(q0) @ system.velocity(p0)))
    if hidden > 1e-8:
        warnings.warn(
            f"p0 is not tangent to the constraint manifold: "
            f"max|G(q0) M^-1 p0| = {hidden:.3e}. RATTLE recommended over SHAKE.",
            RuntimeWarning,
            stacklevel=3,
        )


def _interpolate(t_eval, t_grid, q, p, energy):
    """
    Resolve off-grid output times by interpolating the fixed grid.
    """
    t_eval = _as_1d("t_eval", t_eval)
    lo, hi = min(t_grid[0], t_grid[-1]), max(t_grid[0], t_grid[-1])
    outside = (t_eval < lo - 1e-12) | (t_eval > hi + 1e-12)
    if outside.any():
        raise ValueError(
            f"t_eval values {t_eval[outside][:3]} lie outside the integrated "
            f"interval [{lo}, {hi}]"
        )
    order = np.argsort(t_grid)
    x = t_grid[order]
    q_e = CubicSpline(x, q[:, order], axis=1)(t_eval)
    p_e = CubicSpline(x, p[:, order], axis=1)(t_eval)
    e_e = None if energy is None else CubicSpline(x, energy[order])(t_eval)
    return t_eval, q_e, p_e, e_e


@dataclass(frozen=True)
class Constraints:
    """Holonomic constraints g(q) = 0, for RATTLE and SHAKE.

    Attributes
    ----------
    g : callable ``g(q) -> ndarray`` of shape (m,)
        The m constraint equations, satisfied when equal to zero.
    jac : callable ``G(q) -> ndarray`` of shape (m, n_dof)
        ``dg/dq``. The jacobian of the constaint equations.
    """

    g: Callable[[np.ndarray], np.ndarray]
    jac: Callable[[np.ndarray], np.ndarray]


@dataclass
class SeparableSystem:
    """H = T(p) + V(q).

    Attributes
    ----------
    force : callable ``f(t, q) -> -dV/dq``
    mass : scalar, 1-D array (diagonal M), or 2-D array (full M)
        Constant by construction, otherwise H is non-separable.
    potential : callable ``V(t, q) -> float``, optional
        Needed for energy calculations.
    constraints : `Constraints`, optional
        Holonomic constraints g(q) = 0. Only RATTLE and SHAKE deal with them correctly.
    """

    force: Callable[[float, np.ndarray], np.ndarray]
    mass: float | np.ndarray = 1.0
    potential: Callable[[float, np.ndarray], float] | None = None
    constraints: Constraints | None = None

    def __post_init__(self):
        m = np.asarray(self.mass, dtype=float)

        if m.ndim in (0, 1):
            if not np.all(m > 0):
                raise ValueError(f"mass must be positive, got {self.mass!r}")
            self._chol = None

        elif m.ndim == 2:
            if m.shape[0] != m.shape[1]:
                raise ValueError(f"mass matrix must be square, got shape {m.shape}")
            if not np.allclose(m, m.T):
                raise ValueError("mass matrix must be symmetric")
            try:
                self._chol = cho_factor(m)
            except LinAlgError as exc:
                raise ValueError("mass matrix must be positive definite") from exc

        else:
            raise ValueError(f"mass must be scalar, 1-D, or 2-D, got {m.ndim}-D")

        self._m = m

    def velocity(self, p):
        """
        q̇ = M⁻¹p.
        """
        if self._chol is None:
            return p / self._m
        return cho_solve(self._chol, p)

    @property
    def tracks_energy(self) -> bool:
        """
        Whether `energy` is available.
        """
        return self.potential is not None

    @property
    def is_constrained(self) -> bool:
        """
        Whether the dynamics carry holonomic constraints.
        """
        return self.constraints is not None

    def energy(self, t, q, p):
        """
        H = ½ pᵀM⁻¹p + V(t, q).
        """
        if self.potential is None:
            raise ValueError(
                "energy requires `potential`."
            )
        return 0.5 * p @ self.velocity(p) + self.potential(t, q)


#: Relative step for the finite-difference fallback. eps**(1/3) minimises
#: truncation + round-off for central differences; the resulting gradient
#: error is ~eps**(2/3) scaled by the magnitude of H.
_FD_REL_STEP = float(np.finfo(float).eps) ** (1 / 3)


def _central_gradient(f, x, rel_step):
    """
    Gradient of a scalar f by central differences: 2n evaluations.
    """
    x = np.asarray(x, dtype=float)
    grad = np.empty_like(x)
    for i in range(x.size):
        step = rel_step * max(abs(x[i]), 1.0)
        hi, lo = x.copy(), x.copy()
        hi[i] += step
        lo[i] -= step
        # re-read the actual perturbation: hi[i] - lo[i] is exactly
        # representable, the nominal 2*step generally is not
        grad[i] = (f(hi) - f(lo)) / (hi[i] - lo[i])
    return grad


@dataclass
class GeneralSystem:
    """
    Arbitrary H(t, q, p), given by its partials or by H itself.

    q' =  dH/dp,  p' = -dH/dq

    Supply *either* both partials, *or* ``hamiltonian`` alone and let the
    partials be taken by central differences.

    Attributes
    ----------
    dHdq, dHdp : callable ``f(t, q, p) -> ndarray``, optional
        Must be given together if given.
    hamiltonian : callable ``H(t, q, p) -> float``, optional
        Required to report energy, and sufficient on its own to define the dynamics.
        If dHdq and dHdp are missing, they are built from this with finite differences.
    fd_rel_step : float
        Relative step for when finite differences are used.
    """

    dHdq: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None
    dHdp: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None
    hamiltonian: Callable[[float, np.ndarray, np.ndarray], float] | None = None
    fd_rel_step: float = _FD_REL_STEP

    def __post_init__(self):
        given = (self.dHdq is not None, self.dHdp is not None)
        if given[0] != given[1]:
            missing = "dHdp" if given[0] else "dHdq"
            raise ValueError(
                f"dHdq and dHdp must be given together; missing {missing}. "
                f"Alternatively give `hamiltonian` alone."
            )
        self.numerical_gradients = not given[0]
        if not self.numerical_gradients:
            return
        if self.hamiltonian is None:
            raise ValueError(
                "a GeneralSystem needs either both partials (dHdq, dHdp) or `hamiltonian` on its own"
            )
        if self.fd_rel_step <= 0:
            raise ValueError(
                f"fd_rel_step must be positive, got {self.fd_rel_step!r}"
            )
        ham, step = self.hamiltonian, self.fd_rel_step
        self.dHdq = lambda t, q, p: _central_gradient(
            lambda x: ham(t, x, p), q, step
        )
        self.dHdp = lambda t, q, p: _central_gradient(
            lambda x: ham(t, q, x), p, step
        )

    @property
    def tracks_energy(self) -> bool:
        """
        Whether `energy` is available.
        """
        return self.hamiltonian is not None

    @property
    def is_constrained(self) -> bool:
        """
        False. Constrained non-separable systems are not supported.
        """
        return False

    def energy(self, t, q, p):
        """
        H(t, q, p).
        """
        if self.hamiltonian is None:
            raise ValueError(
                "energy requires `hamiltonian`"
            )
        return self.hamiltonian(t, q, p)


System = SeparableSystem | GeneralSystem


@dataclass
class SymplecticResult:
    """
    Result of a symplectic integration, shaped after scipy's ``OdeResult``.

    Attributes
    ----------
    t : ndarray, shape (n_points,)
        Time grid the solution is reported on.
    q, p : ndarray, shape (n_dof, n_points)
        Positions and momenta. One row per degree of freedom, matching scipy's ``OdeResult``.
    y : ndarray, shape (2 * n_dof, n_points)
        Stacked ``[q; p]`` view.
    energy : ndarray, shape (n_points,) or None
        H at each time in ``t``. ``None`` unless the system was given a
        ``potential`` (separable) or ``hamiltonian`` (general).
    success : bool
        Whether the integration ran to completion.
    message : str
        Termination reason.
    """

    t: np.ndarray
    q: np.ndarray
    p: np.ndarray
    energy: np.ndarray | None = None
    success: bool = True
    message: str = ""
    nfev: int = 0
    method: str = ""

    @property
    def y(self) -> np.ndarray:
        return np.concatenate([self.q, self.p], axis=0)


def solve_symplectic_ivp(
    t_span,
    q0,
    p0,
    *,
    force=None,
    mass=None,
    potential=None,
    dHdq=None,
    dHdp=None,
    hamiltonian=None,
    method="verlet",
    h=None,
    n_steps=None,
    constraints=None,
    t_eval=None,
    **options,
) -> SymplecticResult:
    """
    Integrate a Hamiltonian system with a fixed-step symplectic method.

    The Hamiltonian is described in one of two mutually exclusive ways, and
    which one you pass is the interface contract:

    * ``force`` (+ optional ``mass``) — asserts H = T(p) + V(q). Required by the explicit methods.
    * ``dHdq`` and ``dHdp``, or ``hamiltonian``.

    Passing both, or neither, is an error.

    Parameters
    ----------
    t_span : 2-tuple of float
        Integration interval ``(t0, tf)``.
    q0, p0 : array_like, shape (n_dof,)
        Initial positions and momenta.
    force : callable, optional
        ``f(t, q) -> -dV/dq``.
    mass : scalar, 1-D array, or 2-D array, optional
        Constant mass, defaulting to 1.0. Belongs to the separable form.
    potential : callable ``V(t, q) -> float``, optional
        Alongside ``force``; only needed to report energy.
    dHdq, dHdp : callable, optional
        ``f(t, q, p) -> ndarray``. Must be given together, or omitted in favor of ``hamiltonian``.
    hamiltonian : callable ``H(t, q, p) -> float``, optional
        The general-form Hamiltonian. Reports energy and calculates dHdq / dHdp if missing.
    method : str or object
        ``'symplectic_euler'`` | ``'verlet'`` | ``'yoshida4'`` | ``'rattle'`` | ``'shake'`` | ``'implicit_midpoint'``,
         or an integrator instance.
    h, n_steps : float or int
        Fixed step size, or number of steps across ``t_span``. Exactly one of the two.
    constraints : dict or `Constraints`, optional
        ``{'g': callable, 'jac': callable}``, assembled into a `Constraints` on the resulting system.
        Required for RATTLE/SHAKE, error for every other method.
    t_eval : array_like, optional
        Output times.

    Returns
    -------
    SymplecticResult
        ``.t``, ``.q``, ``.p``, ``.y``, ``.energy``, ``.success``, ``.message``, ``.nfev``, ``.method``.
        A step that fails to converge truncates the result and sets ``success=False``.
        Argument errors raise immediately.
    """
    q0 = _as_1d("q0", q0)
    p0 = _as_1d("p0", p0)
    if q0.shape != p0.shape:
        raise ValueError(
            f"q0 and p0 must have the same shape, got {q0.shape} and {p0.shape}"
        )
    n_dof = q0.size

    try:
        t0, tf = (float(v) for v in t_span)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"t_span must be a 2-tuple (t0, tf), got {t_span!r}") from exc

    h, n_steps = _resolve_grid(t0, tf, h, n_steps)
    integrator = _resolve_method(method, options)
    system, counters = _build_system(
        force, mass, potential, dHdq, dHdp, hamiltonian, constraints
    )
    _check_compatibility(system, integrator)
    _check_initial_state(system, q0, p0)

    t_grid = t0 + h * np.arange(n_steps + 1)
    t_grid[-1] = tf                      # exact, rather than n_steps roundings
    q_out = np.empty((n_dof, n_steps + 1))
    p_out = np.empty((n_dof, n_steps + 1))
    q_out[:, 0], p_out[:, 0] = q0, p0
    e_out = np.empty(n_steps + 1) if system.tracks_energy else None
    if e_out is not None:
        e_out[0] = system.energy(t0, q0, p0)

    q, p = q0, p0
    success, message, reached = True, "", n_steps
    for k in range(n_steps):
        try:
            q, p = integrator.step(t_grid[k], q, p, h, system)
        except RuntimeError as exc:
            # a step that could not converge is a failed integration, not a
            # crash: keep what was computed and report where it stopped
            success, message, reached = False, f"step {k} at t={t_grid[k]:.6g}: {exc}", k
            break
        q_out[:, k + 1], p_out[:, k + 1] = q, p
        if e_out is not None:
            e_out[k + 1] = system.energy(t_grid[k + 1], q, p)

    if not success:
        t_grid = t_grid[: reached + 1]
        q_out, p_out = q_out[:, : reached + 1], p_out[:, : reached + 1]
        if e_out is not None:
            e_out = e_out[: reached + 1]
    else:
        message = "reached the end of the integration interval"

    if t_eval is not None:
        if not success:
            raise RuntimeError(
                f"cannot interpolate onto t_eval: the integration failed at "
                f"{message}"
            )
        t_grid, q_out, p_out, e_out = _interpolate(
            t_eval, t_grid, q_out, p_out, e_out
        )

    return SymplecticResult(
        t=t_grid,
        q=q_out,
        p=p_out,
        energy=e_out,
        success=success,
        message=message,
        nfev=sum(c.n for c in counters),
        method=method if isinstance(method, str) else type(method).__name__,
    )

