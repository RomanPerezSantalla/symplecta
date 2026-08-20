from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

_RANK_DEFICIENT = (
    "constraint Jacobian is rank deficient; the constraints are redundant "
    "or contradictory"
)


def _factor_spd(a):
    """
    Cholesky-factor the multiplier matrix, naming the failure usefully.
    """
    try:
        return cho_factor(a)
    except LinAlgError as exc:
        raise ValueError(_RANK_DEFICIENT) from exc


def _solve_general(a, b):
    """
    Solve the non-symmetric multiplier system, naming the failure usefully.
    """
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(_RANK_DEFICIENT) from exc


class SymplecticEuler:
    """
    First-order symplectic Euler (both variants via ``variant=``).
    """

    order = 1
    interface = "separable"
    handles_constraints = False

    def __init__(self, variant: str = "pq"):
        if variant not in ("pq", "qp"):
            raise ValueError("Valid variants are pq (momentum first) and qp (position first).")
        self.variant = variant

    def step(self, t, q, p, h, system):
        if self.variant == "pq":
            p = p + h * system.force(t, q)
            q = q + h * system.velocity(p)
        elif self.variant == "qp":
            q = q + h * system.velocity(p)
            p = p + h * system.force(t + h, q)
        return q, p


class VelocityVerlet:
    """
    Second-order velocity Verlet / leapfrog for separable H = T(p) + V(q).
    """

    order = 2
    interface = "separable"
    handles_constraints = False

    def step(self, t, q, p, h, system):
        p = p + h / 2 * system.force(t, q)
        q = q + h * system.velocity(p)
        p = p + h / 2 * system.force(t + h, q)
        return q, p


class Yoshida4:
    """
    Fourth-order Yoshida.
    """

    order = 4
    interface = "separable"
    handles_constraints = False
    _cbrt2 = 2 ** (1 / 3)
    w1 = 1 / (2 - _cbrt2)
    w0 = -_cbrt2 / (2 - _cbrt2)
    c1 = w1 / 2
    c2 = (w0 + w1) / 2
    k1 = c1 + 2 * c2

    def step(self, t, q, p, h, system):
        # Step 1
        q = q + self.c1 * h * system.velocity(p)
        p = p + self.w1 * h * system.force(t + self.c1 * h, q)

        # Step 2
        q = q + self.c2 * h * system.velocity(p)
        p = p + self.w0 * h * system.force(t + 0.5 * h, q) # 0.5 is the sum of c1 and c2

        # Step 3
        q = q + self.c2 * h * system.velocity(p)
        p = p + self.w1 * h * system.force(t + self.k1 * h, q)

        # Step 4
        q = q + self.c1 * h * system.velocity(p)

        return q, p


class Shake:
    """
    Position-only constrained variant of Verlet.

    ``newton`` selects how the multiplier solve is done:

    ``'frozen'`` (default)
        Freeze the Jacobian at ``G(q0)``.
    ``'exact'``
        Re-evaluate the Jacobian at the current iterate.
    """

    order = 2
    interface = "separable"
    handles_constraints = True

    def __init__(self, tol: float = 1e-10, max_iter: int = 50,
                 newton: str = "frozen"):
        if newton not in ("frozen", "exact"):
            raise ValueError(
                f"newton must be 'frozen' (Cholesky, Jacobian held at q0) or "
                f"'exact' (LU, Jacobian re-evaluated); got {newton!r}"
            )
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol!r}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter!r}")
        self.tol = tol
        self.max_iter = max_iter
        self.newton = newton

    def step(self, t, q, p, h, system):
        if system.constraints is None:
            raise ValueError(
                f"{type(self).__name__} requires constraints; construct the "
                f"system with constraints=Constraints(g=..., jac=...)"
            )
        con = system.constraints
        g0 = con.jac(q)                                   # (m, n)

        # unconstrained half-kick and drift
        p_unc = p + h / 2 * system.force(t, q)
        q_unc = q + h * system.velocity(p_unc)

        # (h^2/2) M^-1 G0.T, shape (n, m) -- written B in the derivation. It
        # turns a multiplier into the position shift that multiplier produces,
        # so q1 = q_unc - shift @ lam. Note it sits on the *left* of the Newton
        # system; the right-hand side is `residual`. Built column by column
        # because `velocity` expects a 1-D momentum: on a 2-D argument the
        # scalar and diagonal mass branches would broadcast the wrong way.
        shift = h * h / 2 * np.column_stack([system.velocity(row) for row in g0])

        # 'frozen': the matrix is symmetric and constant across iterations,
        # so factor it once here. 'exact': it moves with q1, so it is built
        # inside the loop and there is nothing to factor up front.
        _chol = _factor_spd(g0 @ shift) if self.newton == "frozen" else None

        # Newton on  g(q_unc - shift @ lam) = 0, starting from lam = 0 (that is,
        # from the unconstrained position). q1 and residual are established
        # before the loop so both are defined on every path out of it.
        lam = np.zeros(g0.shape[0])
        q1 = q_unc
        residual = con.g(q1)
        for _ in range(self.max_iter):
            if np.max(np.abs(residual)) <= self.tol:
                break
            if _chol is None:
                lam = lam + _solve_general(con.jac(q1) @ shift, residual)
            else:
                lam = lam + cho_solve(_chol, residual)
            q1 = q_unc - shift @ lam
            residual = con.g(q1)
        else:
            hint = "" if _chol is None else (
                " Try newton='exact', which converges over a wider range of h."
            )
            raise RuntimeError(
                f"{type(self).__name__} did not satisfy the constraints in "
                f"{self.max_iter} "
                f"iterations: residual {np.max(np.abs(residual)):.3e} exceeds "
                f"tol {self.tol:.3e}.{hint}"
            )

        # q1 already holds q_unc - shift @ lam from the loop
        q_new = q1
        p_half = p_unc - h / 2 * (g0.T @ lam)
        p_new = p_half + h / 2 * system.force(t + h, q_new)
        return q_new, p_new


class Rattle(Shake):
    """
    Constrained Verlet with position *and* velocity projection stages.
    """

    order = 2
    interface = "separable"
    handles_constraints = True

    def step(self, t, q, p, h, system):
        # stage one: SHAKE. Its returned momentum is the unprojected
        # half-kick, i.e. exactly the p1_unc stage two needs.
        q1, p1_unc = super().step(t, q, p, h, system)

        # stage two: project the momentum onto the constraint manifold
        con = system.constraints
        g1 = con.jac(q1)
        minv_g1t = np.column_stack([system.velocity(row) for row in g1])
        a = (h / 2) * (g1 @ minv_g1t)
        rhs = g1 @ system.velocity(p1_unc)
        mu = cho_solve(_factor_spd(a), rhs)
        return q1, p1_unc - (h / 2) * (g1.T @ mu)


class ImplicitMidpoint:
    """
    Implicit midpoint rule. Symplectic for arbitrary H(t, q, p).
    """

    order = 2
    interface = "general"
    handles_constraints = False

    def __init__(self, tol: float = 1e-12, max_iter: int = 50):
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol!r}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter!r}")
        self.tol = tol
        self.max_iter = max_iter

    def step(self, t, q, p, h, system):
        tol = self._effective_tol(system)
        tmid = t + h / 2
        qmid, pmid = q, p
        for _ in range(self.max_iter):
            qn = q + h / 2 * system.dHdp(tmid, qmid, pmid)
            pn = p - h / 2 * system.dHdq(tmid, qmid, pmid)
            change = max(np.max(np.abs(qn - qmid)), np.max(np.abs(pn - pmid)))
            qmid, pmid = qn, pn
            if change <= tol:
                break
        else:
            raise RuntimeError(
                f"ImplicitMidpoint did not converge in {self.max_iter} "
                f"iterations: {change:.3e} exceeds tol {tol:.3e}. "
                f"Reduce h, or raise max_iter."
            )
        return 2 * qmid - q, 2 * pmid - p

    def _effective_tol(self, system):
        """
        `tol`, floored at what finite-difference partials can deliver.
        """
        if not getattr(system, "numerical_gradients", False):
            return self.tol
        floor = system.fd_rel_step ** 2
        if self.tol >= floor:
            return self.tol
        warnings.warn(
            f"dHdq/dHdp are finite differences of `hamiltonian`, accurate to "
            f"about {floor:.1e} and tol={self.tol:.1e} is below that."
            f"Provide exact dHdq/dHdp for a solve within tol.",
            RuntimeWarning,
            stacklevel=3,
        )
        return floor


METHODS = {
    "symplectic_euler": SymplecticEuler,
    "verlet": VelocityVerlet,
    "yoshida4": Yoshida4,
    "rattle": Rattle,
    "shake": Shake,
    "implicit_midpoint": ImplicitMidpoint,
}
