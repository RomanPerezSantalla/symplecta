# symplecta

Fixed-step symplectic and geometric integrators for Hamiltonian systems, in pure NumPy and SciPy.

If you have integrated an orbit or a pendulum for a long time with a general-purpose solver, you will have seen
the energy slowly climb or decay even though there is no driving or dampening. That is not a bug in the
solver, and it does not go away by taking smaller steps, it is what a method that does not respect the structure
of Hamilton's equations does. The methods here respect the structure, and their energy error stays in a band instead of
drifting, however long you run them.

`scipy.integrate.solve_ivp` ships only general-purpose adaptive solvers (RK45, RK23, DOP853, Radau, BDF, LSODA)
and none of them are symplectic. We try to fill that gap while keeping a similar structure to scipy for ease of use.

## Status

Pre-alpha. All eight methods and `solve_symplectic_ivp` work, and the test suite checks convergence order, energy
behavior and constraint satisfaction for each of them. The API may still move.

## Install

```bash
pip install -e ".[dev]"
```

## Using it

Describe the Hamiltonian and hand it to the solver. How you describe it decides which methods you can use.

For a separable system, $H = T(p) + V(q)$, give a force and a mass:

```python
import numpy as np
from symplecta import solve_symplectic_ivp

res = solve_symplectic_ivp(
    t_span=(0.0, 100.0),
    q0=np.array([1.0]),
    p0=np.array([0.0]),
    force=lambda t, q: -q,
    potential=lambda t, q: 0.5 * q @ q,   # optional, only to report energy
    method="verlet",
    h=0.01,
)

res.t, res.q, res.p, res.energy
```

For a non-separable one, give the two partials of $H$, or just $H$ itself and let the partials be taken
numerically:

```python
res = solve_symplectic_ivp(
    t_span=(0.0, 100.0),
    q0=q0,
    p0=p0,
    hamiltonian=lambda t, q, p: 0.5 * np.sum(p**2 * (1 + q**2)) + 0.5 * np.sum(q**2),
    method="implicit_midpoint",
    h=0.01,
)
```

For a constrained system, add the constraint and its Jacobian:

```python
res = solve_symplectic_ivp(
    t_span=(0.0, 50.0),
    q0=q0,
    p0=p0,
    force=lambda t, q: np.array([0.0, -9.81]),
    constraints={"g": lambda q: np.array([q @ q - 1.0]),
                 "jac": lambda q: 2 * q[None, :]},
    method="rattle",
    h=0.005,
)
```

The result is shaped after scipy's, with `.t`, `.q`, `.p`, `.y`, `.energy`, `.success` and `.message`, and uses
the same `(n_dof, n_points)` array layout, so `res.q[0]` is the trajectory of the first coordinate.

## Methods

| `method=` | order | needs | cost per step |
| --- | --- | --- | --- |
| `symplectic_euler` | 1 | force, mass | 1 force evaluation |
| `verlet` | 2 | force, mass | 2 force evaluations |
| `yoshida4` | 4 | force, mass | 3 force evaluations |
| `yoshida6` | 6 | force, mass | 7 force evaluations |
| `yoshida8` | 8 | force, mass | 15 force evaluations |
| `shake` | 2 | force, mass, constraints | 2 force evaluations, one nonlinear solve |
| `rattle` | 2 | force, mass, constraints | as SHAKE plus one linear solve |
| `implicit_midpoint` | 2 | dHdq and dHdp, or hamiltonian | several gradient evaluations |

`verlet` is the sensible default for a separable system. `yoshida4` costs one more force evaluation and buys
about three orders of magnitude in the energy band, and `yoshida6` is worth its seven for a long run. Given a
fixed budget of force evaluations rather than a fixed step, the gains are smaller than that table suggests;
[the methods document](docs/methods.md) compares them at equal cost. `rattle` is preferable to `shake` in basically
every scenario.

The solver refuses combinations that would quietly give you the wrong answer, rather than running them.
Unallowed combinations are: explicit method on a non-separable Hamiltonian, an unconstrained method on a 
constrained system, or a mass alongside the general form.

## Documentation

The documents in [`docs/`](docs) are written to be read in order, and are meant to explain the methods rather
than just list the API. I wrote them because I found SHAKE and RATTLE genuinely hard to pick up from the original
papers.

- [Introduction](docs/introduction.md) — equations of motion, separable Hamiltonians, and why a regular method
  loses energy while a symplectic one does not.
- [General methods](docs/methods.md) — symplectic Euler, velocity Verlet and the Yoshida compositions, all
  built out of drifts and kicks.
- [Constraint methods](docs/constraints.md) — Lagrange multipliers from the action, then SHAKE and RATTLE and
  what separates them.
- [Implicit midpoint](docs/implicit_midpoint.md) — what to do when the Hamiltonian does not split.

## Design decisions

- **Fixed step only.** Adaptive stepping breaks the fixed-step map that the energy behavior relies on, so there
  is no adaptive control and no plan to add it.
- **Structure-aware inputs.** The solver takes `q`, `p` and a force rather than a flattened `fun(t, y)`, because
  the methods need the two halves of $H$ separately.
- **`t_eval` by interpolation.** Output at arbitrary times comes from interpolating the fixed grid afterwards,
  not from dense output per step.
- **Small and medium systems.** Few-body mechanics and small molecular toys. Force fields, GPU kernels and
  large-scale MD are [OpenMM's](https://openmm.org/) job, not this one.

## Further reading

The two standard texts, in rough order of how gently they start:

- Leimkuhler, B. & Reich, S. (2005). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
  [doi:10.1017/CBO9780511614118](https://doi.org/10.1017/CBO9780511614118)
- Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*, 2nd ed. Springer.
  [doi:10.1007/3-540-30666-8](https://doi.org/10.1007/3-540-30666-8) — the reference for backward error
  analysis and why the energy error stays bounded.
- Sanz-Serna, J. M. (1992). Symplectic integrators for Hamiltonian problems: an overview. *Acta Numerica* **1**,
  243–286. [doi:10.1017/S0962492900002282](https://doi.org/10.1017/S0962492900002282) — a shorter survey if a
  whole book is too much.

The original papers for the methods here:

- Verlet, L. (1967). Computer "experiments" on classical fluids. *Physical Review* **159**, 98–103.
  [doi:10.1103/PhysRev.159.98](https://doi.org/10.1103/PhysRev.159.98)
- Ryckaert, J.-P., Ciccotti, G. & Berendsen, H. J. C. (1977). Numerical integration of the cartesian equations of
  motion of a system with constraints: molecular dynamics of n-alkanes. *Journal of Computational Physics* **23**,
  327–341. [doi:10.1016/0021-9991(77)90098-5](https://doi.org/10.1016/0021-9991%2877%2990098-5) — SHAKE.
- Andersen, H. C. (1983). RATTLE: a "velocity" version of the SHAKE algorithm for molecular dynamics
  calculations. *Journal of Computational Physics* **52**, 24–34.
  [doi:10.1016/0021-9991(83)90014-1](https://doi.org/10.1016/0021-9991%2883%2990014-1)
- Yoshida, H. (1990). Construction of higher order symplectic integrators. *Physics Letters A* **150**, 262–268.
  [doi:10.1016/0375-9601(90)90092-3](https://doi.org/10.1016/0375-9601%2890%2990092-3)

## Development

```bash
pytest
```

```bash
ruff check .
```

## AI Usage
AI has been used to design a scaffolding for the project, design tests and correct bugs and the documents. Most of 
the code and the text from the docs is original.

## License

MIT
