"""symplecta: symplectic and geometric integrators for Hamiltonian systems.

Fixed-step, structure-aware integrators intended to complement
``scipy.integrate.solve_ivp``.
"""

from symplecta.ivp import (
    Constraints,
    GeneralSystem,
    SeparableSystem,
    SymplecticResult,
    solve_symplectic_ivp,
)

__version__ = "0.0.1.dev0"

__all__ = [
    "Constraints",
    "GeneralSystem",
    "SeparableSystem",
    "SymplecticResult",
    "__version__",
    "solve_symplectic_ivp",
]
