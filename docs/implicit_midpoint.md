# Implicit Midpoint

## Introduction

Everything up to here has leaned on the Hamiltonian splitting into $T(p) + V(q)$. Sometimes it does not. A mass
matrix that depends on where you are, $M(q)$, is the usual culprit, and a double pendulum written in its joint
angles is the example everyone meets first. Velocity-dependent forces do the same thing.

When $H$ does not split, Hamilton's equations become

$$
\dot q = \frac{\partial H}{\partial p}(t, q, p), \qquad \dot p = -\frac{\partial H}{\partial q}(t, q, p)
$$

and both right-hand sides now depend on *both* variables. That is what kills the explicit methods. There is no
order in which you can do the two updates that avoids needing a value you have not computed yet: updating $q$
wants the current $p$, updating $p$ wants the current $q$, and there is no other way around it. The drift and
kick we were composing no longer exist as separate exactly-solvable pieces.

So we have to give something up, and what we give up is explicitness.

## The method

The implicit midpoint rule evaluates the derivatives at the *midpoint* of the step, which is halfway between
where we are and where we are going:

$$
q_1 = q_0 + h\,\frac{\partial H}{\partial p}\Bigl(t_{1/2},\ \frac{q_0+q_1}{2},\ \frac{p_0+p_1}{2}\Bigr),
\qquad
p_1 = p_0 - h\,\frac{\partial H}{\partial q}\Bigl(t_{1/2},\ \frac{q_0+q_1}{2},\ \frac{p_0+p_1}{2}\Bigr)
$$

with $t_{1/2} = t_0 + h/2$. Notice that $q_1$ and $p_1$ appear on both sides. This is not a formula that computes
the new state, it is an equation that the new state satisfies, and that is exactly what "implicit" means. We do
not evaluate it, we solve it.

It is worth noticing that we already did the same thing in SHAKE: there $q_1$
depended on an unknown $\lambda$, and we solved $g(q_1(\lambda)) = 0$ rather than evaluating anything. Same
situation, larger unknown.

### 1. Solving for the midpoint

Rather than solving for $q_1$ and $p_1$ directly it is tidier to solve for the midpoint itself. Writing
$q_{1/2} = (q_0+q_1)/2$ and likewise for $p$, the pair above becomes

$$
q_{1/2} = q_0 + \frac{h}{2}\frac{\partial H}{\partial p}(t_{1/2}, q_{1/2}, p_{1/2}), \qquad
p_{1/2} = p_0 - \frac{h}{2}\frac{\partial H}{\partial q}(t_{1/2}, q_{1/2}, p_{1/2})
$$

which is the same number of unknowns with a cleaner shape: each line says the midpoint equals the starting point
plus half a step taken *at the midpoint*. Reading it as an assignment and repeating gives a fixed-point
iteration, starting from $q_{1/2} = q_0$, $p_{1/2} = p_0$ and stopping when the iterate stops moving by more
than `tol`.

This converges as long as $h$ is small enough, which is the same condition you already need for the method to be
accurate, so in practice it is not a real restriction. It also needs no derivatives of the derivatives: no
Jacobian, no linear solve, nothing but repeated evaluation. That matters more than it looks, because if the
partials are themselves finite differences then a Jacobian would be a finite difference of a finite difference,
which is not worth having.

### 2. Recovering the step

Once the midpoint has settled, the endpoint is undoing the transformation we had done:

$$
q_1 = 2q_{1/2} - q_0, \qquad p_1 = 2p_{1/2} - p_0
$$

One important note, the symplecticity belongs to the *converged* fixed point, not to the
iteration. If you stop early you have an explicit method, so if the method does not converge you cannot simply
recover the last iteration as it would not be symplectic.

## The cost and the tolerance

Every iteration evaluates both partials, and a step takes several iterations, so this is much more expensive than
anything in the explicit family. On the non-separable test problem it works out at about 12 gradient evaluations
per step, against Verlet's 2 force evaluations. That is the price of dropping separability, and it is the reason
the library refuses to run an explicit method on a general system rather than letting you sit and wait for 
much longer than you would need to.

There is one interaction worth knowing about. If you gave only `hamiltonian` and let the library build the
partials by finite differences, those partials are accurate to roughly $10^{-11}$, not to machine precision. Evidently,
a `tol` lower than this cannot be resolved, so the method raises the floor and gives a warning. 
If you need a lower tolerance you are going to have to give the exact `dHdq` and `dHdp`.

## Notation in the code

The implementation is `ImplicitMidpoint.step`, in `src/symplecta/methods.py`.

| here | in the code | note |
| --- | --- | --- |
| $\partial H/\partial q$, $\partial H/\partial p$ | `system.dHdq`, `system.dHdp` | exact if you gave them, finite differences otherwise |
| $t_{1/2}$ | `tmid` | |
| $q_{1/2}$, $p_{1/2}$ | `qmid`, `pmid` | the iterate, started at $q_0$, $p_0$ |
| the new iterate | `qn`, `pn` | both computed from the old one before either is replaced |
| stopping test | `change`, `self.tol` | how far the iterate moved, not a residual |
| iteration cap | `self.max_iter` | exceeded means raise, never return early |
| the reflection | `2 * qmid - q, 2 * pmid - p` | the last line of the step |

Both lines of the loop read the same `qmid, pmid` and only then overwrite them.
