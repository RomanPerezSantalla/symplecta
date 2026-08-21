# Constraint Methods

## Introduction

Suppose you now want to solve the equations of motion of a constrained system. Think for example of a molecule with
rigid bonds which preserve the distance between the atoms. Suppose also that we want a solution with no energy
drift, and thus we require a symplectic method. This is precisely what
[SHAKE](https://doi.org/10.1016/0021-9991%2877%2990098-5) 
and [RATTLE](https://doi.org/10.1016/0021-9991%2883%2990014-1) were conceived to tackle.

For previous methods, we really did not need to know where the equations of motion come from, but for these 
constrained methods I feel a bit of background is quite useful. In physics, one of the governing principles is the 
principle of stationary action. Simplifying a lot, the principle defines an action functional (think of a function
that instead of taking in a countable set of numbers takes in an actual function representing a path) and 
establishes that the "solution" path is such that the variation of the action when you "nudge" that path out of
position is 0. More formally:

$$S[q] = \int_{t_0}^{t_1} L\bigl(q(t), \dot q(t), t\bigr)\,dt \quad \text{and} \quad (\delta S) = 0 \text{ for all 
variations which vanish at the endpoints}$$

Here $L = T-V$ is the lagrangian. From this condition, the famous Euler-Lagrange equations are derived,
from which the equations of motion can be obtained. These equations are:

$$\frac{d}{dt}\frac{\partial L}{\partial\dot q}
-\frac{\partial L}{\partial q}=0$$

Now, if we want to add constraints $g(q(t)) = 0$ we have to work directly with the action, as the paths $q(t)$
are the quantity being constrained.

$$\tilde{S}[q, \lambda] = S[q] - \int_{t_0}^{t_1} \lambda(t)^{\top} g(q(t))dt = 
\int_{t_0}^{t_1}\Bigl[\underbrace{L(q,\dot q,t) - \lambda^{\top}g(q)}_{\textstyle \tilde L}\Bigr]dt$$

The $\lambda$ going into the integral (which is what allows us to end up with $\tilde L$ which is just a modified
lagrangian) feels a little tricky. It's great, because it lets us use the Euler-Lagrange equations, but it does
feel (at least to me) a little too convenient. For me the way to see it clearly is by discretizing the functional 
into a function by turning $q(t)$ into a collection of snapshots in time $q_k$ and applying lagrange multipliers 
as you would in a regular optimization problem. That is:

$$S \;\approx\; \sum_{k} L\!\left(q_k,\ \frac{q_{k+1}-q_k}{h}\right) h$$

Now the constraint $g(q(t)) = 0$ is just $g(q_k) = 0$ for all values of $k$. Applying regular multipliers:

$$\tilde S \;\approx\; \sum_k \Bigl[\underbrace{L_k - \lambda_k^{\top}g(q_k)}_{\textstyle \tilde L_k}\Bigr]h$$

So once again we end up with this modified lagrangian. Evidently this is not the most rigorous proof, but 
it does help me see why this constrained equations of motion problem just requires adding lagrange multipliers
to the standard lagrangian. So, with this out of the way and by solving the Euler-Lagrange equations for $\tilde L$,
we obtain (after using the definition of momentum $p = \partial L/\partial\dot q = M\dot q$ to split the second-order
equation into a first-order pair):
$$
\dot q=M^{-1}p, \qquad \dot p=F-G^{\top}\lambda
$$
Where $G$ is the Jacobian $\partial g/\partial q$, $M$ is the mass matrix and $F$ is the conservative force.

So in the end, the constraints are just a force we introduce. Going back to the molecule example, this
makes a lot of sense, as we are just telling the system that there is something bonding the atoms in those positions,
and it makes perfect sense for that 'something' to be a force.

There is an extra detail we have to be careful with, as in the same way as regular optimization, when we introduce
the lagrange multipliers they "turn into variables" of the function to be optimized. Therefore, we also have to
solve Euler-Lagrange with respect to the multipliers, that is:

$$
\frac{d}{dt}\frac{\partial \tilde L}{\partial\dot \lambda} -\frac{\partial \tilde L}{\partial \lambda}=0
$$

And since $\tilde L$ contains no $\dot \lambda$:
$$
\frac{\partial \tilde L}{\partial \lambda} = - g(q(t)) = 0
$$
A small detail which confused me for some time. We imposed $g(q)=0$ and have now derived it. That seems a bit circular.
However, we began with a constrained problem in $q$ (stationarity in $S$) and now have an unconstrained one in 
$q$ and $\lambda$ (stationarity in $\tilde S$). The constraint had to reappear as an equation, 
or the two would not be equivalent. Notice also that without this equation, $\lambda$ would be a vector of
completely free functions. This is easy to verify as the $S$ problem had $n$ equations and $n$ unknown functions,
but after adding $\lambda$ we have $n+m$ unknown functions and Euler-Lagrange in $q$ still provides only $n$ equations.
$g(q)=0$ recovers exactly the $m$ remaining equations to have a determinate system.

Appending this last equation to the previous first-order pair, we get the set of equations which defines the
problem to solve:

$$
\dot q=M^{-1}p, \qquad \dot p=F-G^{\top}\lambda, \qquad g(q(t)) = 0
$$

Notice also that there is an additional equation we can get by differentiating $g(q(t))=0$ with respect to time
(this will be needed later for RATTLE):

$$\frac{d}{dt}g\bigl(q(t)\bigr)=G(q)\dot q=G(q)M^{-1}p=0 .$$

Keep in mind that this equation requires no extra assumptions, it is established directly by the constraints.

## SHAKE

As with any iterative method, we describe a single step. We will repeat this step until we reach the defined
final time $t_f$.

### 1. Discretising with velocity Verlet

Applying velocity Verlet over $[t_0,t_1]$, $t_1=t_0+h$ for the set of 3 equations we derived at the 
end of the last section:

$$
p_{1/2}=p_0+\tfrac h2\bigl[F(t_0,q_0)-G(q_0)^{\top}\lambda\bigr],\quad
q_1=q_0+hM^{-1}p_{1/2},\quad
0=g(q_1),\quad
p_1=p_{1/2}+\tfrac h2\bigl[F(t_1,q_1)-G(q_1)^{\top}\mu\bigr]
$$

Here the second "kick" for the momentum introduces a new multiplier (remember that we had to apply the multipliers
to every instance of time, so they are applied at every instance of the discretization). SHAKE just takes $\mu = 0$
and this is exactly what RATTLE will correct.

### 2. Solving for $\lambda$

Substituting $p_{1/2}$ into $q_1$:

$$q_1=\underbrace{q_0+hM^{-1}\Bigl(p_0+\tfrac h2F(t_0,q_0)\Bigr)}_{\textstyle q_1^{\mathrm{unc}}}
\;-\;\underbrace{\tfrac{h^{2}}{2}M^{-1}G(q_0)^{\top}}_{\textstyle B\in\mathbb R^{n\times m}}\lambda .$$

So

$$\boxed{\,q_1(\lambda)=q_1^{\mathrm{unc}}-B\lambda\,}$$

where $q_1^{\mathrm{unc}}$ stands for unconstrained, as it is the position Verlet would have produced with no 
constraints.

Substituting into the constraint $g(q_1)=0$:

$$\Phi(\lambda):=g\bigl(q_1^{\mathrm{unc}}-B\lambda\bigr)=0$$

This is what we actually solve in each step.

### 3. Newton step on $\Phi(\lambda)$

By the chain rule (and remembering that $G = \partial g / \partial q$):

$$\frac{\partial\Phi(\lambda)}{\partial \lambda}=
G\bigl(q_1(\lambda)\bigr)\cdot\frac{\partial q_1}{\partial\lambda}
=-\,G\bigl(q_1(\lambda)\bigr)\,B\;\in\mathbb R^{m\times m}$$

Newton's update is $\lambda_{k+1}=\lambda_k-[\Phi'(\lambda_k)]^{-1}\Phi(\lambda_k)$:

$$\lambda_{k+1}=\lambda_k+\bigl[G(q_1^{(k)})B\bigr]^{-1}g\bigl(q_1^{(k)}\bigr),
\qquad q_1^{(k)}=q_1^{\mathrm{unc}}-B\lambda_k,$$

So it is clear this is a regular system of equations as long as we know the previous $\lambda_k$.
For $k=0$, we start with the unconstrained case $\lambda_0 = 0$.
We can now expand $B$ to figure out what we actually need to 
solve.

$$G(q_1)B=\tfrac{h^{2}}{2}\,G(q_1)M^{-1}G(q_0)^{\top}$$

In the code we set up options to solve it both exactly and through an approximation which allows for the use of
Cholesky factorisation. Take also into account that although it seems we are taking the inverse of matrix
$G(q_1)B$, in practice in the code we do not compute it, solving instead the system of equations. This is
because in general, numerically, it is a bad idea computing the inverse if it can be avoided.
It is, in fact, the first of Nicholas Higham's "seven sins" of numerical linear algebra.
[You can read more about the reason here](https://nhigham.com/2022/10/11/seven-sins-of-numerical-linear-algebra/)

#### `newton="exact"`

Use that matrix as is. Since $q_1\neq q_0$ it is **not symmetric**:
$(G_1M^{-1}G_0^{\top})^{\top}=G_0M^{-1}G_1^{\top}$. We must solve it through regular LU. Also, $G$ has to be
reevaluated at every iteration. The convergence is quadratic.

#### `newton="frozen"` (default)

Approximate $G(q_1)\approx G(q_0)$, giving

$$A:=\tfrac{h^{2}}{2}\,G(q_0)M^{-1}G(q_0)^{\top}.$$

$A$ is symmetric, and for $x\neq0$

$$x^{\top}Ax=\tfrac{h^{2}}{2}\bigl(G_0^{\top}x\bigr)^{\top}M^{-1}\bigl(G_0^{\top}x\bigr)>0
\iff G_0^{\top}x\neq0,$$

Since $M$ is a mass matrix it is positive definite, and so is its inverse. If we rewrite $G_0^{\top}x = y$, 
it is easy to check that $A\succ0$ **iff $G(q_0)$ has full row rank** (meaning the constraints are independent). 
This means we can solve the system through Cholesky factorisation which is faster. Furthermore, 
as we get rid of the dependence on 
$\lambda$, we can compute the matrix once per step instead of once per iteration. In exchange,
the convergence is linear instead of quadratic. *Remember, a step refers to 
the calculation for each of the $q_k$, an iteration is however many times we have to update the Newton method for
convergence to the given tolerance. That is, there are multiple iterations per step.*

### 4. Recovering $q_1$ and $p_1$

Once we have $\lambda^{\star}$ within tolerance, we just have to substitute in the original equations for $q_1$ and
$p_1$ which are our outputs for the step:

$$q_1=q_1^{\mathrm{unc}}-B\lambda^{\star},\qquad
p_{1/2}=p_0+\tfrac h2F(t_0,q_0)-\tfrac h2G(q_0)^{\top}\lambda^{\star},\qquad
p_1=p_{1/2}+\tfrac h2F(t_1,q_1).$$

### 5. Notation in the code

The implementation is `Shake.step`, in `src/symplecta/methods.py`. It follows the steps above in order, so the
two can be read side by side. 

| here | in the code | note                                                                       |
| --- | --- |----------------------------------------------------------------------------|
| $g$ | `system.constraints.g` | the constraint function you supply                                         |
| $G$, $G(q_0)$ | `system.constraints.jac`, `g0` | the Jacobian; `g0` is evaluated once per step, at $q_0$                    |
| $F$ | `system.force` |                                                                            |
| $M^{-1}p$ | `system.velocity(p)` | a divide or a Cholesky solve; $M^{-1}$ is never formed. Remember the sins! |
| $q_1^{\mathrm{unc}}$ | `q_unc` | the position unconstrained Verlet would have produced                      |
| $B$ | `shift` | converts a multiplier into the position shift it causes                    |
| $A$, factored | `_chol` | `None` on the `exact` path, where the matrix moves every iteration         |
| $\lambda$, $\lambda_k$ | `lam` | accumulated across iterations, not recomputed                              |
| $g(q_1^{(k)})$ | `residual` | how badly the constraint is currently violated                             |
| $q_1(\lambda_k)$ | `q1` |                                                                            |
| $h$, tolerance, iteration cap | `h`, `self.tol`, `self.max_iter` |                                                                            |

A small thing to keep in mind, `lam` is updated by adding to itself
rather than being reassigned, because $\lambda_k$ carries over between iterations.

## RATTLE

Once again we describe a single step of the method. As we introduced earlier, the difference is that we will not 
set $\mu=0$, and we will reintroduce the constraint we were missing in SHAKE ($G(q)M^{-1} p = 0$) to solve for it.

### 1. Reusing SHAKE
Let's start from the equations which we used for SHAKE:

$$
p_{1/2}=p_0+\tfrac h2\bigl[F(t_0,q_0)-G(q_0)^{\top}\lambda\bigr],\quad
q_1=q_0+hM^{-1}p_{1/2},\quad
0=g(q_1),\quad
p_1=p_{1/2}+\tfrac h2\bigl[F(t_1,q_1)-G(q_1)^{\top}\mu\bigr]
$$

We can now see that the new multipliers $\mu$ only appear for the very final calculation of $p_1$. We can 
use SHAKE to get both $q_1$, which is already the final output for the positions and what 
we will call $p_1^{\mathrm{unc}}$. The name becomes evident when we split the $\mu$ equation in the same
way we had split the $\lambda$ equation in the SHAKE section.

$$p_1=\underbrace{p_{1/2}+\tfrac h2F(t_1,q_1)}_{\textstyle p_1^{\mathrm{unc}}}- 
\underbrace{\tfrac h2G(q_1)^{\top}}_{\textstyle \mathbb R^{n\times m}}\mu $$

We are just missing the final half-step for the momentum. This is also what we do in the code, 
we call the SHAKE step within the RATTLE step to get $q_1$ and $p_1^{\mathrm{unc}}$.

### 2. Final half-kick

We now substitute $p_1$ into the missing constraint:

$$G_1M^{-1}\Bigl[p_1^{\mathrm{unc}}-\tfrac h2G_1^{\top}\mu\Bigr]=0
\qquad\Longrightarrow\qquad
\tfrac h2\,G_1M^{-1}G_1^{\top}\,\mu=G_1M^{-1}p_1^{\mathrm{unc}}
\qquad\Longrightarrow\qquad
\mu = \bigl[\tfrac h2 G_1M^{-1}G_1^{\top} \bigr]^{-1} G_1M^{-1}p_1^{\mathrm{unc}}
$$

And once again we are faced with a linear system, but this time, the matrix of the system is symmetric and 
positive definite with no approximation needed. This means we can always use Cholesky directly.
Once we have $\mu$, we just update $p_1$ and return $q_1$, $p_1$.

### 3. Comparison with SHAKE

The whole point of RATTLE is the constraint SHAKE leaves out, so that is what we should compare. Integrating the
rigid pendulum released from 50 degrees, for 20 time units with $h=0.01$:

| | max $\lvert g(q)\rvert$ | max $\lvert G(q)M^{-1}p\rvert$ | force evaluations |
| --- | --- | --- | --- |
| SHAKE | 9.99e-12 | **1.682e-01** | 4000 |
| RATTLE | 9.89e-12 | **4.441e-16** | 4000 |

Both hold the position constraint to the tolerance we asked for, as they should, (remember that $q_1$ comes
from the same place). The velocity constraint is another story.

Notice also that RATTLE has an error of machine precision on the constraint instead of `tol` as there is no
iteration involved when solving the final half-kick system exactly.

| $h$ | 0.04 | 0.02 | 0.01 |
| --- | --- | --- | --- |
| SHAKE | 6.730e-01 | 3.364e-01 | 1.682e-01 |
| RATTLE | 6.661e-16 | 4.441e-16 | 4.441e-16 |

Halving $h$ halves SHAKE's violation, so the error is $O(h)$ and it does vanish in the limit, SHAKE is
simply first order in this quantity. RATTLE does not depend on $h$ here at all.

That $O(h)$ error is noticeable in the momentum. Here is the error for the pendulum in $p$ at $t=5$:

| $h$ | 0.008 | 0.004 | 0.002 | 0.001 | fitted order |
| --- | --- | --- | --- | --- | --- |
| SHAKE | 4.49e-02 | 2.26e-02 | 1.13e-02 | 5.67e-03 | 0.99 |
| RATTLE | 1.84e-03 | 4.60e-04 | 1.15e-04 | 2.87e-05 | 2.00 |

So SHAKE is second order in position but only **first** order in momentum, while RATTLE is second order in both.

As RATTLE's extra stage costs one more Jacobian evaluation and one small solve per step and nothing else, 
there is not much reason to prefer SHAKE since you usually also need the $p$ coordinates. If you don't need them
at all you could make an argument for using SHAKE but if in doubt, just default to RATTLE.

### 4. Notation in the code

`Rattle.step` is short, because stage one is a call to `Shake.step` and everything in SHAKE's table above still
applies. These are the names that only appear here:

| here | in the code | note |
| --- | --- | --- |
| $p_1^{\mathrm{unc}}$ | `p1_unc` | what `Shake.step` returns as its momentum, i.e. the unprojected half-kick |
| $G_1 = G(q_1)$ | `g1` | the Jacobian again, but at the *new* position, so it is re-evaluated |
| $M^{-1}G_1^{\top}$ | `minv_g1t` | built column by column, since `velocity` expects a 1-D momentum |
| $\tfrac h2 G_1M^{-1}G_1^{\top}$ | `a` | the system matrix, factored by the same helper SHAKE uses |
| $G_1M^{-1}p_1^{\mathrm{unc}}$ | `rhs` | the violation being cancelled |
| $\mu$ | `mu` | one solve, no loop, so there is no iterate to accumulate |

Notice there is no loop, no `tol` and no `newton` in this stage as all of those belong to stage one, and
`Rattle` only accepts them because it inherits from `Shake`.

