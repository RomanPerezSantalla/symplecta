# Introduction

## Equations of Motion

This library is set up to primarily deal with the equations of motion in position and momentum. These equations
govern the movement of the system completely. These equations are usually derived either from the Euler-Lagrange
equations or from Hamilton's equations (technically you could also derive them from Newton's laws,
but for a complicated system it is a lot harder).

$$
\underbrace{\frac{d}{dt}\frac{\partial L}{\partial\dot q}-\frac{\partial L}{\partial q}=0}_{\textstyle \text{Euler-Lagrange}}
\qquad\qquad
\underbrace{\dot q=\frac{\partial H}{\partial p},\qquad \dot p=-\frac{\partial H}{\partial q}}_{\textstyle \text{Hamilton}}
$$

The usual notation for these equations is for $q$ to be the vector of generalized position coordinates
(generalized just means they need not be cartesian) and $p$ to be the vector of generalized momenta.
The derivatives with respect to time in this convention are many times written as a dot on top of the variable,
so $dq/dt := \dot q$.
Euler-Lagrange will produce $N$ 2nd order equations which then get transformed into $2N$ first order pairs and
Hamilton produces the $2N$ equations directly.

### Separable Hamiltonian

Notice the $H$ in Hamilton's equations. This is aptly called the Hamiltonian, and it is the total energy of the 
system. If the Hamiltonian can be written as $H(q, p) = T(p) + V(q)$, where $T$ is identified with the kinetic
energy and $V$ is identified with the potential energy, we call the Hamiltonian **separable**. The usefulness
of the Hamiltonian being separable is easy to see from Hamilton's equations:
$$
\dot p = - \frac{\partial H}{\partial q} = - \frac{\partial V(q)}{\partial q} \qquad
\dot q = \frac{\partial H}{\partial p} = \frac{\partial T(p)}{\partial p}
$$
The right-hand side of the $q$ equation has no $q$ in it, and the right-hand side of the $p$ equation has no $p$. 
So each update is a formula, not an equation to solve. If $H$ does not split this way, both right-hand sides depend on 
both $q$ and $p$, and there is no order in which you can update them that avoids needing the answer before you have it. 
You are left solving a system of equations at every step instead of evaluating a formula.

## Symplectic Methods

The name symplectic refers to a geometric property these methods share: they preserve areas in phase space exactly.
What matters here is the consequence we will reach rather than the definition which right now tells us nothing.
Let's start by seeing what the problem that needs fixing is. Suppose for simplicity that we are dealing with a 
single particle with position $q$ and momentum $p$. Suppose also that we are dealing with a separable system
and that:
$$
H = T(p) + V(q) = \frac{p^2}{2m} + V(q) 
$$
Let's do a regular Euler step with discretization $h$.
$$
q_1 = q_0 + h \frac{p_0}{m}, \qquad p_1 = p_0 + h F(q_0)
$$
Here $F$ is the force, and it is the derivative of the potential (with a minus sign), that is,
in 1 dimension, $F(q) = - dV / dq$.
Let's now calculate the energies for both steps:
$$
E_0 = \frac{p_0^2}{2m} + V(q_0), \qquad E_1 = \frac{p_1^2}{2m} + V(q_1) = \frac{p_0^2}{2m} + \frac{h^2 F(q_0)^2}{2m} + 
\frac{2hp_0F(q_0)}{2m} + V(q_1)
$$
And now we expand the potential with a Taylor series to 2nd order for the $q_1$ equation.
$$
E_1 = \frac{p_0^2}{2m} + \frac{h^2 F(q_0)^2}{2m} + 
\frac{hp_0F(q_0)}{m} + V(q_0) - \frac{hp_0F(q_0)}{m} + \frac{h^2p_0^2 V''(q_0)}{2m^2} = 
\frac{p_0^2}{2m} + \frac{h^2 F(q_0)^2}{2m} + V(q_0) + \frac{h^2p_0^2 V''(q_0)}{2m^2} 
$$

So now, taking the difference between $E_1$ and $E_0$:
$$
E_1 - E_0 = \Delta E = \frac{h^2 F(q_0)^2}{2m} + \frac{h^2p_0^2 V''(q_0)}{2m^2}
= \frac{h^2}{2} \Bigl[\frac{F(q_0)^2}{m} + \frac{p_0^2 V''(q_0)}{m^2} \Bigr]
$$
Now do the same with symplectic Euler.
$$
q_1 = q_0 + h \frac{p_1}{m}, \qquad p_1 = p_0 + h F(q_0)
$$
Notice that the difference is that the $q_1$ equation uses the updated momentum $p_1$ instead of the original
momentum $p_0$. If we do the same calculations for energy we end up getting:
$$
E_1^{\textrm{symp}} - E^{\textrm{symp}}_0 = \Delta E^{\textrm{symp}} = - \frac{h^2 F(q_0)^2}{2m} + 
\frac{h^2p_0^2 V''(q_0)}{2m^2} = \frac{h^2}{2} \Bigl[- \frac{F(q_0)^2}{m} + \frac{p_0^2 V''(q_0)}{m^2} \Bigr]
$$
Notice the additional negative sign on the first term.

So far these are changes over a *single* step. What we actually care about is what happens over many steps,
so we add them up. After $N$ steps the total change in energy is $\sum_n \Delta E_n$, one term per step. 
Writing the bracket as $B$, and using the fact that each step advances time by $h$, that sum is a Riemann sum:

$$
\sum_{n} \Delta E_n = \frac{h^2}{2}\sum_n B_n = \frac{h}{2}\sum_n B_n\,h \;\approx\; \frac{h}{2}\int_0^T B\,dt
$$

So whether the energy drifts over an orbit is exactly the question of what the previous integral does.
Both the symplectic and non-symplectic cases contain the same two terms, 
so let's take the integral with respect to time of the second one:
$$
\int_0^T V''(q)\,\frac{p^2}{m^2}\,dt \;=\; \int_0^T V''(q)\,\dot q^2\,dt
$$

using $p/m = \dot q$. Now, because of the chain rule, notice that  $V''(q)\,\dot q = \dfrac{d}{dt}V'(q)$, 
so the integrand is $\dot q \cdot \dfrac{d}{dt}V'(q)$. Integrate by parts:

$$\int_0^T \dot q \,\frac{d}{dt}V'(q)\,dt
= \underbrace{\Bigl[\dot q\,V'(q)\Bigr]_0^T}_{=\,0\ \text{on a closed orbit}} - \int_0^T \ddot q\,V'(q)\,dt$$

The boundary term vanishes because $\dot q$ and $V'(q)$ both come back to their starting values, as $V$ does not
depend on time and the definition of closed orbit is:
$$
q(T) = q(0) \qquad\textbf{and}\qquad \dot q(T) = \dot q(0)
$$
Substitute $\ddot q = F/m$ and $V' = -F$, and we get the final result:

$$
\int_0^T V''(q)\,\frac{p^2}{m^2}\,dt = \int_0^T \frac{F^2}{m}\,dt
$$

Both integrals are therefore the same number, and that settles both cases at once. For symplectic Euler the
bracket carries a minus sign in front of the first term, so the two cancel and the sum over an orbit vanishes:
the energy moves at every step, but the moves undo each other, and it stays inside a band. 
It is a band rather than exact conservation because the cancellation only completes at the end of a full orbit, 
at each step, the integral need not be 0.
For regular Euler the two are added instead, so they double, and since $F^2/m$ is a square divided by a mass the 
result is not merely non-zero but strictly positive:

$$
\text{energy gained per orbit} \;=\; h\int_0^T \frac{F^2}{m}\,dt \;>\; 0
$$

Every orbit adds the same positive amount, which is what a drift is.

One could think then that the solution would be to just use a smaller step, but the total drift is clearly
proportional to $h\times \textrm{run length}$ so doubling the run brings back the error.

The closed orbit above is the *exact* one, while the sum we are approximating
runs along the numerical trajectory. The two agree while the method is still following the true solution, which
is the only regime in which it is doing its job anyway. What this argument shows, then, is that even while the
regular method is tracking the solution perfectly well, it is quietly bleeding energy on every orbit.

### Summary and clarifications

What we tried to show with these derivations is the need for symplectic methods. One should not consider the 
previous derivations as a proof of anything, just as some napkin math to show the idea. For example, we truncated
the Taylor series at order 2 so we have no idea what goes on from $O(h^3)$ upwards. The idea was for me to try and show 
the failure of non-symplectic methods to conserve energy over long times, 
more so than showing that symplectic methods work.

Symplectic methods in general preserve *exactly* a nearby Hamiltonian to the original one
(think of this new Hamiltonian as $H + O(h^k)$ with $k$ being the method's order) and that is what keeps the energy 
bounded in a band around the original.

Therefore, to answer the original question in this section, the problem is that standard methods do not conserve
energy. If you are integrating for a long time, and you care about the energy, use one of these methods; 
if you only need a short accurate trajectory, a standard solver is fine.