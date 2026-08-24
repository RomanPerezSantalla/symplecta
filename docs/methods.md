# General Methods

These are the methods for the unconstrained, separable case, which is the one most systems fall into. All of them
are built out of the same two moves, so it is worth naming them before we start. Splitting $H = T(p) + V(q)$ and
looking at each half on its own:

$$
\underbrace{\dot q = \frac{p}{m}, \quad \dot p = 0}_{\textstyle \text{drift, from } T(p)}
\qquad\qquad
\underbrace{\dot q = 0, \quad \dot p = F(q)}_{\textstyle \text{kick, from } V(q)}
$$

Each of these is simple to solve. During a drift the momentum does not change, so the position moves in a
straight line. During a kick the position does not change, so the momentum changes linearly. Neither is an
approximation, they are the exact solutions of their own halves, for any step size.

That is the whole trick behind everything below. The methods below are just different ways of alternating drifts
and kicks, and they differ only in the step size and the number of steps taken.

## Euler method

The simplest thing you can do is one kick and one drift. There are two ways round, and both are used, so the
library keeps both behind a `variant` argument:

$$
\underbrace{p_1 = p_0 + h\,F(t_0, q_0), \qquad q_1 = q_0 + h\frac{p_1}{m}}_{\textstyle \texttt{"pq"}}
\qquad
\underbrace{q_1 = q_0 + h\frac{p_0}{m}, \qquad p_1 = p_0 + h\,F(t_1, q_1)}_{\textstyle \texttt{"qp"}}
$$

The important thing in both is that the second line uses the value the first line just produced, not the value
we started with. That single choice is what separates this from the regular Euler method of the previous
document, and it is what makes it symplectic.

Notice the time arguments are not the same in the two variants. In `"pq"` the kick happens before anything has
moved, so the force is evaluated at $t_0$. In `"qp"` the drift happens first and moves the clock along, so
by the time we kick we are at $t_1 = t_0 + h$.

This method is first order and costs one force evaluation per step.

## Velocity Verlet

Verlet takes the same two moves and makes them symmetric: half a kick, a full drift, then the other half of the
kick.

$$
p_{1/2} = p_0 + \frac{h}{2}F(t_0, q_0), \qquad
q_1 = q_0 + h\frac{p_{1/2}}{m}, \qquad
p_1 = p_{1/2} + \frac{h}{2}F(t_1, q_1)
$$

The symmetry is important. Running the step backwards gives you the same step again, and a method with that
property cannot have an error term of odd order, because an odd term would change sign when you reverse time and
break the symmetry. So the first order that survives is the second, and Verlet is second order for the price of
one extra force evaluation over Euler.

Notice that the final kick sits at $t_1$, not $t_0$, same as in the `"qp"` variant above.

Two force evaluations per step, second order.

## Yoshida

The natural question after Verlet is whether we can keep going, and we can. The idea is to run a second-order
method three times in a row with carefully chosen substep sizes so that the leftover third-order error from one
substep cancels against the others. Writing it down fully, alternating drifts and kicks, one Yoshida step is:

$$
\begin{aligned}
&\text{drift } c_1 h \quad \text{kick } w_1 h \quad \text{drift } c_2 h \quad \text{kick } w_0 h
\quad \text{drift } c_2 h \quad \text{kick } w_1 h \quad \text{drift } c_1 h
\end{aligned}
$$

with

$$
w_1 = \frac{1}{2 - \sqrt[3]{2}} \approx 1.35121, \qquad
w_0 = \frac{-\sqrt[3]{2}}{2 - \sqrt[3]{2}} \approx -1.70241, \qquad
c_1 = \frac{w_1}{2}, \qquad c_2 = \frac{w_0 + w_1}{2}
$$

A quick check that these are consistent: the kicks have to add up to one whole step, and $w_0 + 2w_1 = 1$
exactly, as do the drifts. In the code these are computed from the expressions above rather than pasted in as
decimals, which removes any chance of a transcription error.

### The negative substep

Look at $w_0$. It is negative, and not slightly: the middle substep runs **backwards in time** by about 1.7
steps, and the two outer ones overshoot forwards to compensate. Written as times, the three kicks land at

$$
t_0 + 0.675604\,h, \qquad t_0 + 0.5\,h, \qquad t_0 + 0.324396\,h
$$

which is not even in increasing order. The middle one is exactly $t_0 + h/2$, incidentally, which falls out of
$c_1 + c_2 = (w_0 + 2w_1)/2 = 1/2$.

This may look like a mistake when you first see it, however, no explicit composition of this kind can reach
an order above two with all its substeps positive. If you want fourth order out of Verlet substeps, 
some of them have to go backwards. The practical consequence is that intermediate states inside a Yoshida step
can be well outside anything physical, and only the state at the end of the full step means anything.

Three force evaluations per step, fourth order.

### Going higher

The composition that got us here, three copies of a symmetric method with the middle one reversed, was never
specific to Verlet. Given any symmetric method $S_{2k}$ of even order $2k$, the same three-substep sandwich

$$
S_{2k+2}(h) = S_{2k}(w_1 h)\; S_{2k}(w_0 h)\; S_{2k}(w_1 h), \qquad
w_1 = \frac{1}{2 - \sqrt[2k+1]{2}}, \qquad w_0 = 1 - 2 w_1
$$

raises the order by two. Feed it the fourth-order method, and you get sixth, feed it that and, you get eighth, for
however long you want. It is usually called the triple jump.

The trouble is the count. Every application triples the number of substeps, so the chain runs 3, 9, 27: nine
force evaluations for sixth order and twenty-seven for eighth. That is a steep price for two extra orders.

Yoshida's observation was that the triple jump is only one solution among many. Write down a general symmetric
composition of $2n+1$ substeps, ask what the weights must satisfy for the leading error terms to cancel, and you
get a system of polynomial equations; the triple jump is one root of it, not the only one. Solving the system
directly does much better. **Seven** substeps are enough for sixth order and **fifteen** for eighth, against the
triple jump's nine and twenty-seven.

What you give up is the closed form. Those weights are roots of a polynomial system with no expression in
radicals, so unlike $w_1 = 1/(2-\sqrt[3]{2})$ above they can only be written down as decimals, and the library
carries the fifteen digits Yoshida published. He found several distinct roots at each order and labeled them A,
B, C and so on; `yoshida6` and `yoshida8` both use solution A, which are the ones usually quoted. Only the outer
weights are stored. The middle one is fixed by $\sum_i w_i = 1$ and is computed from the others, which is also
the check that would catch a mistyped digit.

## Choosing between them

All five are measured in the test suite, on the harmonic oscillator with $h = 0.05$ integrated to $t = 1000$:

| method | order | force evaluations per step | energy band |
| --- | --- | --- | --- |
| symplectic Euler | 1 | 1 | 2.502e-02 |
| velocity Verlet | 2 | 2 | 3.125e-04 |
| Yoshida 4 | 4 | 3 | 2.382e-07 |
| Yoshida 6 | 6 | 7 | 3.095e-11 |
| Yoshida 8 | 8 | 15 | 2.670e-14 |

Read down the last column and the case for high order looks overwhelming. It is not that simple, because the
column is not a fair fight. Each step from a method of higher order costs more, so you can afford to lower $h$
in the methods of lower order, as at the end of the day, you care about compute time, not the order or the $h$.
Standardizing fot the comparison we get:

| method | $h$ | force evaluations per unit time | energy band |
| --- | --- | --- | --- |
| Yoshida 4 | 0.05 | 60 | 2.382e-07 |
| Yoshida 6 | 0.125 | 56 | 7.586e-09 |
| Yoshida 8 | 0.25 | 60 | 1.109e-09 |

Higher order still wins, but by a factor of thirty and then seven, rather than the thousands the first table
seemed to promise. The extra substeps eat most of what the extra order buys.

There is also a floor. The $2.670$e$-14$ in the first table is not Yoshida 8's error, it is double precision,
pointing at the fact that Yoshida 8 is basically always overkill. We have basically added it to the library 
because the coefficients are calculated in Yoshida's paper, and it was barely any extra effort, and in ways,
it is instructive.

The practical reading is:

- **Verlet** is the default for a separable system. Two force evaluations, symmetric, nothing to go wrong.
- **Yoshida 4** costs one more evaluation than Verlet and buys three orders of magnitude. It is the best value in
  the table and the one to move to as soon as Verlet is not enough.
- **Yoshida 6** is for long integrations where you want the answer to be accurate and not merely stable. Seven
  evaluations is a real cost, and it is worth paying.
- **Yoshida 8** only makes sense at coarse steps, where its fifteen evaluations don't reach machine precision. 
  If you are taking small steps in double precision it is fifteen evaluations spent to reach the same round-off floor
  Yoshida 6 reaches with seven.
- **Symplectic Euler** is in the library mainly because it is the easiest one to understand and to check by hand.
  You would rarely reach for it for a real problem.

## Notation in the code

They all live in `src/symplecta/methods.py` and follow the equations above line by line.

| here | in the code | note |
| --- | --- | --- |
| $F(t, q)$ | `system.force(t, q)` | the force you supply, $-dV/dq$ |
| $p/m$, or $M^{-1}p$ | `system.velocity(p)` | a divide or a Cholesky solve depending on the mass |
| a kick | `p = p + (...) * system.force(...)` | |
| a drift | `q = q + (...) * system.velocity(p)` | |
| `"pq"` / `"qp"` | `SymplecticEuler(variant=...)` | kick first, or drift first |
| $w_0, w_1, c_1, c_2$ | `Yoshida4.w0, w1, c1, c2` | class attributes, computed not pasted |
| $c_1 + 2c_2$ | `Yoshida4.k1` | the time of the third kick |
| $w_i$, the substep weights | `Yoshida6.weights`, `Yoshida8.weights` | the published decimals, middle one computed |
| $c_i$, the drifts between them | `Yoshida6.drifts`, `Yoshida8.drifts` | built once per class, not once per step |

The thing to notice when reading them is how simple they are. Not one of them has a tolerance or a convergence
check, because not one of them ever has to solve anything, every line is just an assignment. The loop the two higher
compositions run is over a fixed list of constants and always turns the same number of times. 
I wouldn't even consider them loops as we could just write it all out explicitly like Yoshida 4. This is why
separability is nice to have, and it is what the constrained and non-separable methods have to give up.
