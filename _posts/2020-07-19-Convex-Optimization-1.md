---
title: "Convex Optimization 1"
date: 2020-07-19 00:00:28 -0400
categories: convex optimization
use_math: true
---

# Why Convex Optimization is important?

Convex optimization is not as useful as before in machine learning, because the problems have changed.
Then why study **Convex Optimization**? Optimization is really important in machine learning now, just not convex optimization. 

I will write about below contents.

1. Gradient Descent, Stochastic Gradient Descent.
2. Proximal Gradient Descent.
3. Linear programming, Duality and KKT conditions.
4. Newton's Method, Quasi-Newton, Primal-Dual interior point method.
5. Linear system sovler: QR and Cholesky decomposition, Jacobi and Gauss-Seidl iterations.
6. Coordinate Descent, Dual Coordinate Descent, ADMM.
7. Frank-Wolfe (projection free) method.
8. Stochastic variance reduction techniques: SVRG, SAG, SAGA.

Those are convex optimization algorithms, but the real-world problems in machine learning nowadays are mostly non-convex. Adaptive algorithms like Adagrad, accelerated method like the
Nestrov's momentum and Katyusha momentum, they are also designed for convex optimization, yet they (or their variants) can be used in non-convex optimization as well.

Optimization algorithm: Solve
$$
min_{x \in \mathcal{D}}f(x)
$$
as fast as possible and as good as possible.

Keywords: fast and good(tradeoff between them).

**Definition**: convex set

A convex set $\mathcal{D}$ is a non-empty set if  $(1-\lambda)x + \lambda y \in \mathcal{D}$ for every $x, y \in \mathcal{D}$, for every $\lambda \in (0,1)$

**Definition**: convex function

A function $f$ over a convex set $\mathcal{D}$ is convex if $f((1-\lambda)x + \lambda y) \leq (1-\lambda)f(x) + \lambda f(y)$

**property of a convex function**

For every differentiable convex function $f$ over a convex set $\mathcal{D}$, for every point $x, y \in \mathcal{D}$, $f(y) \geq f(x) + <\nabla f(x), y-x>$

proof) Suppose there is an  $y \in \mathcal{D}$ s.t $f(y) \leq f(x) + <\nabla f, y-x>-\delta$ for $\delta>0$.

Then, by convexity(definition), for every $\lambda \in (0,1)$, 

$$
\begin{align*}
f((1-\lambda)x + \lambda y) &= f(x+\lambda(y-x))\\
&\leq (1-\lambda)f(x) + \lambda f(y) \\ 
&\leq (1-\lambda) f(x) + \lambda f(x)+\lambda <\nabla f, y-x>-\lambda \delta \\
&= f(x) + \lambda <\nabla f, y-x>-\lambda \delta \\
\end{align*}
$$


It implies that
$$
\frac{f(x+\lambda(y-x))}{\lambda} <\nabla f(x), y-x> -\delta
$$
Let $\lambda \rightarrow 0^{+}$, then, $<\nabla f(x), y-x> <\nabla f(x), y-x> -\delta$

Contradiction!



**Theorem**

When there is no local minima, for a 1st order differentiable convex function $f$, $\nabla f(x^{\star}) = 0 \Leftrightarrow f(x^{\star}) =min_{x}f(x)$

proof) $(\rightarrow)$ for every $y$, and let $x=x^{\star}$, then $f(y) \geq f(x^{\star}) + <\nabla f(x^{\star}), y-x> = f(x^{\star})$.

($\leftarrow$) will be treated below.

For general Lipchitz convex function, $\exists y \in \partial f(x^{\star}), y=0 \Leftrightarrow f(x^{\star}) =min_{x}f(x)$



**QUESTION** How do we find the minimizer and a point that has small gradient?

**Anwser**: **Gradient Descent**: $x_{t+1} = x_{t}-\eta\nabla f(x_{t})$ for $t=0,1,2, \dots$

$x_0$ : starting point, $\eta$: step size

BUT WHY DOES IT WORK?, HOW TO CHOOSE $\eta$?

**Definition**: smoothness

a 1st order differentiable function (not necessarily convex) $f$ over a set (not necessarily convex) $\mathcal{D}$ is called L smooth for L > 0 if
$$
f(y) \leq f(x) + <\nabla f(x), y-x>+\frac{L}{2}||y-x||_{2}^{2}
$$
And sum of two L-smooth functions is a 2L-smooth function.

proof) let $f = f_1 + f_2$ and $f_1, f_2$ are L-smooth functions.

$$
\begin{align*}
f(y) &= f_1(y) +f_2(y) \\
&\leq f_1(x) + <\nabla f_1(x), y-x>+\frac{L}{2}||y-x||_{2}^{2} + f_2(x) + <\nabla f_2(x), y-x>+\frac{L}{2}||y-x||_{2}^{2} \\
&= f(x) + <\nabla f(x), y-x>+\frac{L}{2}||y-x||_{2}^{2}
\end{align*}
$$


Recall
$$
f(y) \geq f(x) + <\nabla f(x), y-x>
$$
$\Rightarrow$ Gradient descent works using Upper linear bound, Mirror descent works using Lower linear bound



**Definition**(alternative): Upper quadratic bound

A second order differentiable function over a *convex* set $\mathcal{D}$ is L-smooth **if and only if**:

1. $v^{T}\nabla^{2}f(x)v \leq L$ for every unit vector $v$, for every $x \in \mathcal{D}$
2. $||\nabla f(x) - \nabla f(y)||_{2} \leq L||x-y||_{2}$



Additional: a second order differentiable convex function over a *convex* set $\mathcal{D}$ is L-smooth **if and only if**:

$v^{T}\nabla^{2}f(x)v \geq 0$ for every vector $v$, for every $x \in \mathcal{D}$



So BUT WHY DOES IT WORK?, HOW TO CHOOSE $\eta$?

Suppose $f(x)$ is L-smooth, by the upper quadratic bound, $ f(y) \leq f(x) + <\nabla f(x), y-x>+\frac{L}{2}||y-x||_{2}^{2} $

Take $x = x_t, y= x_{t+1}$. Then,
$$
f(x_{t+1}) \leq f(x_{t}) + <\nabla f(x_{t}), \eta\nabla f(x_{t})>+\frac{L}{2}||\eta\nabla f(x_{t})||_{2}^{2}
$$
For every $\eta \leq \frac{1}{L}$,  $\eta^{2}\frac{L}{2} \leq \frac{\eta}{2}$. Then,
$$
f(x_{t+1}) \leq f(x_{t}) -\eta\frac{L}{2}||\nabla f(x_{t})||_{2}^{2}
$$


Recall

**Theorem**

When there is no local minima, for a 1st order differentiable convex function $f$, $\nabla f(x^{\star}) = 0 \Leftrightarrow f(x^{\star}) =min_{x}f(x)$

proof) $(\rightarrow)$ for every $y$, and let $x=x^{\star}$, then $f(y) \geq f(x^{\star}) + <\nabla f(x^{\star}), y-x> = f(x^{\star})$.

($\leftarrow$)  Put $x_{t+1}=x_{t}=x^{\star}$. Then $f(x^{\star}) \leq f(x^{\star}) -\eta\frac{L}{2}||\nabla f(x^{\star})||_{2}^{2}$. It implies $||\nabla f(x^{\star})||_{2}^{2} \leq 0$. So $\nabla f(x^{\star}) = 0$



But if the gradient is too small, slowly converges. Use trade off between fast and accurate.

Given $\epsilon >0$, how many iterations $T$ do we need to find an $x_T$ s.t
$$
||\nabla f(x_T)||_2^{2} \leq \epsilon
$$
We need $T_{\epsilon} = \frac{2(f(x_0) - min_x f(x)}{\eta\epsilon}$ interations.

proof)

Suppose for every $t \leq T_{\epsilon}, ||\nabla f(x_{t})||_2^{2} > \epsilon$, then

$$
f(x_{t+1}) < f(x_t) - \frac{\eta}{2}\epsilon
$$
it implies that
$$
f(x_{T_{\epsilon}}) < f(x_0) -\frac{\eta}{2}\epsilon T_{\epsilon} = f(x_0) - (f(x_0) - min_xf(x)) = min_xf(x)
$$
Contradiction!



We can also find that gradient descent convergence does not need $f$ to be a convex function. But only convergence in gradient.



What about function value?
