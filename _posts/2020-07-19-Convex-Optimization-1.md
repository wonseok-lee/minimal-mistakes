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

Optimization algorithm: Solve **$\underset{x \in \mathcal{D}}{\mathbb{min}}f(x)$** as fast as possible and as good as possible.

Keywords: fast and good(tradeoff between them).

>**Definition**: convex set
>
>A convex set $\mathcal{D}$ is a non-empty set if  $(1-\lambda)x + \lambda y \in \mathcal{D}$ for every $x, y \in \mathcal{D}$, for every $\lambda \in (0,1)$

>**Definition**: convex function
>
>A function $f$ over a convex set $\mathcal{D}$ is convex if $f((1-\lambda)x + \lambda y) \leq (1-\lambda)f(x) + \lambda f(y)$

>**property of a convex function**
>
>For every differentiable convex function $f$ over a convex set $\mathcal{D}$, for every point $x, y \in \mathcal{D}$, $f(y) \geq f(x) + \langle\nabla f(x), y-x\rangle$
>
>
>
>proof) Suppose there is an  $y \in \mathcal{D}$ s.t $f(y) \leq f(x) + \langle\nabla f, y-x\rangle-\delta$ for $\delta>0$.
>
>
>
>Then, by convexity(definition), for every $\lambda \in (0,1)$, 
>
>
>$$
>\begin{align*}
>
>f((1-\lambda)x + \lambda y) &= f(x+\lambda(y-x))\\
>&\leq (1-\lambda)f(x) + \lambda f(y) \\ 
>&\leq (1-\lambda) f(x) + \lambda f(x)+\lambda \langle\nabla f, y-x\rangle-\lambda \delta \\
>&= f(x) + \lambda \langle\nabla f, y-x\rangle-\lambda \delta \\
>
>\end{align*}
>$$
>
>
>
>It implies that
>$$
>\frac{f(x+\lambda(y-x))}{\lambda} \leq \langle\nabla f(x), y-x\rangle -\delta
>$$
>
>
>
>Let $\lambda \rightarrow 0^{+}$, then, $\langle\nabla f(x), y-x\rangle \leq \langle\nabla f(x), y-x\rangle -\delta$
>
>Contradiction!



>**Theorem**
>
>When there is no local minima, for a 1st order differentiable convex function $f$, $\nabla f(x^{\star}) = 0 \Leftrightarrow f(x^{\star}) = \underset{x}{\mathbb{min}}f(x)$
>
>
>proof) $(\rightarrow)$ for every $y$, and let $x=x^{\star}$, then $f(y) \geq f(x^{\star}) + \langle\nabla f(x^{\star}), y-x\rangle = f(x^{\star})$.
>
>($\leftarrow$) will be treated below.
>
>For general Lipchitz convex function, $\exists y \in \partial f(x^{\star}), y=0 \Leftrightarrow f(x^{\star}) =\underset{x}{\mathbb{min}}f(x)$



**QUESTION** How do we find the minimizer and a point that has small gradient?

**Anwser**: **Gradient Descent**: $x_{t+1} = x_{t}-\eta\nabla f(x_{t})$ for $t=0,1,2, \dots$

$x_0$ : starting point, $\eta$: step size

BUT WHY DOES IT WORK?, HOW TO CHOOSE $\eta$?

>**Definition**: smoothness
>
>a 1st order differentiable function (not necessarily convex) $f$ over a set (not necessarily convex) $\mathcal{D}$ is called L smooth for L > 0 if
>$$
>f(y) \leq f(x) + \langle\nabla f(x), y-x\rangle+\frac{L}{2}\Vert y-x\Vert_{2}^{2}
>$$


>**Theorem** sum of two L-smooth functions is a 2L-smooth function.
>
>proof) let $f = f_1 + f_2$ and $f_1, f_2$ are L-smooth functions.
>
>$$
>\begin{align*}
>f(y) &= f_1(y) +f_2(y) \\
>&\leq f_1(x) + \langle\nabla f_1(x), y-x\rangle+\frac{L}{2}\Vert y-x\Vert_{2}^{2} + f_2(x) + \langle\nabla f_2(x), y-x\rangle+\frac{L}{2}\Vert y-x\Vert_{2}^{2} \\
>&= f(x) + \langle\nabla f(x), y-x\rangle+\frac{2L}{2}\Vert y-x\Vert_{2}^{2}
>\end{align*}
>$$


Recall
$$
f(y) \geq f(x) + \langle\nabla f(x), y-x\rangle
$$
$\Rightarrow$ Gradient descent works using Upper linear bound, Mirror descent works using Lower linear bound



>**Definition**(alternative): Upper quadratic bound
>
>A second order differentiable function over a *convex* set $\mathcal{D}$ is L-smooth **if and only if**:
>
>1. $v^{T}\nabla^{2}f(x)v \leq L$ for every unit vector $v$, for every $x \in \mathcal{D}$
>
>2. $\Vert\nabla f(x) - \nabla f(y)\Vert_{2} \leq L\Vert x-y\Vert_{2}$



>Additional: a second order differentiable convex function over a *convex* set $\mathcal{D}$ is L-smooth **if and only if**:
>
>$v^{T}\nabla^{2}f(x)v \geq 0$ for every vector $v$, for every $x \in \mathcal{D}$



So BUT WHY DOES IT WORK?, HOW TO CHOOSE $\eta$?

Suppose $f(x)$ is L-smooth, by the upper quadratic bound, $ f(y) \leq f(x) + \langle\nabla f(x), y-x\rangle+\frac{L}{2}\Vert y-x\Vert_{2}^{2} $

Take $x = x_t, y= x_{t+1}$. Then,

$$
\begin{align*}
f(x_{t+1}) &\leq f(x_{t}) - \langle\nabla f(x_{t}), \eta\nabla f(x_{t})\rangle+\frac{L}{2}\Vert\eta\nabla f(x_{t})\Vert_{2}^{2} \\
&= f(x_{t}) - \eta\Vert\nabla f(x_{t})\Vert_{2}^{2} + \frac{L\eta^2}{2}\Vert\nabla f(x_{t})\Vert_{2}^{2}
\end{align*}
$$

For every $\eta \leq \frac{1}{L}$, $\eta^{2}\frac{L}{2} \leq \frac{\eta}{2}$. Then,

$$
f(x_{t+1}) \leq f(x_{t}) -\frac{\eta}{2}\Vert\nabla f(x_{t})\Vert_{2}^{2}
$$


Recall

>**Theorem**
>
>When there is no local minima, for a 1st order differentiable convex function $f$, $\nabla f(x^{\star}) = 0 \Leftrightarrow f(x^{\star}) =\underset{x}{\mathbb{min}}f(x)$
>
>proof) $(\rightarrow)$ for every $y$, and let $x=x^{\star}$, then $f(y) \geq f(x^{\star}) + \langle\nabla f(x^{\star}), y-x\rangle = f(x^{\star})$.
>
>($\leftarrow$)  Put $x_{t+1}=x_{t}=x^{\star}$. Then $f(x^{\star}) \leq f(x^{\star}) -\eta\frac{L}{2}\Vert\nabla f(x^{\star})\Vert_{2}^{2}$. 
>
>It implies 
>$\Vert\nabla f(x^{\star})\Vert_{2}^{2} \leq 0$. So $\nabla f(x^{\star}) = 0$



But if the gradient is too small, slowly converges. Use trade off between fast and accurate.

>Given $\epsilon >0$, how many iterations $T$ do we need to find an $x_T$ s.t
>
>$$
>\Vert\nabla f(x_T)\Vert_2^{2} \leq \epsilon
>$$
>
>We need $T_{\epsilon} = \frac{2(f(x_0) - \underset{x}{\mathbb{min}}f(x))}{\eta\epsilon}$ interations.
>
>proof)
>
>Suppose for every $t \leq T_{\epsilon}, \Vert\nabla f(x_{t})\Vert_2^{2} > \epsilon$, then
>
>$$
>f(x_{t+1}) < f(x_t) - \frac{\eta}{2}\epsilon
>$$
>
>it implies that
>
>$$
>f(x_{T_{\epsilon}}) < f(x_0) -\frac{\eta\epsilon}{2} T_{\epsilon} = f(x_0) - (f(x_0) - \underset{x}{\mathbb{min}}f(x)) = \underset{x}{\mathbb{min}}f(x)
>$$
>
>Contradiction!



We can also find that gradient descent convergence does not need $f$ to be a convex function. But only convergence in gradient.



What about function value?


#### Mirror Descent Lemma



In Gradient Descent convergence does not need $f$ to be a convex function!

But it is only convergence in gradient.

What about function value? $\Rightarrow$ Mirror Descent Lemma



The Mirror Descent Lemma for the update $x_{t+1} = x_t − \eta\nabla f(x_t)$ on a convex function $f$.

Before that, there is a important fact.

>**FACT1(important)**
>
>
>$$
>\begin{align*}
>\langle x_t-x_{t+1}, y-x_t \rangle &= x_t^Ty - x_t^Tx_t -x_{t+1}^Ty +x_{t+1}^Tx_t \\
>&= -\frac{1}{2}[x_t^Tx_t-2x_t^Ty +2x_{t+1}^Ty-2x_{t+1}^Tx_{t}+x_t^Tx_t] \\
>&= -\frac{1}{2}[x_t^Tx_t-2x_t^Ty+y^Ty -y^Ty +2x_{t+1}^Ty-x_{t+1}^Tx_{t+1}+x_{t+1}^Tx_{t+1} -2x_{t+1}^Tx_{t}+x_t^Tx_t] \\
>&= -\frac{1}{2}[\Vert x_t-y\Vert_2^2-\Vert y-x_{t+1}\Vert_2^2 + \Vert x_{t+1}-x_t\Vert_2^2] 
>\end{align*}
>$$





For any point $y$, by the lower linear bound, we have


$$
f(y) \geq f(x)+\langle \nabla f(x),y-x \rangle
$$


> **Three Terms Mirror Descent Lemma**
>
> For every point $y$,
>
>
> $$
> f(y) \geq f(x)+\langle \nabla f(x),y-x \rangle
> $$
> Put $x=x_t$. Then,
>
>
> $$
> \begin{align*}
> f(x_t) &\leq f(y)-\langle \nabla f(x),y-x_t \rangle \\
> &= f(y)\ -\frac{1}{\eta} \langle x_t-x_{t+1},y-x_t\rangle 
> \\
> \\
> &\text{use the FACT1}
> \\
> \\
> &=f(y)\ +\frac{1}{2\eta}(\Vert x_t-y\Vert_2^2-\Vert y-x_{t+1}\Vert_2^2 + \Vert x_{t+1}-x_t\Vert_2^2)
> \end{align*}
> $$

let $y= x^{\ast}:=\underset{x \in \mathcal{D}}{\mathbb{argmin}}f(x)$

Sum the above inequality $\text{from}\ t= 0\ \text{to}\ t= T-1$


$$
\begin{align*}
\sum_{t=0}^{T-1}f(x_t) &\leq Tf(x^{\ast}) +\frac{1}{2\eta}\sum_{t=0}^{T-1}(\Vert x_t-x^{\ast}\Vert_2^2-\Vert x^{\ast}-x_{t+1}\Vert_2^2 + \Vert x_{t+1}-x_t\Vert_2^2) \\
&= Tf(x^{\ast}) +\frac{1}{2\eta}(\Vert x_0-x^{\ast}\Vert_2^2-\Vert x^{\ast}-x_{T}\Vert_2^2 + \sum_{t=0}^{T-1}\Vert x_{t+1}-x_t\Vert_2^2) \\
\end{align*}
$$


It implies, 


$$
\begin{align*}
&\Rightarrow \sum_{t=0}^{T-1}f(x_t) \leq Tf(x^{\ast}) +\frac{1}{2\eta}(\Vert x_0-x^{\ast}\Vert_2^2-\Vert x^{\ast}-x_{T}\Vert_2^2 + \sum_{t=0}^{T-1}\Vert x_{t+1}-x_t\Vert_2^2) 
\\
\\
&\Rightarrow \frac{1}{T}\sum_{t=0}^{T-1}f(x_t) \leq f(x^{\ast}) +\frac{1}{2\eta T}(\Vert x_0-x^{\ast}\Vert_2^2-\Vert x^{\ast}-x_{T}\Vert_2^2 + \sum_{t=0}^{T-1}\Vert x_{t+1}-x_t\Vert_2^2)
\\
\\
&\text{remove the negative term}
\\
\\
&\Rightarrow \frac{1}{T}\sum_{t=0}^{T-1}f(x_t) \leq f(x^{\ast}) +\frac{1}{2\eta T}(\Vert x_0-x^{\ast}\Vert_2^2+ \sum_{t=0}^{T-1}\Vert x_{t+1}-x_t\Vert_2^2)
\\
\\
&\text{use $x_{t+1} = x_t -\nabla f(x_t)$}
\\
\\
&\Rightarrow \frac{1}{T}\sum_{t=0}^{T-1}f(x_t) \leq f(x^{\ast}) +\frac{1}{2\eta T}\Vert x_0-x^{\ast}\Vert_2^2+ \frac{\eta}{2T}\sum_{t=0}^{T-1}\Vert \nabla f(x_t)\Vert_2^2
\\
\\
\end{align*}
$$


Here, think about Gradient Descent Lemma (when $\eta \leq \frac{1}{L}$) - **FACT2**


$$
\begin{align*}
& \ \ \ \ \ \ \ f(x_{t+1}) \leq f(x_t) - \frac{\eta}{2}\Vert \nabla f(x_t)\Vert_{2}^2 \\
\\
&\Rightarrow \frac{\eta}{2T}\sum_{t=0}^{T-1}\Vert \nabla f(x_t)\Vert_2^2 \leq f(x_0) -f(x_T) \leq f(x_0) - f(x^{\ast})

\end{align*}
$$


Also, L-smoothness of $f$ (and $\nabla f(x^{\ast})=0$) - **FACT3**


$$
f(x_0) \leq f(x^{\ast}) +\frac{L}{2}\Vert x_0-x^{\ast}\Vert_2^2
$$


Use **FACT 2,3**


$$
\begin{align*}
\frac{1}{T}\sum_{t=0}^{T-1}f(x_t) &\leq f(x^{\ast}) +\frac{1}{2\eta T}\Vert x_0-x^{\ast}\Vert_2^2+ \frac{\eta}{2T}\sum_{t=0}^{T-1}\Vert \nabla f(x_t)\Vert_2^2
\\
\\
&\leq f(x^{\ast}) +\frac{1}{T}(\frac{1}{2\eta}\Vert x_0-x^{\ast}\Vert_2^2+ f(x_0) - f(x^{\ast}))
\\
\\
&\leq f(x^{\ast}) +\frac{1}{T}(\frac{1}{2\eta}\Vert x_0-x^{\ast}\Vert_2^2+ \frac{L}{2}\Vert x_0-x^{\ast}\Vert_2^2)
\\
\\
&= f(x^{\ast}) +\frac{\Vert x_0-x^{\ast}\Vert_2^2}{T}(\frac{1}{2\eta}+ \frac{L}{2})
\end{align*}
$$


And we have $f(x_T) \leq f(x_t)$ for every $t\leq T$ and $\eta \leq \frac{1}{L}$. Hence


$$
\begin{align*}
f(x_T) &\leq f(x^{\ast}) +\frac{\Vert x_0-x^{\ast}\Vert_2^2}{T}(\frac{1}{2\eta}+ \frac{L}{2})
\\
\\
& \leq f(x^{\ast}) +\frac{\Vert x_0-x^{\ast}\Vert_2^2}{\eta T}
\end{align*}
$$


###### BUT WHAT HAVE WE DONE?

> Recall: Mirror Descent Lemma
> $$
> f(x_t) \leq f(y)\ +\frac{1}{2\eta}(\Vert x_t-y\Vert_2^2-\Vert y-x_{t+1}\Vert_2^2 + \Vert x_{t+1}-x_t\Vert_2^2)
> $$



$\Vert x_{t+1} - x_t\Vert_2^2$ is $O(\eta^2)$. Thus for sufficiently small $\eta$:

Instead of decreasing the function value of $f$ , the algorithm is actually decreasing the distance between $x_t$ to $x^{\ast}$



That’s why its called Mirror Descent. It is extremely important as well for general problems beyond convexity. There, reddecreasing the distance between $x_t$ to $x^{\ast}$ is replaced by a decreasing potential function.
