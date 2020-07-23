---
title: "Convex Optimization 2"
date: 2020-07-24 00:00:28 -0400
categories: convex optimization
use_math: true
---

## Acceleration



Q. Is the Gradient Descent Algorithm: $x_{t+1} = x_{t} − \eta\nabla f(x_t)$  the best algorithm to use in practice?

A. NO



We use Accelerated Gradient Descent algorithm.



Think about what we can do to outrun the update $x_{t+1} = x_{t} − \eta\nabla f(x_t)\ \text{with}\ \eta \leq \frac{1}{L}$ on L-smooth function?



Naive Answer:  use the same update with a larger learning rate $\eta \gg\frac{1}{L} $ 



Let's think about $f(x) =x^2$. Then $x_t = (-1)^t$ which never converges. (Let's call this situation as *bumping back and forth.*) 



It is natural to think about **how we adjust the learning rate automatically according to the shape of the function?**



Always using large learning rate is the answer. $\rightarrow$ use the weighted sum of the gradients from the previous iterations to update the current point.



$\Rightarrow$ **Nesterov's Accelerated Gradient Descent Update** 

>Nesterov's Accelerated Gradient Descent.
>
>For a L-smooth function:
>
>- Gradient Descent step: $z_{t+1} = x_{t} − \eta\nabla f(x_t)$
>
>- Momentum step: $x_{t+1} = (1 − \gamma_t)z_{t+1} + \gamma_tz_t$
>
>- $\lambda_0=0,\ \lambda_t=\frac{1+\sqrt{1+4\lambda_{t-1}^2}}{2},\ \text{and}\ \gamma_t=\frac{1-\lambda_{t1}}{\lambda_{t+1}}$
>
>  (for large $t$, $\lambda_{t+1}\approx \lambda_{t}+\frac{1}{2}\, \lambda_{t} \approx \frac{t}{2}, \text{and}\ \gamma_t\approx-1$)



Let's see this algorithm in detail.



With $\gamma_t\approx-1\ \text{and}\ \Vert x_{t+1}-x_t\Vert_2=O(\eta)$,


$$
\begin{align*}
x_{t+1} &\approx (1 − \gamma_t)(x_{t} − \eta\nabla f(x_t)) + \gamma_t(x_{t-1} − \eta\nabla f(x_{t-1})) 
\\
\\
&\approx x_{t} − \eta\nabla f(x_t) + (x_{t}- x_{t-1} − \eta\nabla f(x_t)  + \eta\nabla f(x_{t-1}))
\\
\\
&\approx x_{t} − \eta\nabla f(x_t) + (x_{t}- x_{t-1}) + O(\eta^2)
\end{align*}
$$


Therefore, 


$$
\begin{align*}
&x_{t+1}-x_{t-1}\approx  (x_{t}- x_{t-1})− \eta\nabla f(x_t)
\\
\\
&(\text{Sum up the above from $1$ to $t$})
\\
\\
&\Rightarrow x_{t}-x_{t}\approx x_{1}- x_{0}− \eta\sum_{s\leq t}\nabla f(x_s)
\end{align*}
$$


Therefore, Accelerated Gradient Descent is approximately taking a step using the sum of past gradients.



Momentum: "Weighted" sum of the past gradients. $\rightarrow$ it makes Gradient Descent more stable.



Gradient is small: We can now use a larger learning rate $\eta \gg\frac{1}{L} $ without bumping back and forth.



But the function is L-smooth, how do we reason about the update with $\eta \gg\frac{1}{L} $?



For simplicity, assuming $f (x^{\ast}) = 0$



1. For a fixed value $K>0$, if $\Vert \nabla f(x_t) \Vert_2^2 \geq K$ holds for every $t$, for $\eta \leq \frac{1}{L} $, (Gradient Descent Lemma):


$$
\begin{align*}
&f(x_{i+1}) \leq f(x_i)-\frac{\eta}{2}\Vert\nabla f(x_i) \Vert_2^2
\\
\\
&(\text{Sum up the above from $0$ to $T-1$ and using $\eta =\frac{1}{L}$})
\\
\\
&f(x_{T}) \leq f(x_0)-\frac{KT}{2L}
\end{align*}
$$


Then, need at most $\frac{Lf(x_0)}{K}$ iterations to find a point $x_T$ with $f(x_{T}) \leq\frac{f(x_0)}{2}$



>Check
>
>$T=\frac{Lf(x_0)}{K} \rightarrow f(x_0)-\frac{KT}{2L}=\frac{f(x_0)}{2}\geq f(x_{T})$   



2. For a fixed value $K>0$, if $\Vert \nabla f(x_t) \Vert_2^2 < K$ holds for every $t$, for every $\eta$, (Mirror Descent Lemma):


$$
\begin{align*}
&\frac{1}{T}\sum_{t=0}^{T-1}f(x_t) \leq f(x^{\ast}) +\frac{1}{2\eta T}\Vert x_0-x^{\ast}\Vert_2^2+ \frac{\eta}{2T}\sum_{t=0}^{T-1}\Vert \nabla f(x_t)\Vert_2^2
\\
\\
\Rightarrow & \frac{1}{T}\sum_{t=0}^{T-1}f(x_t) \leq \frac{1}{2\eta T}\Vert x_0-x^{\ast}\Vert_2^2+ \frac{\eta K}{2}
\end{align*}
$$


With $\eta =\frac{f(x_0)}{2K}$, need at most $\frac{4K\Vert x_0-x^{\ast}\Vert_2^2}{f(x_0)^2}$ iterations to find a point $x_T$ with $f(x_{T}) \leq\frac{f(x_0)}{2}$



>Check
>
> we have $f(x_T) \leq f(x_t)$ for every $t\leq T$
>$$
>\begin{align*}
>f(x_T) &\leq \frac{1}{T}\sum_{t=0}^{T-1}f(x_t)
>\\
>\\
>&\leq \frac{2K}{2f(x_0)}\frac{f(x_0)^2}{4K\Vert x_0-x^{\ast}\Vert_2^2}\Vert x_0-x^{\ast}\Vert_2^2 +\frac{f(x_0)K}{4K} 
>\\
>\\
>&= \frac{f(x_0)}{4} +\frac{f(x_0)}{4}
>\\
>\\
>&=\frac{f(x_0)}{2} 
>\end{align*}
>$$



3. (In both cases,) with $K=\sqrt{\frac{Lf^3(x0)}{4\Vert x_0-x^{\ast}\Vert_2^2}}$, need at most $\frac{2\Vert x_0-x^{\ast}\Vert_2\sqrt{L}}{\sqrt{f(x_0)}}$ iterations to find a point $x_T$ with $f(x_{T}) \leq\frac{f(x_0)}{2}$

   

   > Check
   >
   > 
   >
   > 1. 
   >    $$
   >    \frac{Lf(x_0)}{K} = Lf(x_0)\sqrt{\frac{4\Vert x_0-x^{\ast}\Vert_2^2}{Lf^3(x_0)}}=\frac{2\Vert x_0-x^{\ast}\Vert_2\sqrt{L}}{\sqrt{f(x_0)}}
   >    $$
   >    
   >
   > 2. $$
   >    \begin{align*}
   >    \frac{4K\Vert x_0-x^{\ast}\Vert_2^2}{f(x_0)^2} &= \sqrt{\frac{Lf^3(x0)}{4\Vert x_0-x^{\ast}\Vert_2^2}} \frac{4\Vert x_0-x^{\ast}\Vert_2^2}{f(x_0)^2}
   >    \\
   >    \\
   >    &= \frac{2\Vert x_0-x^{\ast}\Vert_2\sqrt{L}}{\sqrt{f(x_0)}}
   >    \end{align*}
   >    $$


   
   In the second case, when $f (x_0)\approx\Vert x_0 − x^{\ast}\Vert_2\approx 1$, the learning rate is much larger: $\eta =\frac{f(x_0)}{2k} \approx \frac{1}{\sqrt{L}} \gg \frac{1}{L}$



> Think about a case for $f (x_0)=\Vert x_0 − x^{\ast}\Vert_2= 1$
>
>
> 
> need at most $\frac{2\sqrt{L}}{\sqrt{1}}$ iterations to find a point $x_T$ with $f(x_{T}) \leq\frac{1}{2}$
> 
> 
>
> need at most $\frac{2\sqrt{L}}{\sqrt{1/2}}$ iterations to find a point $x_T$ with $f(x_{T}) \leq\frac{1}{4}$
> 
> 
>
> $\cdots$
>
>
> 
> for $\varepsilon >0$, need at most $\frac{2\sqrt{L}}{\sqrt{\varepsilon}}$ iterations to find a point $x_T$ with $f(x_{T_{\varepsilon}}) \leq\varepsilon$



It might neither be the case of $\Vert \nabla f(x_t) \Vert_2^2 \geq K$ holds for every $t$ nor $\Vert \nabla f(x_t) \Vert_2^2 < K$ holds for every $t$.



Every iteration, we do both a step with $\eta =\frac{1}{L} $ (Gradient Descent) and a step with a larger $\eta \gg\frac{1}{L} $ (using Momentum to stablize). 



In the end combine them:



From now on prove Accelerated Gradient Descent algorithm. It will be quite long and tough.



If your teacher does not require the proof, you can pass it :)



#### Proof



> **Linear Coupling**
>
> At every iteration, for a  fixed $\tau$: 
>
> - Update (small learning rate):   $s_{t+1} = x_{t} − \frac{1}{L}\nabla f(x_t)$
> - Update (large learning rate $\eta \gg\frac{1}{L} $):  $I_{t+1} = I_{t} − \eta\nabla f(x_t)$
> - Linear coupling: for a $\tau \in [0,1]$, $x_{t+1} = (1 − \tau)s_{t+1} + \tau l_{t+1}$
>
> $\Rightarrow I_{t+1}=I_0-\eta\sum_{r=0}^{t-1}\nabla f(x_r)$  is the momentum term. The final update combines (small learning rate) gradient descent with this (large learning rate) momentum.



We can see that 


$$
\langle \nabla f(x_t), x^{\ast}-I_t\rangle =\frac{1}{\eta}\langle I_{t}-I_{t+1}, x^{\ast}-I_t\rangle
$$


>  Recall
> $$
> \frac{1}{\eta}\langle I_{t}-I_{t+1}, x^{\ast}-I_t\rangle =-\frac{1}{2\eta}[\Vert x^{\ast}-I_t\Vert_2^2-\Vert x^{\ast}-I_{t+1}\Vert_2^2 + \Vert I_t-I_{t+1}\Vert_2^2]
> $$



Now we have 


$$
\langle \nabla f(x_t), x^{\ast}-I_t\rangle = -\frac{1}{2\eta}[\Vert x^{\ast}-I_t\Vert_2^2-\Vert x^{\ast}-I_{t+1}\Vert_2^2 + \Vert I_t-I_{t+1}\Vert_2^2]
$$



1. lower linear bound


$$
\langle \nabla f(x_t), s_t -x_t\rangle \leq f(s_t) - f(x_t)
\\
\\
\Leftrightarrow f(x_t) - f(s_t) \leq \langle \nabla f(x_t), x_t-s_t\rangle
$$


2. By definition of Linear coupling: for a $\tau \in [0,1]$, $x_{t+1} = (1 − \tau)s_{t+1} + \tau l_{t+1}$


$$
I_t-x^{\ast}+\frac{1-\tau}{\tau}(s_t-x_t)=x_t-x^{\ast}
$$


3.  Gradient Descent Lemma:

   
   $$
   f(s_{t+1}) \leq f(x_r)-\frac{1}{2L}\Vert\nabla f(x_r) \Vert_2^2
   $$
   



Therefore, combining 1,2,3,


$$
\begin{align*}
f(x_t) - f(x^{\ast}) &\leq \langle \nabla f(x_t), x_t -x^{\ast}\rangle 
\\
\\
&= \langle \nabla f(x_t), I_t-x^{\ast}\rangle + \frac{1-\tau}{\tau}\langle \nabla f(x_t), s_t -x_t\rangle
\\
\\
&\leq \frac{1}{2\eta}[\Vert x^{\ast}-I_t\Vert_2^2-\Vert x^{\ast}-I_{t+1}\Vert_2^2 + \Vert I_t-I_{t+1}\Vert_2^2] + \frac{1-\tau}{\tau}\langle \nabla f(x_t), s_t -x_t\rangle
\\
\\
&=\frac{1}{2\eta}[\Vert x^{\ast}-I_t\Vert_2^2-\Vert x^{\ast}-I_{t+1}\Vert_2^2 + \eta^2\Vert \nabla f(x_t)\Vert_2^2] + \frac{1-\tau}{\tau}\langle \nabla f(x_t), s_t -x_t\rangle
\\
\\
&\leq \frac{1}{2\eta}[\Vert x^{\ast}-I_t\Vert_2^2-\Vert x^{\ast}-I_{t+1}\Vert_2^2 + 2L\eta^2(f(x_t)-f(s_{t+1}))] + \frac{1-\tau}{\tau}\langle \nabla f(x_t), s_t -x_t\rangle
\\
\\
&\leq \frac{1}{2\eta}[\Vert x^{\ast}-I_t\Vert_2^2-\Vert x^{\ast}-I_{t+1}\Vert_2^2 + 2L\eta^2(f(x_t)-f(s_{t+1}))] + \frac{1-\tau}{\tau}(f(s_t)-f(x_{t}))
\\
\\
&\text{pick $\tau$ s.t $L\eta = \frac{1-\tau}{\tau}$}
\\
\\
&\leq \frac{1}{2\eta}[\Vert x^{\ast}-I_t\Vert_2^2-\Vert x^{\ast}-I_{t+1}\Vert_2^2 + 2L\eta^2(f(s_t)-f(s_{t+1}))]
\end{align*}
$$


Sum from $t=0$ to $T-1$,assuming $f(x^{\ast})=0$, then,


$$
\frac{1}{T}\sum_{t=0}^{T-1}f(x_t) \leq \frac{1}{2T\eta}[\Vert x^{\ast}-I_0\Vert_2^2+ 2L\eta^2f(s_0)]
$$


Picking $\eta =\frac{\Vert x^{\ast}-I_0\Vert_2}{2\sqrt{f(s_0)}}$, we can find a point $x_T$ with $f(x_{T}) \leq\frac{f(x_0)}{2}$ in 


$$
T_{AGD} = \frac{2\sqrt{2L}\Vert x^{\ast}-I_0\Vert_2}{\sqrt{f(s_0)}}
$$


> Check
>
> Before the start we know $2xy\leq x^2+y^2$.
>
> $$
> \begin{align*}
> \frac{1}{2T\eta}[\Vert x^{\ast}-I_0\Vert_2^2+ 2L\eta^2f(s_0)] &= \frac{1}{2T\eta}[\Vert x^{\ast}-I_0\Vert_2^2 + 2L\frac{\Vert x^{\ast}-I_0\Vert_2^2}{4f(s_0)}f(s_0)]
> \\
> \\
> &= \frac{1}{2T\eta}\Vert x^{\ast}-I_0\Vert_2^2(1 + \frac{L}{2})
> \\
> \\
> &=\frac{1}{2}\frac{\sqrt{f(s_0)}}{2\sqrt{2L}\Vert x^{\ast}-I_0\Vert_2}\frac{2\sqrt{f(s_0)}}{\Vert x^{\ast}-I_0\Vert_2}\Vert x^{\ast}-I_0\Vert_2^2(1 + \frac{L}{2})
> \\
> \\
> &=\frac{1}{2}\frac{f(s_0)}{\sqrt{2L}}(1 + \frac{L}{2})
> \\
> \\
> &(\text{by the cauchy schwartz inequality,}\ \sqrt{2L} \leq 1 + \frac{L}{2})
> \\
> \\
> &=\frac{f(s_0)}{2}
> \end{align*}
> $$



The really important fact of momentumn

 $\rightarrow$ want to use larger learning rate to do gradient descent, beyond the smoothness of the function.

 $\rightarrow$ want to do it stably: Use momentum: \weighted" sum of the past gradients.
