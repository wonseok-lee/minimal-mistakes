---
title: "Biweekly Meeting 07/22"
date: 2020-07-22 00:00:28 -0400
categories: paper
use_math: true
---

## Weekly Meeting 20200722



Christopher Nemeth and Chris Sherlock. (2018). *Merging MCMC Subposteriors through Gaussian-Process Approximations.* Bayesian Analysis



#### Basic Idea

Consider a data set $\mathcal{Y} = \{y_1,\ldots, y_n\}$.

Data are conditionally independent with likelihood $\Pi_{i =1}^{i=n}p(y_i\vert\vartheta)$, where $\vartheta \in \mathcal{\Theta} \subseteq \mathbb{R}^d$ is model parameter.

the data set $\mathcal{Y}$ can be partitioned into C batches $\{\mathcal{Y_1}, \ldots, \mathcal{Y_C}\}$ where we define a subposterior operating on a subset of the data $\mathcal{Y_c}$ as $\pi_c(\vartheta)=p(\vartheta\vert\mathcal{Y_c}) \propto p(\mathcal{Y_c}\vert\vartheta)p(\vartheta)^{1/C}$

#### Run MCMC on each subposterior

subposterior MCMC -> HMC

$\pi(\vartheta, \varphi)\propto \exp(\log\pi(\vartheta)-\frac{1}{2}\varphi^TM^{-1}\varphi)$

sample from the target distribution by simulating $\vartheta$ and $\varphi$ through fictitious time $\tau$ using Hamilton’s equations.

$$
d\vartheta=M^{-1}\varphi d\tau,\\ 
d\varphi = \nabla_{\vartheta} \log\pi(\vartheta)d\tau
$$

The differential equations are intractable and must be solved numerically.

$$
\varphi_{\tau + \frac{\epsilon}{2}}= \varphi_{\tau} + \frac{\epsilon}{2}\nabla_{\vartheta_{\tau}} \log\pi(\vartheta_{\tau}), \\
\vartheta_{\tau +\varepsilon} = \vartheta_{\tau} + \epsilon M^{-1}\varphi_{\tau + \frac{\epsilon}{2}}, \\
\varphi_{\tau + \epsilon}= \varphi_{\tau+\frac{\epsilon}{2}} + \frac{\epsilon}{2}\nabla_{\vartheta_{\tau+ \epsilon}} \log\pi(\vartheta_{\tau+ \epsilon})
$$


#### Fit GP to each subposterior

Parallelising the MCMC procedure over $C$ computing nodes results in $C$ subposteriors $\{\pi_{c}(\vartheta)\}_{c=1}^{C}$

The MCMC algorithm for each subposterior, $c$, has been iterated $J$ times to give $\mathcal{D}_c = \{\vartheta_j, \mathcal{l}_c(\vartheta_j)\}$, 

where $\mathcal{l}_c(\vartheta_j)=\log\pi_c(\vartheta_j)$ and each pair consists of a sample from the Markov chain with its associated log-subposterior density.



Gaussian-process prior distribution

$$
\mathcal{L}_c(\vartheta) \sim \mathcal{GP}(m(\vartheta),K(\vartheta,\vartheta'))
$$

where mean function $m: \vartheta \rightarrow \mathbb{R}$ and covariance $K: \vartheta\times\vartheta \rightarrow \mathbb{R}$

Ensure that $\int \exp\{\mathcal{L_c(\vartheta)}\}d\vartheta < \infty$ (also, $\mathcal{L_c}(\theta) \rightarrow -\infty$ as $\theta \rightarrow \pm\infty$)almost surely setting the mean function.

$$
m(\vartheta) =\beta_0+\vartheta_1^T\beta_1+diag(\vartheta^TV^{-1}\vartheta)\beta_2, \\
\beta_2 < 0,\ i =0,1,2
$$

$V$ is the empirical covariance the posterior for $\vartheta$ obtained from the MCMC sample and $\beta_i$ are unknown constants.

$K(\vartheta,\vartheta')=w^2\exp(-\frac{1}{2}(\vartheta-\vartheta')^T\Lambda^{-1}(\vartheta-\vartheta'))$

Given the choice of prior, $\mathcal{D}_{c}$ are observations of this Gaussian-process generated from an MCMC algorithm targeting the subposterior $\pi_c(\vartheta)$, giving up to a constant of proportionality the posterior distribution, 

$$
p(\mathcal{l_c}(\vartheta)\vert\mathcal{D}_c) \propto p(\mathcal{D}_c\vert\mathcal{l_c}(\vartheta)p(\mathcal{l_c}(\vartheta))
$$

Define $\mathcal{L_c}(\vartheta_{1:J}) := \{\mathcal{L_c}(\vartheta_{1}),\ldots,\mathcal{L_c}(\vartheta_{J})\}$ and, for *some parameter*, or *parameter vector*, $\theta:= \theta_{1:N}:=\{\theta_1,\ldots,\theta_N\}$, define $\mathcal{L_c}(\theta_{1:N}) := \{\mathcal{L_c}(\theta_{1}),\ldots,\mathcal{L_c}(\theta_{N})\}$. 

the posterior distribution of $\mathcal{L_c}(\theta_{1:N}\vert\{\mathcal{L_c}(\vartheta_{1:J}) = \mathcal{l_c}(\vartheta_{1:J})\})$  is also multivariate Gaussian,

$$
\mathcal{L_c}(\theta_{1:N}\vert\mathcal{D_c}) \sim \mathcal{N}(\mu_c(\theta_{1:N}), \Sigma_c(\theta_{1:N}))
$$

with,

$$
\mu_c(\theta_{1:N})\ =\ m_c(\theta_{1:N}) + K_{*}^{T}\tilde{K}^{-1}(\mathcal{L_c}(\vartheta_{1:J}) - m_c(\vartheta_{1:J})) \\
\Sigma_c(\theta_{1:N})\ = \ K_{*,*}-K_{*}^{T}\tilde{K}^{-1}K_{*}
$$


 where $K_{*,*} =K(\theta_{1:N},\theta_{1:N}), K_{*} = K(\vartheta_{1:J},\theta_{1:N}), \tilde{K}=K(\vartheta_{1:J},\vartheta_{1:J})$.

(다변량 정규분포를 생각하면 clear)

###### Merging the subposteriors

$\pi(\theta) \propto \Pi_{c=1}^{C}\pi_c(\theta)$ and $\mathcal{L_c}(\theta) \propto \mathcal{GP}(\cdot,\cdot)$ where $c=1,\ldots,C$

$$
\mathcal{L}(\theta\vert\mathcal{D}) \propto \sum_{c=1}^{C}[\mathcal{L_c}(\theta\vert\mathcal{D_c})] = \mathcal{GP}(\sum_{c=1}^{C}\mu_c(\theta),\sum_{c=1}^{C}\Sigma_c(\theta))
$$


#### Approximating the full posterior



###### 1. The expected posterior density

approximate the full posterior density by its expectation under the Gaussian-process approximation(and the property of log-normal distribution):

$$
\hat{\pi}_E(\theta) \propto \mathbb{E}[\exp(\mathcal{L}(\theta)\vert\mathcal{D}] = \exp\{\sum_{c=1}^{C}\mu_c(\theta)+\frac{1}{2}\sum_{c=1}^{C}\Sigma_c(\theta) \}
$$

Use HMC to obtain an approximate sample, $\{\theta_i\}_{i=1}^{N}$ from $\hat{\pi}_E(\theta)$.

$$
\begin{align*}
\nabla\log\hat{\pi}_E(\theta) &= \sum_{c=1}^{C}\frac{\partial}{\partial\theta}\mu_c(\theta)+\frac{1}{2}\sum_{c=1}^{C}\frac{\partial}{\partial\theta}\Sigma_c(\theta) \\
=& \sum_{c=1}^{C}\frac{\partial}{\partial\theta}m(\theta)+\frac{\partial K_{*}^{T}}{\partial\theta}\tilde{K}^{-1}(\mathcal{l_c}(\vartheta_{1:J}) - m(\vartheta_{1:J}))+ \frac{1}{2}\frac{\partial}{\partial\theta}K_{*,*} - \frac{\partial}{\partial\theta}K_{*}^T\tilde{K}^{-1}
\end{align*}
$$


###### 2. Distributed importance sampling



have to still correct for inaccuracies in $\hat{\pi}_{E}$ using importance sampling while spreading the computational burden across all $C$ cores. 
Each subposterior is evaluated at the same set of $\theta$ values, allowing them to be combined exactly. 
In contrast, the original HMC runs, performed on each individual subposterior, created a different set of $\theta$ values for each subposterior so that a straightforward combination was not possible.



Each value from the sample, $\theta_i$, is then associated with an unnormalised weight, $w(\theta_i) = \frac{\pi(\theta_i)}{\hat{\pi}_E(\theta_i)}$. Define $\hat{Z}_N$ and $w_N(\theta)$ provides an approximation $\mathbb{\hat{E}}_N(h)$ to $\mathbb{E}_{\pi}[h(\theta)]$(By strong law of large numbers). 



Since the unknown normalising constants for both $\pi$ and $\hat{\pi}_E$ appear in both the numerator and the denominator of this expression, **they are not needed**. 

![스크린샷 2020-07-21 오후 3.26.45](/Users/bayeslab/Desktop/스크린샷 2020-07-21 오후 3.26.45.png)





######  Gaussian-process importance sampler



we are interested in $I_h:=\mathbb{E}_{\pi}[h(\theta)] = \frac{1}{Z}\int\pi(\theta)h(\theta)d\theta $. 

Consider approximating this with

$$
I_h(l):= \frac{1}{Z(l)}\int\exp\{l(\theta)\}h(\theta)d\theta
$$

where $l$ is a realization of $\mathcal{L}$ from the distribution, $\mathcal{GP}(\sum_{c=1}^{C}\mu_c(\theta),\sum_{c=1}^{C}\Sigma_c(\theta))$ and $Z(l):= \int\exp\{l(\theta)\}d\theta$

Consider the hypothetical scenario where it is possible to store $l$ completely and evaluate $I_h(l)$. 

A set of $M$ realisations of $\mathcal{L}$, $\{l_m\}^M_{m=1}$ would lead to $M$ associated estimates of $I_h$, $\{I_h(l_m))\}^M_{m=1}$, which would approximate the posterior distribution of $I_h$ under $\mathcal{GP}(\sum_{c=1}^{C}\mu_c(\theta),\sum_{c=1}^{C}\Sigma_c(\theta))$. 



The mean of these would then target, the posterior expectation, 

$$
I^{\mathbb{E}}_h\ := \ \mathbb{E} [\frac{1}{Z(\mathcal{L})}\int h(\theta)\exp(\mathcal{L(\theta)})d\theta]
$$


Unfortunately, it is not possible to store the infinite-dimensional object, and even if it were, for moderate dimensions, numerical evaluation of $I_h(l)$ would be computationally infeasible. 



Instead, use importance sampling. Consider a proposal distribution $q(\theta)$ that approximately mimics the true posterior distribution, $\pi(\theta)$ and sample $N$ independent points from it: $\theta_{1:N} := (\theta_1,\ldots,\theta_N )$. 



For each $m \in \{1,\ldots,M\}$ we then sample the finite-dimensional object $(m(\theta_1),\ldots,m(\theta_N ))$ from the joint distribution of the $\mathcal{GP}(\sum_{c=1}^{C}\mu_c(\theta),\sum_{c=1}^{C}\Sigma_c(\theta))$. For each such realisation we then construct an approximation to the normalisation constant and to $I_h(l)$:

$$
\hat{Z}(l_m) := \frac{1}{N}\sum_{i=1}^{N}\bar{w}(\theta_i;l_m)
$$

and

$$
\hat{I}_h(l_m):= \frac{1}{N\hat{Z}(l_m)}\sum_{i=1}^{N}\bar{w}(\theta_i;l_m)h(\theta_i)
$$


where $\bar{w}(\theta_i;l_m) := \frac{\exp\{l(\theta)\}}{q(\theta)}$ . 



The set $\{\hat{I}_h(l_m)\}^M_{m=1}$ is then used in place of $\{I_h(l_m)\}^M_{m=1}$for posterior inference on $I_h$. 

For the specific case of $I^{\mathbb{E}}_h$ a simplified expression for the approximation may be derived: 

$$
\hat{I}^{\mathbb{E}}_h = \frac{1}{N}\sum_{i=1}^{N}w_ih(\theta_i)
$$

where

$$
\frac{1}{Mq(\theta_i)}\sum_{i=1}^{N}\frac{\exp\{l_m(\theta_i)\}}{\hat{Z}(l_m)}
$$


 Algorithm 2 creates point estimates based upon this.



The proposal density $q(\theta_i)$ should be a good approximation to the posterior density. Let the proposal density $q(\theta_i)$ be a multivariate Student-t distribution on 5 degrees of freedom with mean and variance matching those of the Gaussian posterior that would arise given the mean and variance of each subposterior and if each sub-posterior were Gaussian.



