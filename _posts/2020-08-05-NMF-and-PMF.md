---
title: "NMF and PMF"
date: 2020-08-04 00:00:28 -0400
categories: Machine-learning
use_math: true
---

이 글은 Carneige Mellon University의 Yiming Yang 교수님의 Machine Learning for Text Mining의 수업 중 일부분을 정리한 것입니다. 


**증명 과정이 저도 매우 마음에 들지 않아 추후 업데이트 할 예정입니다.**


## Non-negative Matrix Factorization and Probabilistic Matrix Factorization

Denote that NMF is Non-negative Matrix Factorization, PMF is Probabilistic Matrix Factorization

##### problem definition & difference



###### Non-negative Matrix Factorization



- Decompose non-negative matrix X into non-negative W and H such that
  $$
  X \approx WH
  $$
  

- SVD  requires **orthogonal** factorization

- NMF requires **non-negative** factorization

- SVD는 $U_{r}\Sigma_{r}V_r^T$로 해줄때, orthogonal basis를 이용하지만, NMF는 단순히 모든 element들이 양수인 W,H로 분해하는 방법이다.

  

NMF, VQ, PCA를 image processing할때 쓰는데, NMF는 original 이미지를 approximate한다. 


These properties lead to different factorization results and require different algorithms for optimization.


대략적으로 이걸 왜 할까라는 생각을 해보면, 대략적으로 이렇게 생각 할 수 있습니다. 


**The reason why NMF has become so popular is because of its ability to automatically extract sparse and easily interpretable factors.**


대략적인 행렬로 분해하는것이기 때문에 결과 값으로 얻은 행렬들의 곱이 원래 행렬이 얼마나 "가까운"지를 측정해야한다.


**Definition1.** Frobenius norm of a matrix (Euclidean distance) is 

$$
\Vert X - \hat{X}^2\Vert = \sum_{i,j} (X_{ij} - \hat{X_{ij}})^2
$$


**Definitiion2.** Divergence of a matrix 

$$
D(X\Vert\hat{X}) = \sum_{i,j}(X_{ij}log\frac{X_{ij}}{\hat{X_{ij}}} - X_{ij} + \hat{X_{ij}})
$$


다른 metric들도 굉장히 많은데, 문제마다 metric은 상황에 맞게 알맞은 걸 적용하면 된다.

Two Optimization Problems

$$
1.\ \min_{W,H}\Vert X - WH\Vert^2\ \ \text{s.t}\ \ W,H \geq 0 \\
2.\ \min_{W,H}D(X\Vert W,H)^2\ \ \text{s.t}\ \ W,H \geq 0
$$

These problems are convex in W only if H is fixed or vise versa, but not convex in both variables together. Only local minima can be guaranteed.



*Udapting Rule wrt 1.*

$$
W_{ik} :=  W_{ik}\frac{(XH^T)_{ij}}{(WHH^T)_{ik}}, \\
H_{ik} :=  H_{ik}\frac{(W^TX)_{ij}}{(W^TWH)_{ik}}
$$


*Udapting Rule wrt 2.*

$$
W_{ik} :=  W_{ik}\frac{\sum_{j} H_{kj}X_{ij}/(WH)_{ij}}{\sum_{j'}H_{j'k}}, \\
 H_{kj} :=  H_{kj}\frac{\sum_{i}W_{ik}X_{ij}/(WH)_{ij}}{\sum_{i'}W_{i'k}}
$$




왜 이 알고리즘이 잘 될까? (gradient descent이기 때문)



- To show it equivalent to a specific case of gradient descent (GD)

- To prove that the GD steps are sufficiently small

  

업데이트 하는 방법은 start with an initial value of $W_0$ and $H_0$. Then, 2 update equations one for $W$ and $H$. update $W^{t+1}$ using $H_t$ and $H^{t+1}$ using $W_t$. 

Problem 1의 H에대해서 증명해보자. 

let $\eta_{kj} = \frac{H_{kj}}{(W^TWH)_{kj}}$ then,

$$
\begin{align*}
H_{kj} &= \frac{H_{kj}}{(W^TWH)_{kj}} \\
       &= H_{kj} + \eta_{kj}[(W^TX)_{kj}-(W^TWH)_{kj}]
\end{align*}
$$


We want to show that $[(W^TX)_{kj}-(W^TWH)_{kj}] = -c[\nabla_{H}f(W,H)]_{kj}$,  where $c$ is a constant.

The cost function($f$) is Frobenius norm(**Definition1**).



**Remark**) $(WH)_{ij} = W_iH_j$, $f = \sum_{i,j}\Vert X_{ij}-(WH)_{ij}\Vert^2$

$$
\frac{\partial f}{\partial H} = -2W^T(X-WH) = -2(W^TX-W^TWH)
$$


Finally, $[(W^TX)_{kj}-(W^TWH)_{kj}] = -\frac{1}{2}[\nabla_{H}f(W,H)]_{kj}$



제시한 $\eta$값들이 충분히 작기 때문에 위에 언급한 알고리즘들은 수렴한다.



###### Probabilistic Matrix Factorization



- NMF and SVD minimize the squired errors as

  
  $$
  \hat{X} = \underset{U,V}{\mathbb{argmin}}\Vert X-UV\Vert_F^2
  $$
  

- PMF maximizes the posterior probability of the model as

  
  $$
  \hat{X} = \underset{U,V}{\mathbb{argmax}}\Vert X-UV\Vert_F^2
  $$
  where $X \in R^{N \times M}, U \in R^{D \times N}, V \in R^{D \times M}$

  

let $U,V$ be latent user and movie feature matrices, with column vectors $U_i$ and $V_j$ representing user-specific and movie-specific latent feature vectors respectively.

결국 이것도 왜 하는 것일까 생각해보면 다음의 답을 얻을 수 있습니다. 



- factor a matrix with many missing values, including many sparse rows, with the hope of using the known values to provide information about the missing values.
- PMF provides a probabilistic approach using Gaussian assumptions on the known data and the factor matrices. We can further constrain priors to improve the algorithm, especially in the case of sparse rows.
  
  

*Properties of PMF*

- Using Gaussian models to define the objective function
- Optimizing U and V alternately (with closed-form solutions, i.e., gradient descent is not needed) 
- Scaling linearly in the number of observations (non-zero elements) in the input matrix (sparse matrix가 input으로 들어와도 cost가 크지는 않음)
- Very popular in large applications of matrix factorization
  
  

The generative process



- $U_{i} \sim N(0, \sigma_U^2 \boldsymbol{I})$, where $U_{i} \in R^D$ 

- $V_{j} \sim N(0, \sigma_V^2 \boldsymbol{I})$, where $U_{j} \in R^D$

- $X_{ij} \sim N(0, \sigma^2 \boldsymbol{I})$, where $X_{ij} \in R$ 

- Then the posterior distribution is 

  
  $$
  p(U,V|X,\sigma_U^2,\sigma_V^2,\sigma^2) \propto \prod_{i=1}^{N}N(U_i|0,\sigma_U^2 \boldsymbol{I})\prod_{j=1}^{M}N(V_j|0,\sigma_V^2 \boldsymbol{I}) \prod_{i,j}[N(X_{ij}|U_i^TV_j,\sigma \boldsymbol{I}])^{I_{ij}}
  $$



Take the log to the posterior distribution (log is monotone increase function; computation on the log can be easier)


$$
\begin{align*}
\log p(U,V|X,\sigma_U^2,\sigma_V^2,\sigma^2) &= \sum_{i,j}I_{ij}\log[N(X_{ij}|U_i^TV_j,\sigma^2 \boldsymbol{I})] + \sum_{i=1}^{N}\log N(U_i|0,\sigma_U^2 \boldsymbol{I}) + \sum_{j=1}^{M} \log N(V_j|0,\sigma_V^2 \boldsymbol{I}) \\
    &= -\frac{1}{2\sigma^2}\sum_{i,j}I_{ij}[(X_{ij}-U_i^TV_j)^2)] -\frac{1}{\sigma_U^2}\sum_{i=1}^{N} (U_i^TU_i) - \frac{1}{\sigma_V^2}\sum_{j=1}^{M}(V_j^TV_j) + constant\\
    &= -E + constant
\end{align*}
$$


then the optimization problem is 


$$
\min_{U,V}E = \min_{U,V}[\frac{1}{2\sigma^2}\sum_{i,j}I_{ij}[(X_{ij}-U_i^TV_j)^2)] +\frac{1}{2\sigma_U^2}\sum_{i=1}^{N}(U_i^TU_i) + \frac{1}{2\sigma_V^2}\sum_{j=1}^{M}(V_j^TV_j)]
$$


the first term means sum of squared errors, and the others mean regularization. 

We can optimize this through Greedy Search



- Optimize U (when fixing V) and V (when fixing U), alternately. (번갈아 가면서 업데이트 예를 들자면 U를 먼저 fix해주고 V를 계산, 계산된 V를 다시 U를 업데이트할때 이용하고 이 과정을 번갈아 가면서 업데이트)
-  그러나 Global maximum is not guaranteed.
- Computation time is linear in the number of non-zero elements in $X$ because update is only computed for $I_{ij} = 1$
- Other variants of the PMF are also developed, e.g.,Constrained PMF using the logistic transformation $g(U_i^TV_j)$, instead of $U_i^TV_j$ (detail은 페이퍼에).
  
  

$\sigma$들에 대해서 베이지안 hyperparameter를 추가해주게 되면,



- 원래 버전은 밑의 식을 최대화하는 문제이다 

  
  $$
  \log p(U,V|X,\sigma_U^2,\sigma_V^2,\sigma^2) = \log p(U|\sigma_U^2) + \log p(V|\sigma_V^2) + \log p(X|U,V,\sigma^2)
  $$
  

- 아래의 식을 최대화 하는 $U,V$를 찾으면 된다.

  
  $$
  \log p(U,V,\sigma_U^2,\sigma_V^2,\sigma^2|X) = \log p(U|\sigma_U^2) + \log p(V|\sigma_V^2) + \log p(X|U,V,\sigma^2) + \log p(\sigma_U^2) + \log p(\sigma_V^2) + \log p(\sigma^2)
  $$
  

그리고 PMF 방식이 더 짱짱인걸 실험을 통해서 확인할 수 있었다.


Data Set은



- Randomly sampled subsets from Netflix during 1998 ~ 2005
- Training set with 480K users, 17,770 movies, over 100M ratings
- Validation set with 1.4M ratings
- Test set of 2.8 M user/movie pairs with the withheld ratings



그랬더니 dimension of latent factor가 커질수록 PMF의 Root MSE가 훨씬 더 좋았다.
