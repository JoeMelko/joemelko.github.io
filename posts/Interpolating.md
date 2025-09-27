# Interpolating Between SGD and Muon

My interest in optimizers originates from my desire to rigorously study how training data *causes* models to develop certain characteristics. To do this rigorously, we must measure how training data results in changes to the model. This led me to work on influence function approximations and gradient based data selection methods. It is impossible to work on this without your mind wandering toward the study of how to transform gradients into model updates... optimizers.

> The use of words implying causality is purposeful. This is applied very loosely within the dark art of training massive models. Causality is a north star here; it is not a claim anything has approached it yet.

## Muon

Much of the work on optimizers failed to make it out of theory until Muon. It first garnered excitement with small scale results on cifar-10, before progressing to gpt2-scale models, then to billions and finally >1 trillion parameters [cite]. 

The key insight behind Muon's succes lies in performing optimization under a different geometry than what is used in SGD / Adam [cite]. While SGD/Adam step in the steepest gradient direction under standard Euclidean geometry on parameters, Muon steps in the direction that is steepest under spectral norm contraints. More concretely, consider a gradient $G = U\Sigma V^T$. SGD applies the full gradient:

$W_{new} = W - \eta U\Sigma V^T$

Muon sets all singular values to 1, effectively applying:

$W_{new} = W - \eta UV^T$

This transforms $U\Sigma V^T \rightarrow UV^T$.

## Steepest Descent Under Schatten-$p$

Both of these norm constraints are instantiations of the Schatten-$p$ norm for different $p$. The Schatten-$p$ norm is defined as 

$||A||_p = (\sum_i \sigma_i^p)^{1/p}$

where $\sigma_i$ are the singular values of matrix $A$. If we set $p=2$, we recover the frobenius norm; if we define $p=\infty$, we recover the spectral norm. However, we are not constrained to these 2 choices of p. In fact, we are free to interpolate between these instantiations (and can actually choose $p<2$ as well). 

By modifying $p$, we are choosing how to adjust the singular value spectrum before making updates, for $p=\infty$ updates are maximally smooth and full rank, for $p=1$ updates are maximally spiky and rank 1. For $p=2$ (SGD) the singular values are left unchanged.

## Efficient Implementation with Newton-Schulz

It is far too slow to directly compute the singular values. Luckily, we can make approximate updates to the singular value spectrum using iterative Newton-Schulz steps. [cite] notes that

$$
\begin{aligned}
G' &:= aG + b(GG^\top)G + c(GG^\top)^2 G \\
   &= \big(aI + b(GG^\top) + c(GG^\top)^2\big)G \\
   &= \big(aI + bUS^2U^\top + cUS^4U^\top\big)USV^\top \\
   &= U\big(aS + bS^{3} + cS^{5}\big)V^\top
\end{aligned}
$$

In the original work, this is used to iteratively force all values between 0 and 1 to 1. However, we can choose to fit the coefficients to any spectrum we desire.

## A Family Of Muon-Style Optimizers

Putting this together, we can construct a Muon-style update for any desired singular value spectrum. Let $f_\text{target}(s)$ denote the singular-value transform we want to apply to $G=USV^\top$, i.e., we seek an update $G_\text{target} = U f_\text{target}(S) V^\top$. The Newton–Schulz-style polynomial gives us a convenient parameterization

$$
U\big(aS + bS^3 + cS^5\big)V^\top,
$$

so our task is simply to fit the scalar coefficients $a,b,c$ so that, over the range of singular values we care about,

$$
s\,\big(as + b s^3 + c s^5\big) \approx f_\text{target}(s).
$$

In practice we fit $a,b,c$ by least-squares on a small grid of $s$, then compute
$G' = aG + b(GG^\top)G + c(GG^\top)^2G$.

This recovers familiar cases and a continuum:
- $f_\text{target}(s)=s$ (SGD): $(a,b,c)=(1,0,0)$
- $f_\text{target}(s)=1$ (Muon-like flattening)
- $f_\alpha(s)=s^{\alpha},\ \alpha\in[0,1]$ interpolates Muon $\to$ SGD
- $f_\text{top}(s)=\mathbb{1}[s=s_{\max}]$ (Schatten-1/top singular): rank-1 update $u_1 v_1^\top$

## Some Results


