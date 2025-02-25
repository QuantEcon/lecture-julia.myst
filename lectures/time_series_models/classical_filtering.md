---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.11
---

(classical_filtering)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Classical Filtering With Linear Algebra

```{contents} Contents
:depth: 2
```

## Overview

This is a sequel to the earlier lecture {doc}`Classical Control with Linear Algebra <../time_series_models/classical_filtering>`.

That lecture used linear algebra -- in particular,  the [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)  -- to formulate and solve a class of linear-quadratic optimal control problems.

In this lecture, we'll be using a closely related decomposition, the [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) , to solve linear prediction and filtering problems.

We exploit the useful fact that there is an intimate connection between two superficially different classes of problems:

* deterministic linear-quadratic (LQ) optimal control problems
* linear least squares prediction and filtering problems

The first class of problems involves no randomness, while the second is all about randomness.

Nevertheless,  essentially the same mathematics  solves both type of problem.

This connection, which is often termed "duality," is present whether one uses "classical" or "recursive" solution procedures.

In fact we saw duality at work earlier when we formulated control and prediction problems recursively in lectures {doc}`LQ dynamic programming problems <../dynamic_programming/lqcontrol>`, {doc}`A first look at the Kalman filter <../introduction_dynamics/kalman>`, and {doc}`The permanent income model <../dynamic_programming/perm_income>`.

A useful consequence of duality is that

* With every LQ control problem there is implicitly affiliated a linear least squares prediction or filtering problem.
* With every linear least squares prediction or filtering problem there is implicitly affiliated a LQ control problem.

An understanding of these connections has repeatedly proved useful in cracking interesting applied problems.

For example, Sargent {cite}`Sargent1987` [chs. IX, XIV] and Hansen and Sargent {cite}`HanSar1980` formulated and solved control and filtering problems using $z$-transform methods.

In this lecture we investigate these ideas using mostly elementary linear algebra.

### References

Useful references include {cite}`Whittle1963`, {cite}`HanSar1980`, {cite}`Orfanidisoptimum1988`, {cite}`Athanasios1991`, and {cite}`Muth1960`.


## Infinite Horizon Prediction and Filtering Problems

We pose two related prediction and filtering problems.

We let $Y_t$ be a univariate $m^{\rm th}$ order moving average, covariance stationary stochastic process,

```{math}
:label: eq_24

Y_t = d(L) u_t
```

where $d(L) = \sum^m_{j=0} d_j L^j$, and $u_t$ is a serially uncorrelated stationary random process satisfying

```{math}
:label: eq_25

\begin{aligned}
    \mathbb{E} u_t &= 0\\
    \mathbb{E} u_t u_s &=
    \begin{cases}
        1 & \text{ if } t = s \\
        0 & \text{ otherwise}
    \end{cases}
\end{aligned}
```

We impose no conditions on the zeros of $d(z)$.

A second covariance stationary process is $X_t$ given by

```{math}
:label: eq_26

X_t = Y_t + \varepsilon_t
```

where $\varepsilon_t$ is a serially uncorrelated stationary
random process with $\mathbb{E} \varepsilon_t = 0$ and $\mathbb{E} \varepsilon_t \varepsilon_s$ = $0$ for all distinct $t$ and $s$.

We also assume that $\mathbb{E} \varepsilon_t u_s = 0$ for all $t$ and $s$.

The **linear least squares prediction problem** is to find the $L_2$
random variable $\hat X_{t+j}$ among linear combinations of
$\{ X_t,\  X_{t-1},
\ldots \}$ that minimizes $\mathbb{E}(\hat X_{t+j} - X_{t+j})^2$.

That is, the problem is to find a $\gamma_j (L) = \sum^\infty_{k=0} \gamma_{jk}\, L^k$ such that $\sum^\infty_{k=0} \vert \gamma_{jk} \vert^2 < \infty$ and $\mathbb{E} [\gamma_j \, (L) X_t -X_{t+j}]^2$ is minimized.

The **linear least squares filtering problem** is to find a $b\,(L) = \sum^\infty_{j=0} b_j\, L^j$ such that $\sum^\infty_{j=0}\vert b_j \vert^2 < \infty$ and $\mathbb{E} [b\, (L) X_t -Y_t ]^2$ is minimized.

Interesting versions of these problems related to the permanent income theory were studied by {cite}`Muth1960`.

### Problem formulation

These problems are solved as follows.

The covariograms of $Y$ and $X$ and their cross covariogram are, respectively,

```{math}
:label: eq_27

\begin{aligned}
    C_X (\tau) &= \mathbb{E}X_t X_{t-\tau} \\
    C_Y (\tau) &= \mathbb{E}Y_t Y_{t-\tau}  \qquad \tau = 0, \pm 1, \pm 2, \ldots \\
    C_{Y,X} (\tau) &= \mathbb{E}Y_t X_{t-\tau}
\end{aligned}
```

The covariance and cross covariance generating functions are defined as

```{math}
:label: eq_28

\begin{aligned}
    g_X(z) &= \sum^\infty_{\tau = - \infty} C_X (\tau) z^\tau \\
    g_Y(z) &= \sum^\infty_{\tau = - \infty} C_Y (\tau) z^\tau \\
    g_{YX} (z) &= \sum^\infty_{\tau = - \infty} C_{YX} (\tau) z^\tau
\end{aligned}
```

The generating functions can be computed by using the following facts.

Let $v_{1t}$ and $v_{2t}$ be two mutually and serially uncorrelated white noises with unit variances.

That is, $\mathbb{E}v^2_{1t} = \mathbb{E}v^2_{2t} = 1, \mathbb{E}v_{1t} = \mathbb{E}v_{2t} = 0, \mathbb{E}v_{1t} v_{2s} = 0$ for all $t$ and $s$, $\mathbb{E}v_{1t} v_{1t-j} = \mathbb{E}v_{2t} v_{2t-j} = 0$ for all $j \not = 0$.

Let $x_t$ and $y_t$ be two random process given by

$$
\begin{aligned}
    y_t &= A(L) v_{1t} + B(L) v_{2t} \\
    x_t &= C(L) v_{1t} + D(L) v_{2t}
\end{aligned}
$$

Then, as shown for example in {cite}`Sargent1987` [ch. XI], it is true that

```{math}
:label: eq_29

\begin{aligned}
    g_y(z) &= A(z) A(z^{-1}) + B (z) B(z^{-1}) \\
    g_x (z) &= C(z) C(z^{-1}) + D(z) D(z^{-1}) \\
    g_{yx} (z) &= A(z) C(z^{-1}) + B(z) D(z^{-1})
\end{aligned}
```

Applying these formulas to {eq}`eq_24` -- {eq}`eq_27`, we have

```{math}
:label: eq_30

\begin{aligned}
    g_Y(z) &= d(z)d(z^{-1}) \\
    g_X(z) &= d(z)d(z^{-1}) + h\\
    g_{YX} (z) &= d(z) d(z^{-1})
\end{aligned}
```

The key step in obtaining solutions to our problems is to factor the covariance generating function  $g_X(z)$ of $X$.

The solutions of our problems are given by formulas due to Wiener and Kolmogorov.

These formulas utilize the Wold moving average representation of the $X_t$ process,

```{math}
:label: eq_31

X_t = c\,(L)\,\eta_t
```

where $c(L) = \sum^m_{j=0} c_j\, L^j$, with

```{math}
:label: eq_32

c_0 \eta_t
= X_t - \mathbb{\hat E} [X_t \vert X_{t-1}, X_{t-2}, \ldots]
```

Here $\mathbb{\hat E}$ is the linear least squares projection operator.

Equation {eq}`eq_32`  is the condition that $c_0 \eta_t$ can be the one-step ahead error in predicting $X_t$ from its own past values.

Condition {eq}`eq_32`  requires that $\eta_t$ lie in the closed
linear space spanned by $[X_t,\  X_{t-1}, \ldots]$.

This will be true if and only if the zeros of $c(z)$ do not lie inside the unit circle.

It is an implication of {eq}`eq_32` that $\eta_t$ is a serially
uncorrelated random process, and that a normalization can be imposed so
that $\mathbb{E}\eta_t^2 = 1$.

Consequently, an implication of {eq}`eq_31`  is
that the covariance generating function of $X_t$ can be expressed
as

```{math}
:label: eq_33

g_X(z) = c\,(z)\,c\,(z^{-1})
```

It remains to discuss how $c(L)$ is to be computed.

Combining {eq}`eq_29`  and {eq}`eq_33`  gives

```{math}
:label: eq_34

d(z) \,d(z^{-1}) + h = c \, (z) \,c\,(z^{-1})
```

Therefore, we have already showed constructively how to factor the covariance generating function $g_X(z) = d(z)\,d\,(z^{-1}) + h$.

We now introduce the **annihilation operator**:

```{math}
:label: eq_35

\left[
    \sum^\infty_{j= - \infty} f_j\, L^j
\right]_+
\equiv \sum^\infty_{j=0} f_j\,L^j
```

In words, $[\phantom{00}]_+$ means "ignore negative powers of $L$".

We have defined the solution of the prediction problem as $\mathbb{\hat E} [X_{t+j} \vert X_t,\, X_{t-1}, \ldots] = \gamma_j\, (L) X_t$.

Assuming that the roots of $c(z) = 0$ all lie outside the unit circle, the Wiener-Kolmogorov formula for $\gamma_j (L)$ holds:

```{math}
:label: eq_36

\gamma_j\, (L) =
\left[
    {c (L) \over L^j}
\right]_+ c\,(L)^{-1}
```

We have defined the solution of the filtering problem as $\mathbb{\hat E}[Y_t \mid X_t, X_{t-1}, \ldots] = b (L)X_t$.

The Wiener-Kolomogorov formula for $b(L)$ is

$$
b(L) = \left({g_{YX} (L) \over c(L^{-1})}\right)_+ c(L)^{-1}
$$

or

```{math}
:label: eq_37

b(L) = \left[ {d(L)d(L^{-1}) \over c(L^{-1})} \right]_+ c(L)^{-1}
```

Formulas {eq}`eq_36` and {eq}`eq_37`  are discussed in detail in  {cite}`Whittle1983` and {cite}`Sargent1987`.

The interested reader can there find several examples of the use of these formulas in economics
Some classic examples using these formulas are due to {cite}`Muth1960`.

As an example of the usefulness of formula {eq}`eq_37`, we let $X_t$ be a stochastic process with Wold moving average representation

$$
X_t = c(L) \eta_t
$$

where $\mathbb{E}\eta^2_t = 1, \hbox { and } c_0 \eta_t = X_t - \mathbb{\hat E} [X_t \vert X_{t-1}, \ldots], c (L) = \sum^m_{j=0} c_j L$.

Suppose that at time $t$, we wish to predict a geometric sum of future $X$'s, namely

$$
y_t \equiv \sum^\infty_{j=0} \delta^j X_{t+j} = {1 \over 1 - \delta L^{-1}}
X_t
$$

given knowledge of $X_t, X_{t-1}, \ldots$.

We shall use {eq}`eq_37`  to obtain the answer.

Using the standard formulas  {eq}`eq_29`, we have that

$$
\begin{aligned}
    g_{yx}(z) &= (1-\delta z^{-1})c(z) c (z^{-1}) \\
    g_x (z) &= c(z) c (z^{-1})
\end{aligned}
$$

Then {eq}`eq_37`  becomes

```{math}
:label: eq_38

b(L)=\left[{c(L)\over 1-\delta L^{-1}}\right]_+ c(L)^{-1}
```

In order to evaluate the term in the annihilation operator, we use the following result from {cite}`HanSar1980`.

**Proposition** Let

* $g(z) = \sum^\infty_{j=0} g_j \, z^j$ where $\sum^\infty_{j=0} \vert g_j \vert^2 < + \infty$
* $h\,(z^{-1}) =$ $(1- \delta_1 z^{-1}) \ldots (1-\delta_n z^{-1})$, where $\vert \delta_j \vert < 1$, for $j = 1, \ldots, n$

Then

```{math}
:label: eq_39

\left[{g(z)\over h(z^{-1})}\right]_+ = {g(z)\over h(z^{-1})} - \sum^n_{j=1}
\ {\delta_j g (\delta_j) \over \prod^n_{k=1 \atop k \not = j} (\delta_j -
\delta_k)} \ \left({1 \over z- \delta_j}\right)
```

and, alternatively,

```{math}
:label: eq_40

\left[
    {g(z)\over h(z^{-1})}
\right]_+
=\sum^n_{j=1} B_j
\left(
    {zg(z)-\delta_j g (\delta_j) \over z- \delta_j}
\right)
```

where $B_j = 1 / \prod^n_{k=1\atop k+j} (1 - \delta_k / \delta_j)$.

Applying formula {eq}`eq_40`  of the proposition to evaluating  {eq}`eq_38`  with $g(z) = c(z)$ and $h(z^{-1}) = 1 - \delta z^{-1}$ gives

$$
b(L)=\left[{Lc(L)-\delta c(\delta)\over L-\delta}\right] c(L)^{-1}
$$

or

$$
b(L) =
\left[
    {1-\delta c (\delta) L^{-1} c (L)^{-1}\over 1-\delta L^{-1}}
\right]
$$

Thus, we have

```{math}
:label: eq_41

\mathbb{\hat E}
\left[
    \sum^\infty_{j=0} \delta^j X_{t+j}\vert X_t,\, x_{t-1},
    \ldots
\right]  =
\left[
    {1-\delta c (\delta) L^{-1} c(L)^{-1} \over 1 - \delta L^{-1}}
\right]
\, X_t
```

This formula is useful in solving stochastic versions of problem 1 of lecture {doc}`Classical Control with Linear Algebra <lu_tricks>` in which the randomness emerges because $\{a_t\}$ is a stochastic
process.

The problem is to maximize

```{math}
:label: eq_42

\mathbb{E}_0
\lim_{N \rightarrow \infty}\
\sum^N_{t-0} \beta^t
\left[
    a_t\, y_t - {1 \over 2}\ hy^2_t-{1 \over 2}\ [d(L)y_t]^2
\right]
```

where $\mathbb{E}_t$ is mathematical expectation conditioned on information
known at $t$, and where $\{ a_t\}$ is a covariance
stationary stochastic process with Wold moving average representation

$$
a_t = c(L)\, \eta_t
$$

where

$$
c(L) = \sum^{\tilde n}_{j=0} c_j L^j
$$

and

$$
\eta_t =
a_t - \mathbb{\hat E} [a_t \vert a_{t-1}, \ldots]
$$

The problem is to maximize {eq}`eq_42`  with respect to a contingency plan
expressing $y_t$ as a function of information known at $t$,
which is assumed to be
$(y_{t-1},\  y_{t-2}, \ldots, a_t, \ a_{t-1}, \ldots)$.

The solution of this problem can be achieved in two steps.

First, ignoring the uncertainty, we can solve the problem assuming that $\{a_t\}$ is a known sequence.

The solution is, from above,

$$
c(L) y_t = c(\beta L^{-1})^{-1} a_t
$$

or

```{math}
:label: eq_43

(1-\lambda_1 L) \ldots (1 - \lambda_m L) y_t
= \sum^m_{j=1} A_j
\sum^\infty_{k=0} (\lambda_j \beta)^k\, a_{t+k}
```

Second, the solution of the problem under uncertainty is obtained by
replacing the terms on the right-hand side of the above expressions with
their linear least squares predictors.

Using {eq}`eq_41` and {eq}`eq_43`, we have
the following solution

$$
(1-\lambda_1 L) \ldots (1-\lambda_m L) y_t
=
\sum^m_{j=1} A_j
 \left[
     \frac{1-\beta \lambda_j \, c (\beta \lambda_j) L^{-1} c(L)^{-1} }
     { 1-\beta \lambda_j L^{-1} }
 \right] a_t
$$

## Finite Dimensional Prediction

Let $(x_1, x_2, \ldots, x_T)^\prime = x$ be a $T \times 1$ vector of random variables with mean $\mathbb{E} x = 0$ and covariance matrix $\mathbb{E} xx^\prime = V$.

Here $V$ is a $T \times T$ positive definite matrix.

We shall regard the random variables as being
ordered in time, so that $x_t$ is thought of as the value of some
economic variable at time $t$.

For example, $x_t$ could be generated by the random process described  by the Wold representation presented in equation {eq}`eq_31`.

In this case, $V_{ij}$ is given by the coefficient on $z^{\mid i-j \mid}$ in the expansion of $g_x (z) = d(z) \, d(z^{-1}) + h$, which equals
$h+\sum^\infty_{k=0} d_k d_{k+\mid i-j \mid}$.

We shall be interested in constructing $j$ step ahead linear least squares predictors of the form

$$
\mathbb{\hat E}
\left[
    x_T\vert x_{T-j}, x_{T-j + 1}, \ldots, x_1
\right]
$$

where $\mathbb{\hat E}$ is the linear least squares projection operator.

The solution of this problem can be exhibited by first constructing an
orthonormal basis of random variables $\varepsilon$ for $x$.

Since $V$ is a positive definite and symmetric, we know that there
exists a (Cholesky) decomposition of $V$ such that

$$
V = L^{-1} (L^{-1})^\prime
$$

or

$$
L \, V \, L^\prime = I
$$

where $L$ is lower-trangular, and therefore so is $L^{-1}$.

Form the random variable $Lx = \varepsilon$.

Then $\varepsilon$ is an orthonormal basis for $x$, since $L$ is nonsingular, and $\mathbb{E} \, \varepsilon \, \varepsilon^\prime =
L \mathbb{E} xx^\prime L^\prime = I$.

It is convenient to write out the equations $Lx = \varepsilon$ and $L^{-1} \varepsilon = x$

```{math}
:label: eq_53

\begin{aligned}
    L_{11} x_1 &= \varepsilon_1 \\
    L_{21}x_1 + L_{22} x_2 &= \varepsilon_2 \\ \, \vdots \\
    L_{T1} \, x_1 \, \ldots \, + L_{TTx_T} &= \varepsilon_T
\end{aligned}
```

or

```{math}
:label: eq_54

\sum^{t-1}_{j=0} L_{t,t-j}\, x_{t-j} = \varepsilon_t, \quad t = 1, \, 2, \ldots T
```

We also have

```{math}
:label: eq_55

x_t = \sum^{t-1}_{j=0} L^{-1}_{t,t-j}\, \varepsilon_{t-j}\ .
```

Notice from {eq}`eq_55` that $x_t$ is in the space spanned by
$\varepsilon_t, \, \varepsilon_{t-1}, \ldots, \varepsilon_1$, and from {eq}`eq_54`   that
$\varepsilon_t$ is in the space spanned by $x_t,\, x_{t-1}, \ldots,\, x_1$.

Therefore, we have that for $t-1\geq m \geq 1$

```{math}
:label: eq_56

\mathbb{\hat E}
[ x_t \mid x_{t-m},\, x_{t-m-1}, \ldots, x_1 ] =
\mathbb{\hat E}
[x_t \mid \varepsilon_{t-m}, \varepsilon_{t-m-1},\ldots, \varepsilon_1]
```

For $t-1 \geq m \geq 1$ rewrite {eq}`eq_55`  as

```{math}
:label: eq_57

x_t = \sum^{m-1}_{j=0} L_{t,t-j}^{-1}\, \varepsilon_{t-j} + \sum^{t-1}_{j=m}
L^{-1}_{t, t-j}\, \varepsilon_{t-j}
```

Representation {eq}`eq_57`  is an orthogonal decomposition of $x_t$ into a part $\sum^{t-1}_{j=m} L_{t, t-j}^{-1}\, \varepsilon_{t-j}$ that lies in the space spanned by
$[x_{t-m},\, x_{t-m+1},\, \ldots, x_1]$, and an orthogonal
component not in this space.

### Implementation

Code that computes solutions to  LQ control and filtering problems  using the methods described here and in {doc}`Classical Control with Linear Algebra <lu_tricks>` can be found in the file [control_and_filter.jl](https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/lu_tricks/control_and_filter.jl).

Here's how it looks

```{code-cell} julia
---
tags: [remove-cell]
---
using Test
```

```{code-cell} julia
using LinearAlgebra, Statistics
using Polynomials.PolyCompat, LinearAlgebra
import Polynomials.PolyCompat: roots, coeffs

function LQFilter(d, h, y_m;
                  r = nothing,
                  beta = nothing,
                  h_eps = nothing)
    m = length(d) - 1
    m == length(y_m) ||
        throw(ArgumentError("y_m and d must be of same length = $m"))

    # define the coefficients of phi up front
    phi = zeros(2m + 1)
    for i in (-m):m
        phi[m - i + 1] = sum(diag(d * d', -i))
    end
    phi[m + 1] = phi[m + 1] + h

    # if r is given calculate the vector phi_r
    if isnothing(r)
        k = nothing
        phi_r = nothing
    else
        k = size(r, 1) - 1
        phi_r = zeros(2k + 1)

        for i in (-k):k
            phi_r[k - i + 1] = sum(diag(r * r', -i))
        end

        if isnothing(h_eps) == false
            phi_r[k + 1] = phi_r[k + 1] + h_eps
        end
    end

    # if beta is given, define the transformed variables
    if isnothing(beta)
        beta = 1.0
    else
        d = beta .^ (collect(0:m) / 2) * d
        y_m = y_m * beta .^ (-collect(1:m) / 2)
    end

    return (; d, h, y_m, m, phi, beta, phi_r, k)
end

function construct_W_and_Wm(lqf, N)
    (; d, m) = lqf

    W = zeros(N + 1, N + 1)
    W_m = zeros(N + 1, m)

    # terminal conditions
    D_m1 = zeros(m + 1, m + 1)
    M = zeros(m + 1, m)

    # (1) constuct the D_{m+1} matrix using the formula

    for j in 1:(m + 1)
        for k in j:(m + 1)
            D_m1[j, k] = dot(d[1:j, 1], d[(k - j + 1):k, 1])
        end
    end

    # Make the matrix symmetric
    D_m1 = D_m1 + D_m1' - Diagonal(diag(D_m1))

    # (2) Construct the M matrix using the entries of D_m1

    for j in 1:m
        for i in (j + 1):(m + 1)
            M[i, j] = D_m1[i - j, m + 1]
        end
    end
    M

    # Euler equations for t = 0, 1, ..., N-(m+1)
    phi, h = lqf.phi, lqf.h

    W[1:(m + 1), 1:(m + 1)] = D_m1 + h * I
    W[1:(m + 1), (m + 2):(2m + 1)] = M

    for (i, row) in enumerate((m + 2):(N + 1 - m))
        W[row, (i + 1):(2m + 1 + i)] = phi'
    end

    for i in 1:m
        W[N - m + i + 1, (end - (2m + 1 - i) + 1):end] = phi[1:(end - i)]
    end

    for i in 1:m
        W_m[N - i + 2, 1:((m - i) + 1)] = phi[(m + 1 + i):end]
    end

    return W, W_m
end

function roots_of_characteristic(lqf)
    (; m, phi) = lqf

    # Calculate the roots of the 2m-polynomial
    phi_poly = Poly(phi[end:-1:1])
    proots = roots(phi_poly)
    # sort the roots according to their length (in descending order)
    roots_sorted = sort(proots, by = abs)[end:-1:1]
    z_0 = sum(phi) / polyval(poly(proots), 1.0)
    z_1_to_m = roots_sorted[1:m]     # we need only those outside the unit circle
    lambda = 1 ./ z_1_to_m
    return z_1_to_m, z_0, lambda
end

function coeffs_of_c(lqf)
    (; m) = lqf
    z_1_to_m, z_0, lambda = roots_of_characteristic(lqf)
    c_0 = (z_0 * prod(z_1_to_m) * (-1.0)^m)^(0.5)
    c_coeffs = coeffs(poly(z_1_to_m)) * z_0 / c_0
    return c_coeffs
end

function solution(lqf)
    z_1_to_m, z_0, lambda = roots_of_characteristic(lqf)
    c_0 = coeffs_of_c(lqf)[end]
    A = zeros(m)
    for j in 1:m
        denom = 1 - lambda / lambda[j]
        A[j] = c_0^(-2) / prod(denom[1:m .!= j])
    end
    return lambda, A
end

function construct_V(lqf; N = nothing)
    if isnothing(N)
        error("N must be provided!!")
    end
    if !(N isa Integer)
        throw(ArgumentError("N must be Integer!"))
    end

    (; phi_r, k) = lqf
    V = zeros(N, N)
    for i in 1:N
        for j in 1:N
            if abs(i - j) <= k
                V[i, j] = phi_r[k + abs(i - j) + 1]
            end
        end
    end
    return V
end

function simulate_a(lqf, N)
    V = construct_V(N + 1)
    d = MVNSampler(zeros(N + 1), V)
    return rand(d)
end

function predict(lqf, a_hist, t)
    N = length(a_hist) - 1
    V = construct_V(N + 1)

    aux_matrix = zeros(N + 1, N + 1)
    aux_matrix[1:(t + 1), 1:(t + 1)] = Matrix(I, t + 1, t + 1)
    L = cholesky(V).U'
    Ea_hist = inv(L) * aux_matrix * L * a_hist

    return Ea_hist
end

function optimal_y(lqf, a_hist, t = nothing)
    (; beta, y_m, m) = lqf

    N = length(a_hist) - 1
    W, W_m = construct_W_and_Wm(lqf, N)

    F = lu(W, Val(true))

    L, U = F
    D = diagm(0 => 1.0 ./ diag(U))
    U = D * U
    L = L * diagm(0 => 1.0 ./ diag(D))

    J = reverse(Matrix(I, N + 1, N + 1), dims = 2)

    if isnothing(t)                      # if the problem is deterministic
        a_hist = J * a_hist

        # transform the a sequence if beta is given
        if beta != 1
            a_hist = reshape(a_hist * (beta^(collect(N:0) / 2)), N + 1, 1)
        end

        a_bar = a_hist - W_m * y_m        # a_bar from the lecutre
        Uy = \(L, a_bar)                  # U @ y_bar = L^{-1}a_bar from the lecture
        y_bar = \(U, Uy)                  # y_bar = U^{-1}L^{-1}a_bar
        # Reverse the order of y_bar with the matrix J
        J = reverse(Matrix(I, N + m + 1, N + m + 1), dims = 2)
        y_hist = J * vcat(y_bar, y_m)     # y_hist : concatenated y_m and y_bar
        # transform the optimal sequence back if beta is given
        if beta != 1
            y_hist = y_hist .* beta .^ (-collect((-m):N) / 2)
        end

    else                                  # if the problem is stochastic and we look at it
        Ea_hist = reshape(predict(a_hist, t), N + 1, 1)
        Ea_hist = J * Ea_hist

        a_bar = Ea_hist - W_m * y_m       # a_bar from the lecutre
        Uy = \(L, a_bar)                  # U @ y_bar = L^{-1}a_bar from the lecture
        y_bar = \(U, Uy)                  # y_bar = U^{-1}L^{-1}a_bar

        # Reverse the order of y_bar with the matrix J
        J = reverse(Matrix(I, N + m + 1, N + m + 1), dims = 2)
        y_hist = J * vcat(y_bar, y_m)     # y_hist : concatenated y_m and y_bar
    end
    return y_hist, L, U, y_bar
end
```

Let's use this code to tackle two interesting examples.

### Example 1

Consider a stochastic process with moving average representation

$$
x_t = (1 - 2 L) \varepsilon_t
$$

where $\varepsilon_t$ is a serially uncorrelated random process with mean zero and variance unity.

We want to use the Wiener-Kolmogorov formula {eq}`eq_36` to compute the linear least squares forecasts $\mathbb{E} [x_{t+j} \mid x_t, x_{t-1}, \ldots]$, for $j = 1,\, 2$.

We can do everything we want by setting $d = r$, generating an instance of LQFilter, then invoking pertinent methods of LQFilter

```{code-cell} julia
m = 1
y_m = zeros(m)
d = [1.0, -2.0]
r = [1.0, -2.0]
h = 0.0
example = LQFilter(d, h, y_m, r = d)
```

The Wold representation is computed by example.coefficients_of_c().

Let's check that it "flips roots" as required

```{code-cell} julia
coeffs_of_c(example)
```

```{code-cell} julia
roots_of_characteristic(example)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
  @test coeffs_of_c(example) ≈ [2.0, -1.0]
  @test roots_of_characteristic(example) == ([2.0], -2.0, [0.5])
end
```

Now let's form the covariance matrix of a time series vector of length $N$
and put it in $V$.

Then we'll take a Cholesky decomposition of $V = L^{-1} L^{-1} = Li Li'$ and use it to form the vector of "moving average representations" $x = Li \varepsilon$ and the vector of "autoregressive representations" $L x = \varepsilon$

```{code-cell} julia
V = construct_V(example, N = 5)
```

Notice how the lower rows of the "moving average representations" are converging to the appropriate infinite history Wold representation

```{code-cell} julia
F = cholesky(V)
Li = F.L
```

Notice how the lower rows of the "autoregressive representations" are converging to the appropriate infinite history autoregressive representation

```{code-cell} julia
L = inv(Li)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
  @test L[2, 1] ≈ 0.1951800145897066
  @test L[3, 3] ≈ 0.4970501217477084
end
```

**Remark** Let $\pi (z) = \sum^m_{j=0} \pi_j z^j$ and let $z_1, \ldots,
z_k$ be the zeros of $\pi (z)$ that are inside the unit circle, $k < m$.

Then define

$$
\theta (z) = \pi (z) \Biggl( {(z_1 z-1) \over (z-z_1)} \Biggr)
\Biggl( { (z_2 z-1) \over (z-z_2) } \Biggr ) \ldots \Biggl({(z_kz-1) \over
(z-z_k) }\Biggr)
$$

The term multiplying $\pi (z)$ is termed a "Blaschke factor".

Then it can be proved directly that

$$
\theta (z^{-1}) \theta (z) = \pi (z^{-1}) \pi (z)
$$

and that the zeros of $\theta (z)$ are not inside the unit circle.

### Example 2

Consider a stochastic process $X_t$ with moving average
representation

$$
X_t = (1 - \sqrt 2 L^2) \varepsilon_t
$$

where $\varepsilon_t$ is a serially uncorrelated random process
with mean zero and variance unity.

Let's find a Wold moving average representation for $x_t$.

Let's use the Wiener-Kolomogorov formula {eq}`eq_36` to compute the linear least squares forecasts
$\mathbb{\hat E}\left[X_{t+j} \mid X_{t-1}, \ldots\right] \hbox { for } j = 1,\, 2,\, 3$.

We proceed in the same way as example 1

```{code-cell} julia
m = 2
y_m = [0.0, 0.0]
d = [1, 0, -sqrt(2)]
r = [1, 0, -sqrt(2)]
h = 0.0
example = LQFilter(d, h, y_m, r = d)
```

```{code-cell} julia
coeffs_of_c(example)
```

```{code-cell} julia
roots_of_characteristic(example)
```

```{code-cell} julia
V = construct_V(example, N = 8)
```

```{code-cell} julia
F = cholesky(V)
Li = F.L
Li[(end - 2):end, :]
```

```{code-cell} julia
L = inv(Li)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
  @test L[3, 1] ≈ 0.30860669992418377
  @test L[2, 2] ≈ 0.5773502691896257
end
```

### Prediction

It immediately follows from the "orthogonality principle" of least squares (see {cite}`Athanasios1991` or {cite}`Sargent1987` [ch. X]) that

```{math}
:label: eq_58

\begin{aligned}
    \mathbb{\hat E} & [x_t \mid x_{t-m},\, x_{t-m+1}, \ldots x_1]
                    = \sum^{t-1}_{j=m} L^{-1}_{t,t-j}\, \varepsilon_{t-j} \\
               & = [L_{t, 1}^{-1}\, L^{-1}_{t,2},\, \ldots, L^{-1}_{t,t-m}\ 0 \ 0 \ldots 0] L \, x
\end{aligned}
```

This can be interpreted as a finite-dimensional version of the Wiener-Kolmogorov $m$-step ahead prediction formula.

We can use {eq}`eq_58`  to represent the linear least squares projection of
the vector $x$ conditioned on the first $s$ observations
$[x_s, x_{s-1} \ldots, x_1]$.

We have

```{math}
:label: eq_59

\mathbb{\hat E}[x \mid x_s, x_{s-1}, \ldots, x_1]
= L^{-1}
\left[
    \begin{matrix}
        I_s & 0 \\
        0 & 0_{(t-s)}
    \end{matrix}
\right] L x
```

This formula will be convenient in representing the solution of control problems under uncertainty.

Equation {eq}`eq_55`  can be recognized as a finite dimensional version of a moving average representation.

Equation  {eq}`eq_54` can be viewed as a finite dimension version of an autoregressive representation.

Notice that even
if the $x_t$ process is covariance stationary, so that $V$
is such that $V_{ij}$ depends only on $\vert i-j\vert$, the
coefficients in the moving average representation are time-dependent,
there being a different moving average for each $t$.

If
$x_t$ is a covariance stationary process, the last row of
$L^{-1}$ converges to the coefficients in the Wold moving average
representation for $\{ x_t\}$ as $T \rightarrow \infty$.

Further, if $x_t$ is covariance stationary, for fixed $k$
and $j > 0, \, L^{-1}_{T,T-j}$ converges to
$L^{-1}_{T-k, T-k-j}$ as $T \rightarrow \infty$.

That is,
the “bottom” rows of $L^{-1}$ converge to each other and to the
Wold moving average coefficients as $T \rightarrow \infty$.

This last observation gives one simple and widely-used practical way of
forming a finite $T$ approximation to a Wold moving average
representation.

First, form the covariance matrix
$\mathbb{E}xx^\prime = V$, then obtain the Cholesky decomposition
$L^{-1} L^{-1^\prime}$ of $V$, which can be accomplished
quickly on a computer.

The last row of $L^{-1}$ gives the approximate Wold moving average coefficients.

This method can readily be generalized to multivariate systems.

(fdcp)=
## Combined Finite Dimensional Control and Prediction

Consider the finite-dimensional control problem, maximize

$$
\mathbb{E} \, \sum^N_{t=0} \,
\left\{
     a_t y_t - {1 \over 2} h y^2_t - {1 \over 2} [d(L) y_t ]^2
\right\},\  \quad h > 0
$$

where $d(L) = d_0 + d_1 L+ \ldots + d_m L^m$, $L$ is the
lag operator, $\bar a = [ a_N, a_{N-1} \ldots, a_1, a_0]^\prime$ a
random vector with mean zero and $\mathbb{E}\,\bar a \bar a^\prime = V$.

The variables $y_{-1}, \ldots, y_{-m}$ are given.

Maximization is over choices of $y_0,
y_1 \ldots, y_N$, where $y_t$ is required to be a linear function
of $\{y_{t-s-1}, t+m-1\geq 0;\ a_{t-s}, t\geq s\geq 0\}$.

We saw in the lecture {doc}`Classical Control with Linear Algebra <../time_series_models/lu_tricks>`  that the solution of this problem under certainty could be represented in feedback-feedforward form

$$
U \bar y
   = L^{-1}\bar a + K
   \left[
     \begin{matrix}
         y_{-1}\\
         \vdots\\
         y_{-m}
     \end{matrix}
   \right]
$$

for some $(N+1)\times m$ matrix $K$.

Using a version of formula {eq}`eq_58`, we can express $\mathbb{\hat E}[\bar a \mid a_s,\, a_{s-1}, \ldots, a_0 ]$ as

$$
\mathbb{\hat E}
[ \bar a \mid a_s,\, a_{s-1}, \ldots, a_0]
= \tilde U^{-1}
\left[
    \begin{matrix}
        0 & 0 \\
        0 & I_{(s+1)}
    \end{matrix}
\right]
\tilde U \bar a
$$

where $I_{(s + 1)}$ is the $(s+1) \times (s+1)$ identity
matrix, and $V = \tilde U^{-1} \tilde U^{-1^{\prime}}$, where
$\tilde U$ is the *upper* trangular Cholesky factor of the
covariance matrix $V$.

(We have reversed the time axis in dating the $a$'s relative to earlier)

The time axis can be reversed in representation {eq}`eq_59` by replacing $L$ with $L^T$.

The optimal decision rule to use at time $0 \leq t \leq N$ is then
given by the $(N-t +1)^{\rm th}$ row of

$$
U \bar y = L^{-1} \tilde U^{-1}
    \left[
        \begin{matrix}
            0 & 0 \\
            0 & I_{(t+1)}
        \end{matrix}
    \right]
    \tilde U \bar a + K
    \left[
    \begin{matrix}
        y_{-1}\\
        \vdots\\
        y_{-m}
    \end{matrix}
    \right]
$$

## Exercises

### Exercise 1

Let $Y_t = (1 - 2 L ) u_t$ where $u_t$ is a mean zero
white noise with $\mathbb{E} u^2_t = 1$. Let

$$
X_t = Y_t + \varepsilon_t
$$

where $\varepsilon_t$ is a serially uncorrelated white noise with
$\mathbb{E} \varepsilon^2_t = 9$, and $\mathbb{E} \varepsilon_t u_s = 0$ for all
$t$ and $s$.

Find the Wold moving average representation for $X_t$.

Find a formula for the $A_{1j}$'s in

$$
\mathbb{E} \widehat X_{t+1} \mid X_t, X_{t-1}, \ldots = \sum^\infty_{j=0} A_{1j}
X_{t-j}
$$

Find a formula for the $A_{2j}$'s in

$$
\mathbb{\hat E} X_{t+2} \mid X_t, X_{t-1}, \ldots = \sum^\infty_{j=0} A_{2j}
X_{t-j}
$$

### Exercise 2

(Multivariable Prediction) Let $Y_t$ be an $(n\times 1)$
vector stochastic process with moving average representation

$$
Y_t = D(L) U_t
$$

where $D(L) = \sum^m_{j=0} D_j L^J, D_j$ an $n \times n$
matrix, $U_t$ an $(n \times 1)$ vector white noise with
:math: mathbb{E} U_t =0 for all $t$, $\mathbb{E} U_t U_s' = 0$ for all $s \neq t$,
and $\mathbb{E} U_t U_t' = I$ for all $t$.

Let $\varepsilon_t$ be an $n \times 1$ vector white noise with mean $0$ and contemporaneous covariance matrix $H$, where $H$ is a positive definite matrix.

Let $X_t = Y_t +\varepsilon_t$.

Define the covariograms as $C_X
(\tau) = \mathbb{E} X_t X^\prime_{t-\tau}, C_Y (\tau) = \mathbb{E} Y_t Y^\prime_{t-\tau},
C_{YX} (\tau) = \mathbb{E} Y_t X^\prime_{t-\tau}$.

Then define the matrix
covariance generating function, as in {eq}`onetwenty`, only interpret all the
objects in {eq}`onetwenty` as matrices.

Show that the covariance generating functions are given by

$$
\begin{aligned}
    g_y (z) &= D (z) D (z^{-1})^\prime \\
    g_X (z) &= D (z) D (z^{-1})^\prime + H \\
    g_{YX} (z) &= D (z) D (z^{-1})^\prime
\end{aligned}
$$

A factorization of $g_X (z)$ can be found (see {cite}`Rozanov1967` or {cite}`Whittle1983`) of the form

$$
D (z) D (z^{-1})^\prime + H = C (z) C (z^{-1})^\prime, \quad C (z) =
\sum^m_{j=0} C_j z^j
$$

where the zeros of $\vert C(z)\vert$ do not lie inside the unit
circle.

A vector Wold moving average representation of $X_t$ is then

$$
X_t = C(L) \eta_t
$$

where $\eta_t$ is an $(n\times 1)$ vector white noise that
is "fundamental" for $X_t$.

That is, $X_t - \mathbb{\hat E}\left[X_t \mid X_{t-1}, X_{t-2}
\ldots\right] = C_0 \, \eta_t$.

The optimum predictor of $X_{t+j}$ is

$$
\mathbb{\hat E} \left[X_{t+j} \mid X_t, X_{t-1}, \ldots\right]
 = \left[{C(L) \over L^j} \right]_+ \eta_t
$$

If $C(L)$ is invertible, i.e., if the zeros of $\det$
$C(z)$ lie strictly outside the unit circle, then this formula can
be written

$$
\mathbb{\hat E} \left[X_{t+j} \mid X_t, X_{t-1}, \ldots\right]
    = \left[{C(L) \over L^J} \right]_+ C(L)^{-1}\, X_t
$$

