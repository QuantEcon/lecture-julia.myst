---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia
---

(quadrature_interpolation)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Quadrature and Interpolation

```{contents} Contents
:depth: 2
```

## Overview

In this section we will explore the related concepts of quadrature, interpolation, and discretization of continuous functions and distributions.


```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Statistics, Distributions
using QuadGK, FastGaussQuadrature, SpecialFunctions
using Interpolations, Plots
using QuantEcon
```

## Numerical Integration

Many applications require directly calculating a numerical derivative and calculating expectations.

### General Packages

Julia's [Integrals.jl](https://github.com/SciML/Integrals.jl) provides a unified interface to many quadrature backends and is differentiable with common AD systems (e.g. Zygote, ForwardDiff, Enzyme), which can be useful when an integral appears inside an optimization or learning problem.

### Adaptive Quadrature

A high accuracy solution for calculating numerical integrals is [QuadGK](https://github.com/JuliaMath/QuadGK.jl).

```{code-cell} julia
using QuadGK
@show value, tol = quadgk(cos, -2π, 2π);
```

This is an adaptive Gauss-Kronrod integration technique that's relatively accurate for smooth functions.

However, its adaptive implementation makes it slow and not well suited to inner loops.

### Gauss Legendre

Alternatively, many integrals can be done efficiently with (non-adaptive) [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature).

For example, using [FastGaussQuadrature.jl](https://github.com/ajt60gaibb/FastGaussQuadrature.jl)

```{code-cell} julia
x, w = FastGaussQuadrature.gausslegendre(100_000); # i.e. find 100,000 nodes

# integrates f(x) = x^2 from -1 to 1
f(x) = x^2
@show dot(w, f.(x)); # calculate integral
```

With the `FastGaussQuadrature` package you may need to deal with affine transformations to the non-default domains yourself.


### Gauss Legendre

Another commonly used quadrature well suited to random variables with bounded support is [Gauss–Jacobi quadrature](https://en.wikipedia.org/wiki/Gauss–Jacobi_quadrature).

It provides nodes $s_n\in[-1,1]$ and weights $\omega_n$ for

$$
\int_{-1}^1 g(s)\,(1-s)^{a}(1+s)^{b}\,ds \;\approx\; \sum_{n=1}^N \omega_n\, g(s_n).
$$

For $X\sim\mathrm{Beta}(\alpha,\beta)$,

$$
\mathbb{E}[f(X)] = \int_0^1 f(x)\,\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}\,dx,
$$

with the change of variables $s=2x-1$ (so $x=(s+1)/2$, $dx=ds/2$). This yields Gauss–Jacobi exponents $a=\beta-1$, $b=\alpha-1$ and a factor $C=2^{-(\alpha+\beta-1)}/B(\alpha,\beta)$:

$$
\mathbb{E}[f(X)] \;\approx\; C\sum_{n=1}^N \omega_n\, f\!\left(\tfrac{s_n+1}{2}\right).
$$

```{code-cell} julia
function gauss_jacobi(F::Beta, N)
    s, wj = FastGaussQuadrature.gaussjacobi(N, F.β - 1, F.α - 1)
    x = (s .+ 1) ./ 2
    C = 2.0^(-(F.α + F.β - 1.0)) / SpecialFunctions.beta(F.α, F.β)
    w = C .* wj
    return x, w
end

N = 32
F = Beta(2, 2)
F2 = Beta(0.5, 1.2)
x, w = gauss_jacobi(F, N)
x2, w2 = gauss_jacobi(F2, N)
f(x) = x^2
@show dot(w, f.(x)), mean(f.(rand(F, 1000)))
@show dot(w2, f.(x2)), mean(f.(rand(F2, 1000)));
```

## Gauss Hermite

Many expectations are of the form $\mathbb{E}\left[f(X)\right]\approx \sum_{n=1}^N w_n f(x_n)$ where $X \sim N(0,1)$.  Alternatively, integrals of the form $\int_{-\infty}^{\infty}f(x) exp(-x^2) dx$.

Gauss-hermite quadrature provides weights and nodes of this form, where the `normalize = true` argument provides the appropriate rescaling for the normal distribution.

Through a change of variables this can be used to calculate expectations of $N(\mu,\sigma^2)$ variables, through

```{code-cell} julia
function gauss_hermite_normal(N::Integer, mu::Real, sigma::Real)
    s, w = FastGaussQuadrature.gausshermite(N; normalize = true)

    # X ~ N(mu, sigma^2), X \sim mu + sigma N(0,1) we transform the standard-normal nodes
    x = mu .+ sigma .* s
    return x, w
end

N = 32
x, w = gauss_hermite_normal(N, 1.0, 0.1)
x2, w2 = gauss_hermite_normal(N, 0.0, 0.05)
f(x) = x^2
@show dot(w, f.(x)), mean(f.(rand(Normal(1.0, 0.1), 1000)))
@show dot(w2, f.(x2)), mean(f.(rand(Normal(0.0, 0.05), 1000)));
```

## Gauss Laguerre

Another quadrature scheme appropriate integrals defined on $[0,\infty]$ is Gauss Laguerre, which approximates integrals of the form $\int_0^{\infty} f(x) x^{\alpha} \exp(-x) dx \approx \sum_{n=1}^N w_n f(x_n)$.

One application is to calculate expectations of exponential variables.  The PDF of an exponential distribution with parameter $\theta$ is $f(x;\theta) = \frac{1}{\theta}\exp(-x/\theta)$.  With a change of variables we can use Gauss Laguerre quadrature

```{code-cell} julia
function gauss_laguerre_exponential(N, theta)
    #   E[f(X)] = \int_0^\infty f(x) (1/theta) e^{-x/theta} dx = \int_0^\infty f(theta*y) e^{-y} dy.
    s, w = FastGaussQuadrature.gausslaguerre(N)  # alpha = 0 (default)
    x = theta .* s
    return x, w
end

N = 64
theta = 0.5
x, w = gauss_laguerre_exponential(N, theta)
f(x) = x^2 + 1
@show dot(w, f.(x)), mean(f.(rand(Exponential(theta), 1_000)))
```

Similarly, the Gamma distribution with shape parameter $\alpha$ and scale $\theta$ has PDF $f(x; \alpha, \theta) = \frac{x^{\alpha-1} e^{-x/\theta}}{\Gamma(\alpha) \theta^\alpha}$ for $x > 0$ with $\Gamma(\cdot)$ the Gamma special function.

Using a change of variable and Gauss Laguerre quadrature

```{code-cell} julia
function gauss_laguerre_gamma(N, alpha, theta)
    # For Gamma(shape=alpha, scale=theta) with pdf
    #   x^{alpha-1} e^{-x/theta} / (Gamma(alpha) theta^alpha)
    # change variable y = x/theta -> x = theta*y, dx = theta dy
    # E[f(X)] = 1/Gamma(alpha) * ∫_0^∞ f(theta*y) y^{alpha-1} e^{-y} dy
    # FastGaussQuadrature.gausslaguerre(N, a) returns nodes/weights for
    # ∫_0^∞ g(y) y^a e^{-y} dy, so pass a = alpha - 1.

    s, w = FastGaussQuadrature.gausslaguerre(N, alpha - 1)
    x = theta .* s
    w = w ./ SpecialFunctions.gamma(alpha)
    return x, w
end

N = 256
alpha = 7.0
theta = 1.1
x, w = gauss_laguerre_gamma(N, alpha, theta)
f(x) = x^2 + 1
@show dot(w, f.(x)), mean(f.(rand(Gamma(alpha, theta), 100_000)))
```



## Interpolation

In economics we often wish to interpolate discrete data (i.e., build continuous functions that join discrete sequences of points).

The package we usually turn to for this purpose is [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl).

There are a variety of options, but we will only demonstrate the convenient notations.

### Univariate with a Regular Grid

Let's start with the univariate case.

We begin by creating some data points, using a sine function

```{code-cell} julia
using Interpolations
using Plots

x = -7:7 # x points, coase grid
y = sin.(x) # corresponding y points

xf = -7:0.1:7        # fine grid
plot(xf, sin.(xf), label = "sin function")
scatter!(x, y, label = "sampled data", markersize = 4)
```

To implement linear and cubic [spline](https://en.wikipedia.org/wiki/Spline_%28mathematics%29) interpolation

```{code-cell} julia
li = LinearInterpolation(x, y)
li_spline = CubicSplineInterpolation(x, y)

@show li(0.3) # evaluate at a single point

scatter(x, y, label = "sampled data", markersize = 4)
plot!(xf, li.(xf), label = "linear")
plot!(xf, li_spline.(xf), label = "spline")
```

### Univariate with Irregular Grid

In the above, the `LinearInterpolation` function uses a specialized function
for regular grids since `x` is a `Range` type.

For an arbitrary, irregular grid

```{code-cell} julia
x = log.(range(1, exp(4), length = 10)) .+ 1  # uneven grid
y = log.(x) # corresponding y points

interp = LinearInterpolation(x, y)

xf = log.(range(1, exp(4), length = 100)) .+ 1 # finer grid

plot(xf, interp.(xf), label = "linear")
scatter!(x, y, label = "sampled data", markersize = 4, size = (800, 400))
```

At this point, `Interpolations.jl` does not have support for cubic splines with irregular grids, but there are plenty of other packages that do (e.g. [Dierckx.jl](https://github.com/kbarbary/Dierckx.jl)  and [GridInterpolations.jl](https://github.com/sisl/GridInterpolations.jl)).

### Multivariate Interpolation

Interpolating a regular multivariate function uses the same function

```{code-cell} julia
f(x, y) = log(x + y)
xs = 1:0.2:5
ys = 2:0.1:5
A = [f(x, y) for x in xs, y in ys]

# linear interpolation
interp_linear = LinearInterpolation((xs, ys), A)
@show interp_linear(3, 2) # exactly log(3 + 2)
@show interp_linear(3.1, 2.1) # approximately log(3.1 + 2.1)

# cubic spline interpolation
interp_cubic = CubicSplineInterpolation((xs, ys), A)
@show interp_cubic(3, 2) # exactly log(3 + 2)
@show interp_cubic(3.1, 2.1) # approximately log(3.1 + 2.1);
```

See [Interpolations.jl documentation](https://github.com/JuliaMath/Interpolations.jl#convenience-notation) for more details on options and settings.


## Discretization of Stochastic Processes

In many cases a stochastic process is defined on a continuous space, but numerical algorithms are ideal if the space is discretized.

For a Markov process in discrete time, this leads to a Markov chain.  See {doc}`Finite Markov Chains <../introduction_dynamics/finite_markov>` for more details.
### Discretizing an AR(1) Process

Markov chains are routinely generated as discrete approximations to {doc}`AR(1) processes <../introduction_dynamics/ar1_processes>` of the form

$$
X_{t+1} = \rho X_t + \mu + \sigma W_{t+1}
$$

Here $W_{t+1}$ is assumed to be i.i.d. and $N(0, 1)$.

When $|\rho| < 1$, the process is stationary with unconditional mean $\mu_X$ and standard deviation $\sigma_X$:

$$
\mu_X = \frac{\mu}{1-\rho}, \quad \sigma_X = \frac{\sigma}{\sqrt{1 - \rho^2}}
$$

The goal of a discretization is to find a set of discrete states, $\{x_1, \ldots, x_N\}$, and a stochastic transition matrix $P$ which best approximates the continuous dynamics.

Typically, these approximations adapt the grid of states to the variance of the stationary distribution to ensure sufficient coverage. This is usually parameterized by choosing $m > 0$ standard deviations on both sides of the stationary distribution to cover.

### Tauchen's Method (1986)

Tauchen's method {cite}`Tauchen1986` is the most common method for approximating this process with a finite state Markov chain.

Standard implementations often simplify the calculation by discretizing the zero-mean process $Z_t = X_t - \mu_X$ first, and then shifting the resulting grid.  This is equivalent due to the linearity of the AR(1) process.

The zero-mean dynamics are:

$$
Z_{t+1} = \rho Z_t + \sigma W_{t+1}
$$

The grid for $Z$, denoted $z := \{z_1, \ldots, z_N\}$, is constructed to cover $[-m \sigma_X, + m \sigma_X]$. The points are evenly spaced with step size $d$:

$$
d = \frac{z_N - z_1}{N-1}
$$

We associate each state $z_j$ with the interval $[z_j - d/2, z_j + d/2]$. To construct the transition matrix $P$, we calculate the probability of moving from $z_n$ to the interval associated with $z_j$.

Let $F$ be the CDF of the standard normal distribution.

Since $Z_{t+1} | Z_t = z_n \sim \mathcal{N}(\rho z_n, \sigma^2)$, we standardize with $\sigma$.

1. For interior points $j = 2, \ldots, N-1$:

$$
P_{nj} \approx \mathbb{P}(z_j - d/2 \leq Z_{t+1} \leq z_j + d/2\,|\,Z_t = z_n) = F\left(\frac{z_j + d/2 - \rho z_n}{\sigma}\right) - F\left(\frac{z_j - d/2 - \rho z_n}{\sigma}\right)
$$

2. For the left boundary $j = 1$:

$$
P_{n1} \approx \mathbb{P}(Z_{t+1} \leq z_1 + d/2\,|\,Z_t = z_n) = F\left(\frac{z_1 + d/2 - \rho z_n}{\sigma}\right)
$$

3. For the right boundary $j = N$:

$$
P_{nN} \approx \mathbb{P}(Z_{t+1} \geq z_N - d/2\,|\,Z_t = z_n) = 1 - F\left(\frac{z_N - d/2 - \rho z_n}{\sigma}\right)
$$

Finally, the state vector for the original process $X$ is recovered by shifting the grid: $x_n = z_n + \mu_X$.

The following code implements this procedure using Julia's vectorization syntax.

```{code-cell} julia
function tauchen(N, rho, sigma, mu, m = 3)
    @assert abs(rho) < 1
    mu_X = mu / (1 - rho)
    sigma_X = sigma / sqrt(1 - rho^2)

    # zero-centered grid and midpoints as cutoffs
    z = range(-m*sigma_X, m*sigma_X, length = N)
    d = step(z)
    midpoints = (z[1:(end - 1)] .+ d/2)'
    means = rho .* z
    Z_scores = (midpoints .- means) ./ sigma

    # Construct P: [Left Tail, Interval Diffs, Right Tail]
    F = cdf.(Normal(), Z_scores)
    P = [F[:, 1] diff(F, dims = 2) (1 .- F[:, end])]
    x = z .+ mu_X
    return (; P, mu_X, sigma_X, x)
end
```

As an example, consider the AR(1) process with parameters $\rho = 0.9$, $\mu = 0.2$, and $\sigma = 0.1$.

We can check that all rows of the transition matrix sum to one, and inspect the transition probabilities from a particular state.

```{code-cell} julia
rho = 0.9
mu = 0.2
sigma = 0.1
N = 5
(; P, x) = tauchen(N, rho, sigma, mu)
@show x
println("Row sums of P: ", sum(P, dims = 2))

state_index = 3
println("Transition probabilities from state x = $(x[state_index]):")
for j in 1:N
    println("  to x = $(x[j]): P = $(P[state_index, j])")
end
```

```{code-cell} julia
---
tags: [remove-cell]
---
using Test, QuantEcon
@testset "Tauchen vs QuantEcon" begin
    mc_qe = QuantEcon.tauchen(N, rho, sigma, mu)
    P_qe = mc_qe.p
    x_qe = mc_qe.state_values
    @test maximum(abs.(P - P_qe)) < 1e-10
    @test maximum(abs.(x - x_qe)) < 1e-10
    @test P[3, 3] ≈ 0.6578982972494045  # canary: central transition probability
end
```


Note that the majority of the mass is concentrated around the diagonal, reflecting the high persistence of the AR(1) process with $\rho = 0.9$.

We can visualize these sorts of transition matrices as a heatmap to get a better sense of the structure.

Below we add more states to make it closer to the continuous process and increase the persistence.

```{code-cell} julia
N = 100
rho = 0.98
mu = 0.1
(; P, x) = tauchen(N, rho, sigma, mu)

heatmap(x, x, P,
        xlabel = "To State", ylabel = "From State",
        title = "Transition Matrix Heatmap", colorbar_title = "Probability")
```

Note that the transition matrix is highly concentrated close to the diagonal.

The exception are those close to the first and last rows, which are at the boundaries of the truncated state space.

Below we see that the majority of the transition probabilities are close to zero.

```{code-cell} julia
threshold = 1E-6
num_nonzero = sum(P .> threshold)
println("Proportion of transitions > $threshold: ", num_nonzero / (N^2))
```
