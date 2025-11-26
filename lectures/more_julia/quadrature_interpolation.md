---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.12
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

Julia has both a large number of useful, well written libraries and many incomplete poorly maintained proofs of concept.

A major advantage of Julia libraries is that, because Julia itself is sufficiently fast, there is less need to mix in low level languages like C and Fortran.

As a result, most Julia libraries are written exclusively in Julia.

Not only does this make the libraries more portable, it makes them much easier to dive into, read, learn from and modify.

See {doc}`general, data, and statistical packages <../more_julia/data_statistical_packages>` and {doc}`optimization, solver, and related packages <../more_julia/optimization_solver_packages>` for more domain specific packages.

In this section we will explore the related concepts of Quadrature and Interpolation.


```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Statistics, Distributions
using QuadGK, FastGaussQuadrature, SpecialFunctions
using Interpolations, Plots
```

## Numerical Integration

Many applications require directly calculating a numerical derivative and calculating expectations.

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

xf = log.(range(1,exp(4), length = 100)) .+ 1 # finer grid

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