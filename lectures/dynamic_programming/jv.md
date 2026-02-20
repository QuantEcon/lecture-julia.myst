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

(jv)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`Job Search V: On-the-Job Search <single: Job Search V: On-the-Job Search>`

```{index} single: Models; On-the-Job Search
```

```{contents} Contents
:depth: 2
```

## Overview

### Model features

```{index} single: On-the-Job Search; Model Features
```

* job-specific human capital accumulation combined with on-the-job search
* infinite horizon dynamic programming with one state variable and two controls



```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Statistics
using Distributions, Interpolations
using FastGaussQuadrature, SpecialFunctions
using LaTeXStrings, Plots, NLsolve, Random

```

## Model

```{index} single: On-the-Job Search; Model
```

Let

* $x_t$ denote the time-$t$ job-specific human capital of a worker employed at a given firm
* $w_t$ denote current wages

Let $w_t = x_t(1 - s_t - \phi_t)$, where

* $\phi_t$ is investment in job-specific human capital for the current role
* $s_t$ is search effort, devoted to obtaining new offers from other firms

For as long as the worker remains in the current job, evolution of
$\{x_t\}$ is given by $x_{t+1} = G(x_t, \phi_t)$.

When search effort at $t$ is $s_t$, the worker receives a new job
offer with probability $\pi(s_t) \in [0, 1]$.

Value of offer is $U_{t+1}$, where $\{U_t\}$ is iid with common distribution $F$.

Worker has the right to reject the current offer and continue with existing job.

In particular, $x_{t+1} = U_{t+1}$ if accepts and $x_{t+1} = G(x_t, \phi_t)$ if rejects.

Letting $b_{t+1} \in \{0,1\}$ be binary with $b_{t+1} = 1$ indicating an offer, we can write

```{math}
:label: jd

x_{t+1}
= (1 - b_{t+1}) G(x_t, \phi_t) + b_{t+1}
    \max \{ G(x_t, \phi_t), U_{t+1}\}
```

Agent's objective: maximize expected discounted sum of wages via controls $\{s_t\}$ and $\{\phi_t\}$.

Taking the expectation of $V(x_{t+1})$ and using {eq}`jd`,
the Bellman equation for this problem can be written as

```{math}
:label: jvbell

V(x)
= \max_{s + \phi \leq 1}
    \left\{
        x (1 - s - \phi) + \beta (1 - \pi(s)) V[G(x, \phi)] +
        \beta \pi(s) \int V[G(x, \phi) \vee u] F(du)
     \right\}.
```

Here nonnegativity of $s$ and $\phi$ is understood, while
$a \vee b := \max\{a, b\}$.

### Parameterization

```{index} single: On-the-Job Search; Parameterization
```

In the implementation below, we will focus on the parameterization.

$$
G(x, \phi) = A (x \phi)^{\alpha},
\quad
\pi(s) = \sqrt s
\quad \text{and} \quad
F = \text{Beta}(2, 2)
$$

with default parameter values

* $A = 1.4$
* $\alpha = 0.6$
* $\beta = 0.96$

The Beta(2,2) distribution is supported on $(0,1)$.  It has a unimodal, symmetric density peaked at 0.5.

### Quadrature
In order to calculate expectations over the continuously valued $F$ distribution, we need to either draw values
and use Monte Carlo integration, or discretize.

[Gaussian Quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature) methods use orthogonal polynomials to generate $N$ nodes, $x$ and weights, $w$, to calculate integrals of the form $\int f(x) dx \approx \sum_{n=1}^N w_n f(x_n)$ for various bounded domains.

Here we will use [Gauss-Jacobi Quadrature](https://en.wikipedia.org/wiki/Gauss–Jacobi_quadrature) which is ideal for expectations over beta.

See {doc}`quadrature and interpolation <../more_julia/quadrature_interpolation>` for details on the derivation in this particular case.

```{code-cell} julia
function gauss_jacobi(F::Beta, N)
    s, wj = FastGaussQuadrature.gaussjacobi(N, F.β - 1, F.α - 1)
    x = (s .+ 1) ./ 2
    C = 2.0^(-(F.α + F.β - 1.0)) / SpecialFunctions.beta(F.α, F.β)
    w = C .* wj
    return x, w
end
f(x) = x^2
F = Beta(2, 2)
x, w = gauss_jacobi(F, 20)
# compare to monte-carlo integration
@show dot(w, f.(x)), mean(f.(rand(F, 1000)));
```



(jvboecalc)=
### Back-of-the-Envelope Calculations

Before we solve the model, let's make some quick calculations that
provide intuition on what the solution should look like.

To begin, observe that the worker has two instruments to build
capital and hence wages:

1. invest in capital specific to the current job via $\phi$
1. search for a new job with better job-specific capital match via $s$

Since wages are $x (1 - s - \phi)$, marginal cost of investment via either $\phi$ or $s$ is identical.

Our risk neutral worker should focus on whatever instrument has the highest expected return.

The relative expected return will depend on $x$.

For example, suppose first that $x = 0.05$

* If $s=1$ and $\phi = 0$, then since $G(x,\phi) = 0$,
  taking expectations of {eq}`jd` gives expected next period capital equal to $\pi(s) \mathbb{E} U
  = \mathbb{E} U = 0.5$.
* If $s=0$ and $\phi=1$, then next period capital is $G(x, \phi) = G(0.05, 1) \approx 0.23$.

Both rates of return are good, but the return from search is better.

Next suppose that $x = 0.4$

* If $s=1$ and $\phi = 0$, then expected next period capital is again $0.5$
* If $s=0$ and $\phi = 1$, then $G(x, \phi) = G(0.4, 1) \approx 0.8$

Return from investment via $\phi$ dominates expected return from search.

Combining these observations gives us two informal predictions:

1. At any given state $x$, the two controls $\phi$ and $s$ will function primarily as substitutes --- worker will focus on whichever instrument has the higher expected return.
1. For sufficiently small $x$, search will be preferable to investment in job-specific human capital.  For larger $x$, the reverse will be true.

Now let's turn to implementation, and see if we can match our predictions.

## Implementation

```{index} single: On-the-Job Search; Programming Implementation
```

The following code solves the DP problem described above

```{code-cell} julia
---
tags: [remove-cell]
---
using Test
```

```{code-cell} julia
function jv_worker(; A = 1.4,
                     alpha = 0.6,
                     beta = 0.96,
                     grid_size = 50,
                     quad_size = 30,
                     search_grid_size = 15,
                     epsilon = 1e-4)
    G(x, phi) = A * (x * phi)^alpha
    pi_func = sqrt
    F = Beta(2, 2)

    u, w = gauss_jacobi(F, quad_size)

    grid_max = max(A^(1.0 / (1.0 - alpha)), quantile(F, 1 - epsilon))
    x_grid = range(epsilon, grid_max, length = grid_size)
    search_grid = range(epsilon, 1.0, length = search_grid_size)

    # Pre-calculate the flat list of valid (s, phi) tuples which are feasible
    choices = vec([(s, phi) for s in search_grid, phi in search_grid if s + phi <= 1.0])

    return (; A, alpha, beta, x_grid, choices, G,
              pi_func, F, u, w, epsilon)
end

function T(jv, V)
    (; G, pi_func, beta, u, w, choices, x_grid) = jv
    Vf = LinearInterpolation(x_grid, V, extrapolation_bc = Line())

    # Objective takes a tuple 'c' which contains (s, phi)
    function objective(x, c)
        s, phi = c 
        g_val = G(x, phi)
        integral = sum(w[j] * Vf(max(g_val, u[j])) for j in eachindex(u))
        continuation = (1.0 - pi_func(s)) * Vf(g_val) + pi_func(s) * integral
        return x * (1.0 - s - phi) + beta * continuation
    end

    # Pre-allocate output arrays
    new_V = similar(x_grid)
    s_policy = similar(x_grid)
    phi_policy = similar(x_grid)

    # Loop over states
    for (i, x) in enumerate(x_grid)
        # Broadcast: evaluate 'objective' for this 'x' across all 'choices'
        vals = objective.(x, choices)
        
        # Find the best value and its index
        v_max, idx = findmax(vals)

        # Store results
        new_V[i] = v_max
        s_policy[i], phi_policy[i] = choices[idx] # Unpack the tuple
    end

    return new_V, (s_policy, phi_policy)
end
```

The code is written to be relatively generic---and hence reusable.

* For example, we use generic $G(x,\phi)$ instead of specific $A (x \phi)^{\alpha}$.

Function `jv_worker` packages all parameters for the model. The Bellman
operator $T$ acts on a candidate value function via {eq}`jvbell`. In
code, `T` returns a fresh value array together with policies for $s$ and
$\phi$ on the state grid. It builds a linear interpolant `Vf` on
`x_grid` and then evaluates

```{math}
w(s, \phi) =
 x (1 - s - \phi) + \beta (1 - \pi(s)) V[G(x, \phi)] +
 \beta \pi(s) \int V[G(x, \phi) \vee u] F(du)
```

on a coarse feasible grid, taking the maximizer over $s + \phi \leq 1$.
Expectations are computed with the quadrature nodes `u` and weights `w`.
The second return value collects the maximizing $s(x)$ and $\phi(x)$ at
each state.

(jv_solve)=
## Solving for Policies

```{index} single: On-the-Job Search; Solving for Policies
```

Let's plot the optimal policies and see what they look like.

The code is as follows

```{code-cell} julia
wp = jv_worker(; grid_size = 25)
v_init = collect(wp.x_grid) .* 0.5

V = fixedpoint(v -> T(wp, v)[1], v_init)
sol_V = V.zero

_, (s_policy, phi_policy) = T(wp, sol_V)

# plot solution
p = plot(wp.x_grid, [phi_policy s_policy sol_V],
         title = [L"$\phi$ policy" L"$s$ policy" "value function"],
         color = [:orange :blue :green],
         xaxis = (L"x", (0.0, maximum(wp.x_grid))),
         yaxis = ((-0.1, 1.1)), size = (800, 800),
         legend = false, layout = (3, 1),
         bottom_margin = Plots.PlotMeasures.Length(:mm, 20))
```

The horizontal axis is the state $x$, while the vertical axis gives $s(x)$ and $\phi(x)$.

Overall, the policies match well with our predictions from {ref}`section <jvboecalc>`.

* Worker switches from one investment strategy to the other depending on relative return.
* For low values of $x$, the best option is to search for a new job.
* Once $x$ is larger, worker does better by investing in human capital specific to the current position.

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "First Plot Tests" begin
  @test [s_policy[4], phi_policy[4]] ≈ [0.0001, 0.9285785714285715]
  # Canary: value function endpoints depend on full convergence
  @test sol_V[1] ≈ 9.782149056468151
  @test sol_V[end] ≈ 12.041010496888022
end
```

## Exercises

(jv_ex1)=
### Exercise 1

Let's look at the dynamics for the state process $\{x_t\}$ associated with these policies.

The dynamics are given by {eq}`jd` when $\phi_t$ and $s_t$ are
chosen according to the optimal policies, and $\mathbb{P}\{b_{t+1} = 1\}
= \pi(s_t)$.

Since the dynamics are random, analysis is a bit subtle.

One way to do it is to plot, for each $x$ in a relatively fine grid
called `plot_grid`, a
large number $K$ of realizations of $x_{t+1}$ given $x_t =
x$.  Plot this with one dot for each realization, in the form of a 45 degree
diagram.  Set

```{code-block} julia
K = 50
plot_grid_max, plot_grid_size = 1.2, 100
plot_grid = range(0, plot_grid_max, length = plot_grid_size)
plot(plot_grid, plot_grid, color = :black, linestyle = :dash,
     lims = (0, plot_grid_max), legend = :none)
```

By examining the plot, argue that under the optimal policies, the state
$x_t$ will converge to a constant value $\bar x$ close to unity.

Argue that at the steady state, $s_t \approx 0$ and $\phi_t \approx 0.6$.

(jv_ex2)=
### Exercise 2

In the preceding exercise we found that $s_t$ converges to zero
and $\phi_t$ converges to about 0.6.

Since these results were calculated at a value of $\beta$ close to
one, let's compare them to the best choice for an *infinitely* patient worker.

Intuitively, an infinitely patient worker would like to maximize steady state
wages, which are a function of steady state capital.

You can take it as given---it's certainly true---that the infinitely patient worker does not
search in the long run (i.e., $s_t = 0$ for large $t$).

Thus, given $\phi$, steady state capital is the positive fixed point
$x^*(\phi)$ of the map $x \mapsto G(x, \phi)$.

Steady state wages can be written as $w^*(\phi) = x^*(\phi) (1 - \phi)$.

Graph $w^*(\phi)$ with respect to $\phi$, and examine the best
choice of $\phi$.

Can you give a rough interpretation for the value that you see?

## Solutions

### Exercise 1

Here's code to produce the 45 degree diagram

```{code-cell} julia
wp = jv_worker(grid_size = 25)
# simplify notation
(; G, pi_func, F) = wp

v_init = collect(wp.x_grid) * 0.5
f2(v) = T(wp, v)[1]
V2 = fixedpoint(f2, v_init)
sol_V2 = V2.zero
_, (s_policy, phi_policy) = T(wp, sol_V2)

# Turn the policy function arrays into CoordInterpGrid objects for interpolation
s = LinearInterpolation(wp.x_grid, s_policy, extrapolation_bc = Line())
phi = LinearInterpolation(wp.x_grid, phi_policy, extrapolation_bc = Line())

h_func(x, b, U) = (1 - b) * G(x, phi(x)) + b * max(G(x, phi(x)), U)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Solutions 1 Tests" begin
  @test s(3) ≈ 0.0001
  @test phi(4) ≈ 0.2857857142857143
end
```

```{code-cell} julia
using Random
Random.seed!(42)
K = 50

plot_grid_max, plot_grid_size = 1.2, 100
plot_grid = range(0, plot_grid_max, length = plot_grid_size)
ticks = [0.25, 0.5, 0.75, 1.0]

xs = []
ys = []
for x in plot_grid
    for i in 1:K
        b = rand() < pi_func(s(x)) ? 1 : 0
        U = rand(wp.F)
        y = h_func(x, b, U)
        push!(xs, x)
        push!(ys, y)
    end
end

plot(plot_grid, plot_grid, color = :black, linestyle = :dash, legend = :none)
scatter!(xs, ys, alpha = 0.25, color = :green, lims = (0, plot_grid_max),
         ticks = ticks)
plot!(xlabel = L"x_t", ylabel = L"x_{t+1}", guidefont = font(16))
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "More Solutions 1 Tests" begin
  @test round(ys[4], digits = 5) ≈ 0.26326
  @test ticks ≈ [0.25, 0.5, 0.75, 1.0]
  @test plot_grid[1] ≈ 0.0 && plot_grid[end] == plot_grid_max && plot_grid_max ≈ 1.2
  @test length(plot_grid) == plot_grid_size && plot_grid_size == 100
  # Canary: stochastic endpoints depend on many prior draws
  @test ys[1] ≈ 0.46419758957634827
  @test ys[end] ≈ 1.030492516550024
end
```

Looking at the dynamics, we can see that

- If $x_t$ is below about 0.2 the dynamics are random, but
  $x_{t+1} > x_t$ is very likely
- As $x_t$ increases the dynamics become deterministic, and
  $x_t$ converges to a steady state value close to 1

Referring back to the figure {ref}`here <jv_solve>`

we see that $x_t \approx 1$ means that
$s_t = s(x_t) \approx 0$ and
$\phi_t = \phi(x_t) \approx 0.6$.

### Exercise 2

```{code-cell} julia
wp = jv_worker(grid_size = 25)

xbar(phi) = (wp.A * phi^wp.alpha)^(1.0 / (1.0 - wp.alpha))

phi_grid = range(0, 1, length = 100)

plot(phi_grid, [xbar(phi) * (1 - phi) for phi in phi_grid], color = :blue,
     label = L"w^\phi", legendfont = font(12), xlabel = L"\phi",
     guidefont = font(16), grid = false, legend = :topleft)
```

Observe that the maximizer is around 0.6.

This this is similar to the long run value for $\phi$ obtained in
exercise 1.

Hence the behaviour of the infinitely patent worker is similar to that
of the worker with $\beta = 0.96$.

This seems reasonable, and helps us confirm that our dynamic programming
solutions are probably correct.
