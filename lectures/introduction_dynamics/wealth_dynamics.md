---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.8
---

(wd)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`Wealth Distribution Dynamics <single: Wealth Distribution Dynamics>`

```{contents} Contents
:depth: 2
```

## Overview

This notebook gives an introduction to wealth distribution dynamics, with a
focus on

* modeling and computing the wealth distribution via simulation,
* measures of inequality such as the Lorenz curve and Gini coefficient, and
* how inequality is affected by the properties of wage income and returns on assets.

One interesting property of the wealth distribution we discuss is Pareto
tails.

The wealth distribution in many countries exhibits a Pareto tail

* The [Pareto Distribution](https://en.wikipedia.org/wiki/Pareto_distribution) is a canonical example of a [heavy-tailed distribution](https://en.wikipedia.org/wiki/Heavy-tailed_distribution).  
* See [here](https://python.quantecon.org/heavy_tails.html) for a lecture on heavy-tailed distributions using Python.
* For a review of the empirical evidence on the wealth distribution, see, for example, {cite}`benhabib2018skewed`.
* See {cite}`Gabaix2009` for a review of the theory and empirics of power-laws and Kesten Processes.

This is consistent with high concentration of wealth amongst the richest households.

It also gives us a way to quantify such concentration, in terms of the tail index.

One question of interest is whether or not we can replicate Pareto tails from a relatively simple model.

### A Note on Assumptions

The evolution of wealth for any given household depends on their
savings behavior.

Modeling such behavior will form an important part of this lecture series.

However, in this particular lecture, we will be content with rather ad hoc (but plausible) savings rules.

We do this to more easily explore the implications of different specifications of income dynamics and investment returns.

At the same time, all of the techniques discussed here can be plugged into models that use optimization to obtain savings rules.

```{code-cell} julia
using Distributions, Plots, LaTeXStrings, LinearAlgebra
```

## Lorenz Curves and the Gini Coefficient

Before we investigate wealth dynamics, we briefly review some measures of
inequality.

### Lorenz Curves

One popular graphical measure of inequality is the [Lorenz curve](https://en.wikipedia.org/wiki/Lorenz_curve).

Since we are using an unweighted Lorenz curve, we can write an efficient version ourselves which implements the simple case of [the definition](https://en.wikipedia.org/wiki/Lorenz_curve#Definition_and_calculation) with equal probabilities and assumes the input vector is already sorted.

The package [Inequality.jl](https://github.com/JosepER/Inequality.jl) can be used for fancier and more general cases.

To illustrate, suppose that

```{code-cell} julia
n = 10_000
w = sort(exp.(randn(n)));  # lognormal draws
```

is data representing the wealth of 10,000 households.

We can compute the Lorenz curve using the [simple, unweighted definition](https://en.wikipedia.org/wiki/Lorenz_curve#Definition_and_calculation):

```{code-cell} julia
function lorenz(v)  # assumed sorted vector
    S = cumsum(v)  # cumulative sums: [v[1], v[1] + v[2], ... ]
    F = (1:length(v)) / length(v)
    L = S ./ S[end]
    return (; F, L) # returns named tuple
end
```

where:

$$
S = [\sum_{j = 1}^1 v_1, \sum_{j = 1}^2 v_2, \dots, \sum_{j = 1}^n v_n]
$$

$$
F = [\frac{1}{n}, \frac{2}{n}, \dots, \frac{n}{n}]
$$

$$
L = [\frac{S_1}{S_n}, \frac{S_2}{S_n}, \dots, \frac{S_n}{S_n}]
$$

We can then plot the curve:

```{code-cell} julia
(; F, L) = lorenz(w)
plot(F, L, label = "Lorenz curve, lognormal sample", legend = :topleft)
plot!(F, F, label = "Lorenz curve, equality")
```

This curve can be understood as follows: if point $(x,y)$ lies on the curve, it means that, collectively, the bottom $(100x)\%$ of the population holds $(100y)\%$ of the wealth.

The "equality" line is the 45 degree line (which might not be exactly 45
degrees in the figure, depending on the aspect ratio).

A sample that produces this line exhibits perfect equality.

The other line in the figure is the Lorenz curve for the lognormal sample, which deviates significantly from perfect equality.

For example, the bottom 80% of the population holds around 40% of total wealth.

Here is another example, which shows how the Lorenz curve shifts as the
underlying distribution changes.

We generate 10,000 observations using the Pareto distribution with a range of
parameters, and then compute the Lorenz curve corresponding to each set of
observations.

```{code-cell} julia
a_vals = (1, 2, 5)
n = 10_000
plt = plot(F, F, label = "equality", legend = :topleft)
for a in a_vals
    u = rand(n)
    y = sort(u.^(-1/a))  # distributed as Pareto with tail index a
    (; F, L) = lorenz(y)
    plot!(plt, F, L, label = L"a = %$a")
end
plt
```

You can see that, as the tail parameter of the Pareto distribution increases, inequality decreases.

This is to be expected, because a higher tail index implies less weight in the tail of the Pareto distribution.

### The Gini Coefficient

The definition and interpretation of the Gini coefficient can be found on the corresponding [Wikipedia page](https://en.wikipedia.org/wiki/Gini_coefficient).

A value of 0 indicates perfect equality (corresponding the case where the
Lorenz curve matches the 45 degree line) and a value of 1 indicates complete
inequality (all wealth held by the richest household).

Since we are using an unweighted Gini coefficient, we can write an efficient version ourselves using a [simplification](https://en.wikipedia.org/wiki/Gini_coefficient#Alternative_expressions) and assuming the input vector is already sorted.

The [Inequality.jl](https://github.com/JosepER/Inequality.jl) package can be used for a more complete implementation, including weighted Gini indices.

We can test it on the Weibull distribution with parameter $a$, where the Gini coefficient is known to be

$$
G = 1 - 2^{-1/a}
$$

Let's see if the Gini coefficient computed from a simulated sample matches
this at each fixed value of $a$.

```{code-cell} julia
gini(v) = (2 * sum(i * y for (i,y) in enumerate(v))/sum(v)
           - (length(v) + 1))/length(v)

a_vals = 1:19
n = 100
ginis = [gini(sort(rand(Weibull(a), n))) for a in a_vals]
ginis_theoretical = [1 - 2^(-1/a) for a in a_vals]

plot(a_vals, ginis, label = "estimated gini coefficient",
     xlabel = L"Weibull parameter $a$", ylabel = "Gini coefficient")
plot!(a_vals, ginis_theoretical, label = "theoretical gini coefficient")
```

The simulation shows that the fit is good.

## A Model of Wealth Dynamics

Having discussed inequality measures, let us now turn to wealth dynamics.

The model we will study is

```{math}
:label: wealth_dynam_ah

w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```

where

- $w_t$ is wealth at time $t$ for a given household,
- $r_t$ is the rate of return of financial assets,
- $y_t$ is current non-financial (e.g., labor) income and
- $s(w_t)$ is current wealth net of consumption

Letting $\{z_t\}$ be a correlated state process of the form

$$
z_{t+1} = a z_t + b + \sigma_z \epsilon_{t+1}
$$

we’ll assume that

$$
R_t := 1 + r_t = c_r \exp(z_t) + \exp(\mu_r + \sigma_r \xi_t)
$$

and

$$
y_t = c_y \exp(z_t) + \exp(\mu_y + \sigma_y \zeta_t)
$$

Here $\{ (\epsilon_t, \xi_t, \zeta_t) \}$ is IID and standard
normal in $\mathbb R^3$.

The value of $c_r$ should be close to zero, since rates of return
on assets do not exhibit large trends.

When we simulate a population of households, we will assume all shocks are idiosyncratic (i.e.,  specific to individual households and independent across them).

Regarding the savings function $s$, our default model will be

```{math}
:label: sav_ah

s(w) = s_0 w \cdot \mathbb 1\{w \geq \hat w\}
```

where $s_0$ is a positive constant.

Thus, for $w < \hat w$, the household saves nothing. For
$w \geq \bar w$, the household saves a fraction $s_0$ of
their wealth.

We are using something akin to a fixed savings rate model, while
acknowledging that low wealth households tend to save very little.

## Implementation

First, we will write a function which collects all of the parameters into a named tuple.  While we could also use a Julia `struct` (see {doc}`creating new types <../getting_started_julia/introduction_to_types>`) they tend to be more difficult to use properly.

```{code-cell} julia
function wealth_dynamics_model(; # all named arguments
                 w_hat=1.0, # savings parameter
                 s_0=0.75, # savings parameter
                 c_y=1.0, # labor income parameter
                 μ_y=1.0, # labor income parameter
                 σ_y=0.2, # labor income parameter
                 c_r=0.05, # rate of return parameter
                 μ_r=0.1, # rate of return parameter
                 σ_r=0.5, # rate of return parameter
                 a=0.5, # aggregate shock parameter
                 b=0.0, # aggregate shock parameter
                 σ_z=0.1 # aggregate shock parameter
                 )
    z_mean = b / (1 - a)
    z_var = σ_z^2 / (1 - a^2)
    exp_z_mean = exp(z_mean + z_var / 2)
    R_mean = c_r * exp_z_mean + exp(μ_r + σ_r^2 / 2)
    y_mean = c_y * exp_z_mean + exp(μ_y + σ_y^2 / 2)
    α = R_mean * s_0

    # Distributions
    z_stationary_dist = Normal(z_mean, sqrt(z_var))

    @assert α <= 1 # check stability condition that wealth does not diverge
    return (;w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, a, b, σ_z, z_mean, z_var,
             z_stationary_dist, exp_z_mean, R_mean, y_mean, α)
end            

# example of model with all default values
wealth_dynamics_model()    
```

## Simulating Wealth Dynamics

To implement this process, first we can write a simple example a loop to simulate an individuals wealth dynamics, then we will write a higher-performance version for an entire distribution.

The `params` argument is a named-tuple or struct consist with the `wealth_dynamics_model` function above.

```{code-cell} julia
function simulate_wealth_dynamics(w_0, z_0, T, params)
    (;w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, a, b, σ_z) = params # unpack
    w = zeros(T+1)
    z = zeros(T+1)
    w[1] = w_0
    z[1] = z_0

    for t = 2:T+1
        z[t] = a*z[t-1] + b + σ_z * randn()
        y = c_y*exp(z[t]) + exp(μ_y + σ_y*randn())
        w[t] = y
        if w[t-1] >= w_hat # if above minimum wealth level, add savings
            R = c_r * exp(z[t]) + exp(μ_r + σ_r*randn())
            w[t] += R * s_0 * w[t-1]
        end
    end    
    return w, z
end
```
Let's look at the wealth dynamics of an individual household.

```{code-cell} julia
p = wealth_dynamics_model()  # all defaults
y_0 = p.y_mean
z_0 = rand(p.z_stationary_dist)
T = 200
w, z = simulate_wealth_dynamics(y_0, z_0, T, p)

plot(w, caption = "Wealth simulation", xlabel="t", label=L"w(t)")
```

Notice the large spikes in wealth over time.

Such spikes are similar to what is observed in a time series with a Kesten process.  See [here](https://python.quantecon.org/kesten_processes.html) for a related lecture using Python.


## Inequality Measures

Let's look at how inequality varies with returns on financial assets

The code above could be used to simulate a number of different households, but it would be relatively slow if the number of households becomes large.

First, lets write a function which simulates a single time-step for a cross section of households.

```{code-cell} julia
function step_wealth(w, z, params)
    N = length(w) # panel size    
    (;w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, a, b, σ_z) = params
    zp = a*z .+ b .+ σ_z * randn(N) # vectorized
    y = c_y*exp.(zp) .+ exp.(μ_y .+ σ_y*randn(N))

    # return set to zero if no savings, simplifies vectorization
    R = (w .> w_hat).*(c_r*exp.(zp) .+ exp.(μ_r .+ σ_r*randn(N)))
    wp = y .+ s_0*R.*w # note R = 0 if not saving since w < w_hat
    return wp, zp
end
```
<!--
Testing code
w = rand(10)
z = rand(10)
step_wealth(w, z, params)
-->

Using this function, we can iterate forward from a distribution of wealth and income

```{code-cell} julia
function simulate_panel(N, T, params)
    y_0 = params.y_mean * ones(N) # start at the mean
    z_0 = rand(params.z_stationary_dist, N)

    # iterate forward from initial condition
    w = y_0 # start at mean of income process
    z = z_0
    for t in 1:T
        w, z = step_wealth(w, z, params) # steps forward, discarding results
    end
    sort!(w) # sorts the wealth so we can calculate gini/lorenz        
    F, L = lorenz(w)
    return (;w, F, L, gini = gini(w))
end
```

To use this function, we pass in parameters and can access the resulting wealth distribution and inequality measures.

```{code-cell} julia
p = wealth_dynamics_model()
res = simulate_panel(100_000, 200, p)
@show median(res.w), mean(res.w), res.gini;
```

Now we investigate how the Lorenz curves associated with the wealth distribution change as return to savings varies.

The code below simulates the wealth distribution, Lorenz curve, and gini for multiple values of $\mu_r$.

```{code-cell} julia
N = 100_000
T = 500
μ_r_vals = LinRange(0.0, 0.075, 5)
results = map(μ_r -> simulate_panel(N, T, wealth_dynamics_model(;μ_r)), μ_r_vals);
```

Using these results, we can plot the Lorenz curves for each value of $\mu_r$ and compare to perfect equality.

```{code-cell} julia
plt = plot(F, F, label = "equality", legend = :topleft, ylabel="Lorenz Curve")
[plot!(plt, res.F, res.L, label = L"\psi^*, \mu_r = %$(round(μ_r; sigdigits=1))") 
 for (μ_r, res) in zip(μ_r_vals, results)]
plt
```

The Lorenz curve shifts downwards as returns on financial income rise, indicating a rise in inequality.


Now let’s check the Gini coefficient.
```{code-cell} julia
ginis = [res.gini for res in results]
plot(μ_r_vals, ginis, label = "Gini coefficient", xlabel = L"\mu_r")
```
Once again, we see that inequality increases as returns on financial income rise, and the relationship is roughly linear.

Let's finish this section by investigating what happens when we change the
volatility term $\sigma_r$ in financial returns.

```{code-cell} julia
σ_r_vals = LinRange(0.35, 0.53, 5)
results = map(σ_r -> simulate_panel(N, T, wealth_dynamics_model(;σ_r)), σ_r_vals);
plt = plot(F, F, label = "equality", legend = :topleft, ylabel="Lorenz Curve")
[plot!(plt, res.F, res.L, label = L"\psi^*, \sigma_r = %$(round(σ_r; sigdigits=2))") for (σ_r, res) in zip(σ_r_vals, results)]
plt
```

We see that greater volatility has the effect of increasing inequality in this model.

```{code-cell} julia
ginis = [res.gini for res in results]
plot(σ_r_vals, ginis, label = "Gini coefficient", xlabel = L"\sigma_r")
```

Similarly, the Gini coefficient shows that greater volatility increases inequality and approaches a Gini of 1 (i.e., perfect inequality) as the volatility increases where a $\sigma_r \approx 0.53$ is close to the maximum value fixing the other parameters at their default values.

In this case, the divergence occurs as the $\alpha < 1$ condition begins to fail because high volatility increases mean rate of return, leading to explosive savings behavior.

## In-place Functions and Performance

When working with large vectors and matrices, a performance advantage of Julia is its ability to manage allocations and perform in-place operations.

As always, don't prematurely optimize your code - but in cases where the datastructures are large and the code is of equivalent complexity, don't be afraid to use in-place operations.

To demonstrate this, we will compare an inplace Lorenz calculation with the one above. 

The convention in Julia is to use `!` to denote a function which mutates its arguments and to put any arguments that will be modified first.

In the following case, the `L` is pre-allocated and will be overwritten.

```{code-cell} julia
function lorenz!(L, v)
   # cumulative sum but inplace: [v[1], v[1] + v[2], ... ]
    cumsum!(L, v)    
    L ./= L[end]  # inplace division to normalize
    F = (1:length(v)) / length(v)  # this doesn't allocate anything so no need to be inplace
    return F, L # using inplace we can still return the L vector
end
```

```{admonition} Performance benchmarking
For performance comparisons, always use the [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).  Whenever passing in arguments that are not scalars, you typically want to [interpolate](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/#Interpolating-values-into-benchmark-expressions) by prepending with the `$` (e.g., `@btime lorenz($v)` rather than `@btime lorenz(v)`) or else it may treat the value as a global variable and gives distorted performance results.

For more detailed results, replace `@btime` with `@benchmark`.
```

```{code-cell} julia
using BenchmarkTools
n = 1_000_000
a = 2
u = rand(n)
v = sort(u.^(-1/a))
@btime lorenz($v); # performance with out-of-place
```

Note the speed and allocations.  Next use the inplace version

```{code-cell} julia
L = similar(v) # preallocate of same type, size
@btime lorenz!($L, $v);
```

Depending on your system, this should be perhaps twice as fast but have no allocations.

On the other hand, if we use a smaller vector such as `n=1000` above, then the performance difference is much smaller - perhaps only 30% improvement.

Furthermore, this benefit is only felt if we are reusing the same `L` in repeated calls.  If we need to allocate (e.g. a `L = similar(v)`) each time, then there is no benefit.
<!--
```{code-cell} julia
n = 1000
a = 2
u = rand(n)
v = sort(u.^(-1/a))
L = similar(v) # preallocate of same type, size
@btime lorenz($v)
@btime lorenz!($L, $v)
```
--> 

This provides a common and cautionary lesson: for some algorithms, avoiding allocations does not have a significant difference and may not be worth the trouble.

This all depends on the steps of the underlying algorithm.  In the case above, the `cumsum` is significantly more expensive than the data allocation.

In other cases, such as those in large-scale difference or differential equations, in-place operations can have an enormous impact.

### (Premature) Performance Optimizations of the Simulation
Before asking how we can make our simulations faster, we should ask whether it is already fast enough for our needs - which it seems to be in this case for the above figures.

That said, lets consider if we had a compelling need to handle much larger panels or variations on simulating for many more parameters.

Minimizing allocations and doing calculations in-place can lead to faster code - though usually in the case of {doc}`numerical methods for linear algebra <../tools_and_techniques/numerical_linear_algebra>` and  In {doc}`iterative methods  <../tools_and_techniques/iterative_methods_sparsity>`.

Example operations for this are the in-place versions of existing functions (e.g., `lmul!(v, 2.0)` for inplace scalar multiplication or `mul!(y, A, x)` for inplace matrix-multiplication).

Using these operations, we can carefully rewrite our functions to be in-place and modify `w` and `z` rather than return new vectors each time.

In addition, a standard trick is to pass in a pre-allocated buffer which is replaced in each iteration, which we provide for the `y` and `R`.

```{code-cell} julia
function step_wealth!(w, z, y, R, params)
    N = length(w) # panel size    
    (;w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, a, b, σ_z) = params

    # Splitting up old: zp = a*z .+ b .+ σ_z * randn(N) # vectorized
    shock_z = randn(N)
    lmul!(σ_z, shock_z) # σ_z * randn(N)
    shock_z .+= b # b .+ σ_z * randn(N)
    lmul!(a, z) # a*z
    z .+= shock_z # zp = a*z .+ b .+ σ_z * randn(N)

    # Splitting up old: y = c_y*exp.(zp) .+ exp.(μ_y .+ σ_y*randn(N))
    exp_zp = exp.(z)
    copy!(y, exp_zp) # y = exp.(zp)
    lmul!(c_y, y) # y = c_y*exp.(zp)
    shock_y = randn(N)
    lmul!(σ_y, shock_y) # σ_y*randn(N)
    shock_y .+= μ_y # μ_y .+ σ_y*randn(N)    
    y .+= exp.(shock_y) # y = c_y*exp.(zp) .+ exp.(μ_y .+ σ_y*randn(N))

    # Split up the: R .= (w .> w_hat).*(c_r*exp.(zp) .+ exp.(μ_r .+ σ_r*randn(N)))
    shock_R = randn(N)
    lmul!(σ_r, shock_R) # σ_r*randn(N)
    shock_R .+= μ_r # μ_r .+ σ_y*randn(N)    
    copy!(R, exp_zp) # exp.(zp)
    lmul!(c_r, R) # c_r*exp.(zp)
    R .+= exp.(shock_R) # c_r*exp.(zp) .+ exp.(μ_r .+ σ_r*randn(N))
    R .*= (w .> w_hat) # (w .> w_hat).*(c_r*exp.(zp) .+ exp.(μ_r .+ σ_r*randn(N)))

    # Split up the: wp = y .+ s_0*R.*w
    lmul!(s_0, w) # s_0*w
    w .*= R # s_0*R.*w
    w .+= y # wp = y .+ s_0*R.*w
    return w, z
end
function simulate_panel!(N, T, params)
    y_0 = params.y_mean * ones(N) # start at the mean
    z_0 = rand(params.z_stationary_dist, N)

    # iterate forward from initial condition
    w = y_0 # start at mean of income process
    z = z_0
    y = similar(z)
    R = similar(z)
    for t in 1:T
        step_wealth!(w, z, y, R, params) # steps forward in-place
    end
    sort!(w) # sorts the wealth so we can calculate gini/lorenz        
    F, L = lorenz(w)
    return (;w, F, L, gini = gini(w))
end
```
As you can see, this was a lot of work, most of which was attempting to eliminate temporary values.

We can identify those temporaries because of the assignment (i.e., `wp = ` typically allocates where `wp .= ` overwrites).

Was all of this trouble worthwhile?  Let's first benchmark the old version

```{code-cell} julia
p = wealth_dynamics_model()
N = 100_000
T = 200
@btime simulate_panel(N, T, $p);
```

Then compare,

```{code-cell} julia
@btime simulate_panel!(N, T, $p);
```

All of that work led to only a modest speedup of less than a factor of 50% on most systems.

The allocations decrease more substantially (e.g., a factor of 4) but that does not seem to be the key bottleneck.

This is a common result of micro-optimizations - a lot of work with very little gain.  Don't optimize code unless you can justify the benefits in hours of work leading to only modest speedups.

As discussed above, cases with more significant linear algebra can be much more amenable to in-place operations and lead to more significant speedups as discussed in {doc}`iterative methods  <../tools_and_techniques/iterative_methods_sparsity>` and related lectures.