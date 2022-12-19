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

plt = plot()
plot!(plt, F, L, label = "Lorenz curve, lognormal sample")
plot!(plt, F, F, label = "Lorenz curve, equality", legend = :topleft)
plt
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
plt = plot()
for a in a_vals
    u = rand(n)
    y = sort(u.^(-1/a))  # distributed as Pareto with tail index a
    (; F, L) = lorenz(y)
    plot!(plt, F, L, label = L"a = %$a")
end
plot!(plt, F, F, label = "equality", legend = :topleft)
plt
```

You can see that, as the tail parameter of the Pareto distribution increases, inequality decreases.

This is to be expected, because a higher tail index implies less weight in the tail of the Pareto distribution.

### In-place Functions and Performance

When working with large vectors and matrices, a key performance advantage of Julia is its ability to manage allocations and perform in-place operations.

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

To compare the performance, we should use the `BenchmarkTools` package.  To test performance, pass in large structures with the `$` (e.g., `$v` and `$L`) or else it copies the values with each sample, confusing performance results.

```{code-cell} julia
n = 1_000_000
a = 2
u = rand(n)
v = sort(u.^(-1/a))
@btime lorenz($v) # performance with out-of-place
```

Note the speed and allocations.  Next use the inplace version

```{code-cell} julia
L = similar(v) # preallocate of same type, size
@btime lorenz!($L, $v)
```

Depending on your system, this should be perhaps twice as fast but have no allocations.

On the other hand, if we use a smaller vector such as `n=1000` above, then the performance difference is much smaller - perhaps only 30% improvement.

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
gini(v) = (2 * sum(i * y for (i,y) in enumerate(v))/sum(v) - (length(v) + 1))/length(v)

a_vals = 1:19
n = 100
ginis = [gini(sort(rand(Weibull(a), n))) for a in a_vals]
ginis_theoretical = [1 - 2^(-1/a) for a in a_vals]

plt = plot()
plot!(plt, a_vals, ginis, label = "estimated gini coefficient")
plot!(plt, a_vals, ginis_theoretical, label = "theoretical gini coefficient")
xlabel!(plt, L"Weibull parameter $a$")
ylabel!(plt, "Gini coefficient")
plt
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
function wealth_dynamics_model(
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
    z_stationary_dist = Normal(z_mean, sqrt(z_var))

    @assert α <= 1 # check stability condition that wealth does not diverge
    return (;w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, a, b, σ_z, z_mean, z_var, z_stationary_dist, exp_z_mean, R_mean, y_mean, α)
end            

# example of model with all default values
wealth_dynamics_model()    
```

## Simulating Wealth Dynamics

To implement this process, first we can write a simple example a loop to simulate an individuals wealth dynamics, then we will write a higher-performance version for an entire distribution.

The `params` argument is a named-tuple or struct consist with the `wealth_dynamics_model` function above.

```{code-cell} julia
function simulate_wealth_dynamics(params, w_0, z_0, T)
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
params = wealth_dynamics_model()  # all defaults
y_0 = params.y_mean
z_0 = rand(params.z_stationary_dist)
T = 200
w, z = simulate_wealth_dynamics(params, y_0, z_0, T)

plot(w, caption = "Wealth simulation", xlabel="t", label=L"w(t)")
```

Notice the large spikes in wealth over time.

Such spikes are similar to what is observed in a time series with a Kesten process.  See [here](https://python.quantecon.org/kesten_processes.html) for a Python implementation.


## Wealth Dynamics of a Distribution

The code above could be used to simulate a number of different households, but it would be relatively slow if the number of households becomes large.

To show higher-performance code, the following code calculates a panel of individuals with in-place operations and vectorized operations.  Pre-allocated arrays where possible.

These sorts of optimizations are frequently overkill, but it is useful to have them in your toolbox when required.

First, lets write an out-of-place function which simulates a cross section of households forward in time.

The following is out of place and easier to read
```{code-cell} julia
using LinearAlgebrafunction step_wealth_optimized(params, w, z)
    N = length(w) # panel size    
    (;w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, a, b, σ_z) = params
    zp = a*z + b + σ_z * randn(N) # vectorized
    y = c_y*exp(zp) + exp(μ_y + σ_y*randn(N))

    # return set to zero if no savings, simplifies vectorization
    R = (w > w_hat)*(c_r*exp(zp) + exp(μ_r + σ_r*randn(N))
    wp = y + R*s_0*w # note R = 0 if not saving since w < w_hat
    return wp, zp
end
```

The following is heavily optimized with mostly in-place operations and fewer allocations.  To completely eliminate allocations, more effort and temporary variables would be required.

The key operations to do in-place operations are `lmul!(v, 2.0)` which does a scalar multiplication inplace, etc.

```{code-cell} julia

function step_wealth_optimized!(w, z, y, R)
    z .= a*z + b + σ_z * randn() # requires temporaries.
    y = c_y*exp(z[t]) + exp(μ_y + σ_y*randn())

    R .= c_r * exp(z[t]) + exp(μ_r + σ_r*randn())
    R .*= (w > w_hat) # set to zero if no savings.
    w .*= R * s_0
w .+= y
end
```

As you can see, the approach is to go through the calculations and build it step by step, avoiding all hidden temporary values.  Typically this is best done by starting with a functioning code and modifying it one step at a time - verifying as you go.

And, as an example to see the impact of these performance steps, we can compare...

As always, note that this may not be worthwhile!

############################
Note the use of parallelization to speed up computation.

```{code-cell} ipython3
@njit(parallel=True)
def update_cross_section(wdy, w_distribution, shift_length=500):
    """
    Shifts a cross-section of household forward in time

    * wdy: an instance of WealthDynamics
    * w_distribution: array_like, represents current cross-section

    Takes a current distribution of wealth values as w_distribution
    and updates each w_t in w_distribution to w_{t+j}, where
    j = shift_length.

    Returns the new distribution.

    """
    new_distribution = np.empty_like(w_distribution)

    # Update each household
    for i in prange(len(new_distribution)):
        z = wdy.z_mean + np.sqrt(wdy.z_var) * np.random.randn()
        w = w_distribution[i]
        for t in range(shift_length-1):
            w, z = wdy.update_states(w, z)
        new_distribution[i] = w
    return new_distribution
```

Parallelization is very effective in the function above because the time path
of each household can be calculated independently once the path for the
aggregate state is known.

## Applications

Let's try simulating the model at different parameter values and investigate
the implications for the wealth distribution.



### Inequality Measures

Let's look at how inequality varies with returns on financial assets.

The next function generates a cross section and then computes the Lorenz
curve and Gini coefficient.

```{code-cell} ipython3
def generate_lorenz_and_gini(wdy, num_households=100_000, T=500):
    """
    Generate the Lorenz curve data and gini coefficient corresponding to a
    WealthDynamics mode by simulating num_households forward to time T.
    """
    ψ_0 = np.full(num_households, wdy.y_mean)
    z_0 = wdy.z_mean

    ψ_star = update_cross_section(wdy, ψ_0, shift_length=T)
    return qe.gini_coefficient(ψ_star), qe.lorenz_curve(ψ_star)
```

Now we investigate how the Lorenz curves associated with the wealth distribution change as return to savings varies.

The code below plots Lorenz curves for three different values of $\mu_r$.

If you are running this yourself, note that it will take one or two minutes to execute.

This is unavoidable because we are executing a CPU intensive task.

In fact the code, which is JIT compiled and parallelized, runs extremely fast relative to the number of computations.

```{code-cell} ipython3
fig, ax = plt.subplots()
μ_r_vals = (0.0, 0.025, 0.05)
gini_vals = []

for μ_r in μ_r_vals:
    wdy = WealthDynamics(μ_r=μ_r)
    gv, (f_vals, l_vals) = generate_lorenz_and_gini(wdy)
    ax.plot(f_vals, l_vals, label=f'$\psi^*$ at $\mu_r = {μ_r:0.2}$')
    gini_vals.append(gv)

ax.plot(f_vals, f_vals, label='equality')
ax.legend(loc="upper left")
plt.show()
```

The Lorenz curve shifts downwards as returns on financial income rise, indicating a rise in inequality.

We will look at this again via the Gini coefficient immediately below, but
first consider the following image of our system resources when the code above
is executing:

(htop_again)=
```{figure} /_static/lecture_specific/wealth_dynamics/htop_again.png
:scale: 80
```

Notice how effectively Numba has implemented multithreading for this routine:
all 8 CPUs on our workstation are running at maximum capacity (even though
four of them are virtual).

Since the code is both efficiently JIT compiled and fully parallelized, it's
close to impossible to make this sequence of tasks run faster without changing
hardware.

Now let's check the Gini coefficient.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(μ_r_vals, gini_vals, label='gini coefficient')
ax.set_xlabel("$\mu_r$")
ax.legend()
plt.show()
```

Once again, we see that inequality increases as returns on financial income
rise.

Let's finish this section by investigating what happens when we change the
volatility term $\sigma_r$ in financial returns.

```{code-cell} ipython3
fig, ax = plt.subplots()
σ_r_vals = (0.35, 0.45, 0.52)
gini_vals = []

for σ_r in σ_r_vals:
    wdy = WealthDynamics(σ_r=σ_r)
    gv, (f_vals, l_vals) = generate_lorenz_and_gini(wdy)
    ax.plot(f_vals, l_vals, label=f'$\psi^*$ at $\sigma_r = {σ_r:0.2}$')
    gini_vals.append(gv)

ax.plot(f_vals, f_vals, label='equality')
ax.legend(loc="upper left")
plt.show()
```

We see that greater volatility has the effect of increasing inequality in this model.

## Exercises

```{exercise}
:label: wd_ex1

For a wealth or income distribution with Pareto tail, a higher tail index suggests lower inequality.

Indeed, it is possible to prove that the Gini coefficient of the Pareto
distribution with tail index $a$ is $1/(2a - 1)$.

To the extent that you can, confirm this by simulation.

In particular, generate a plot of the Gini coefficient against the tail index
using both the theoretical value just given and the value computed from a sample via `qe.gini_coefficient`.

For the values of the tail index, use `a_vals = np.linspace(1, 10, 25)`.

Use sample of size 1,000 for each $a$ and the sampling method for generating Pareto draws employed in the discussion of Lorenz curves for the Pareto distribution.

To the extent that you can, interpret the monotone relationship between the
Gini index and $a$.
```

```{solution-start} wd_ex1
:class: dropdown
```

Here is one solution, which produces a good match between theory and
simulation.

```{code-cell} ipython3
a_vals = np.linspace(1, 10, 25)  # Pareto tail index
ginis = np.empty_like(a_vals)

n = 1000                         # size of each sample
fig, ax = plt.subplots()
for i, a in enumerate(a_vals):
    y = np.random.uniform(size=n)**(-1/a)
    ginis[i] = qe.gini_coefficient(y)
ax.plot(a_vals, ginis, label='sampled')
ax.plot(a_vals, 1/(2*a_vals - 1), label='theoretical')
ax.legend()
plt.show()
```

In general, for a Pareto distribution, a higher tail index implies less weight
in the right hand tail.

This means less extreme values for wealth and hence more equality.

More equality translates to a lower Gini index.

```{solution-end}
```

```{exercise-start}
:label: wd_ex2
```

The wealth process {eq}`wealth_dynam_ah` is similar to a {doc}`Kesten process <kesten_processes>`.

This is because, according to {eq}`sav_ah`, savings is constant for all wealth levels above $\hat w$.

When savings is constant, the wealth process has the same quasi-linear
structure as a Kesten process, with multiplicative and additive shocks.

The Kesten--Goldie theorem tells us that Kesten processes have Pareto tails under a range of parameterizations.

The theorem does not directly apply here, since savings is not always constant and since the multiplicative and additive terms in {eq}`wealth_dynam_ah` are not IID.

At the same time, given the similarities, perhaps Pareto tails will arise.

To test this, run a simulation that generates a cross-section of wealth and
generate a rank-size plot.

If you like, you can use the function `rank_size` from the `quantecon` library (documentation [here](https://quanteconpy.readthedocs.io/en/latest/tools/inequality.html#quantecon.inequality.rank_size)).

In viewing the plot, remember that Pareto tails generate a straight line.  Is
this what you see?

For sample size and initial conditions, use

```{code-cell} ipython3
num_households = 250_000
T = 500                                      # shift forward T periods
ψ_0 = np.full(num_households, wdy.y_mean)   # initial distribution
z_0 = wdy.z_mean
```

```{exercise-end}
```

```{solution-start} wd_ex2
:class: dropdown
```

First let's generate the distribution:

```{code-cell} ipython3
num_households = 250_000
T = 500  # how far to shift forward in time
ψ_0 = np.full(num_households, wdy.y_mean)
z_0 = wdy.z_mean

ψ_star = update_cross_section(wdy, ψ_0, shift_length=T)
```

Now let's see the rank-size plot:

```{code-cell} ipython3
fig, ax = plt.subplots()

rank_data, size_data = qe.rank_size(ψ_star, c=0.001)
ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5)
ax.set_xlabel("log rank")
ax.set_ylabel("log size")

plt.show()
```

```{solution-end}
```
```{solution-end}
```