---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.6
---

(optgrowth)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`Optimal Growth I: The Stochastic Optimal Growth Model <single: Optimal Growth I: The Stochastic Optimal Growth Model>`

```{contents} Contents
:depth: 2
```

## Overview

In this lecture we're going to study a simple optimal growth model with one agent.

The model is a version of the standard one sector infinite horizon growth model studied in

* {cite}`StokeyLucas1989`, chapter 2
* {cite}`Ljungqvist2012`, section 3.1
* [EDTC](http://johnstachurski.net/edtc.html), chapter 1
* {cite}`Sundaram1996`, chapter 12

The technique we use to solve the model is dynamic programming.

Our treatment of dynamic programming follows on from earlier
treatments in our lectures on {doc}`shortest paths <../dynamic_programming/short_path>` and
{doc}`job search <../dynamic_programming/mccall_model>`.

We'll discuss some of the technical details of dynamic programming as we
go along.

## The Model

```{index} single: Optimal Growth; Model
```

Consider an agent who owns an amount $y_t \in \mathbb R_+ := [0, \infty)$ of a consumption good at time $t$.

This output can either be consumed or invested.

When the good is invested it is transformed one-for-one into capital.

The resulting capital stock, denoted here by $k_{t+1}$, will then be used for production.

Production is stochastic, in that it also depends on a shock $\xi_{t+1}$ realized at the end of the current period.

Next period output is

$$
y_{t+1} := f(k_{t+1}) \xi_{t+1}
$$

where $f \colon \mathbb{R}_+ \to \mathbb{R}_+$ is called the production function.

The resource constraint is

```{math}
:label: outcsdp0

k_{t+1} + c_t \leq y_t
```

and all variables are required to be nonnegative.

### Assumptions and Comments

In what follows,

* The sequence $\{\xi_t\}$ is assumed to be IID.
* The common distribution of each $\xi_t$ will be denoted $\phi$.
* The production function $f$ is assumed to be increasing and continuous.
* Depreciation of capital is not made explicit but can be incorporated into the production function.

While many other treatments of the stochastic growth model use $k_t$ as the state variable, we will use $y_t$.

This will allow us to treat a stochastic model while maintaining only one state variable.

We consider alternative states and timing specifications in some of our other lectures.

### Optimization

Taking $y_0$ as given, the agent wishes to maximize

```{math}
:label: texs0_og2

\mathbb E \left[ \sum_{t = 0}^{\infty} \beta^t u(c_t) \right]
```

subject to

```{math}
:label: og_conse

y_{t+1} = f(y_t - c_t) \xi_{t+1}
\quad \text{and} \quad
0 \leq c_t \leq y_t
\quad \text{for all } t
```

where

* $u$ is a bounded, continuous and strictly increasing utility function and
* $\beta \in (0, 1)$ is a discount factor

In {eq}`og_conse` we are assuming that the resource constraint {eq}`outcsdp0` holds with equality --- which is reasonable because $u$ is strictly increasing and no output will be wasted at the optimum.

In summary, the agent's aim is to select a path $c_0, c_1, c_2, \ldots$ for consumption that is

1. nonnegative,
1. feasible in the sense of {eq}`outcsdp0`,
1. optimal, in the sense that it maximizes {eq}`texs0_og2` relative to all other feasible consumption sequences, and
1. *adapted*, in the sense that the action $c_t$ depends only on
   observable outcomes, not future outcomes such as $\xi_{t+1}$

In the present context

* $y_t$ is called the *state* variable --- it summarizes the "state of the world" at the start of each period.
* $c_t$ is called the *control* variable --- a value chosen by the agent each period after observing the state.

### The Policy Function Approach

```{index} single: Optimal Growth; Policy Function Approach
```

One way to think about solving this problem is to look for the best **policy function**.

A policy function is a map from past and present observables into current action.

We'll be particularly interested in **Markov policies**, which are maps from the current state $y_t$ into a current action $c_t$.

For dynamic programming problems such as this one (in fact for any [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process)), the optimal policy is always a Markov policy.

In other words, the current state $y_t$ provides a sufficient statistic
for the history in terms of making an optimal decision today.

This is quite intuitive but if you wish you can find proofs in texts such as {cite}`StokeyLucas1989` (section 4.1).

Hereafter we focus on finding the best Markov policy.

In our context, a Markov policy is a function $\sigma \colon
\mathbb R_+ \to \mathbb R_+$, with the understanding that states are mapped to actions via

$$
c_t = \sigma(y_t) \quad \text{for all } t
$$

In what follows, we will call $\sigma$ a *feasible consumption policy* if it satisfies

```{math}
:label: idp_fp_og2

0 \leq \sigma(y) \leq y
\quad \text{for all} \quad
y \in \mathbb R_+
```

In other words, a feasible consumption policy is a Markov policy that respects the resource constraint.

The set of all feasible consumption policies will be denoted by $\Sigma$.

Each $\sigma \in \Sigma$ determines a {doc}`continuous state Markov process <../tools_and_techniques/stationary_densities>` $\{y_t\}$ for output via

```{math}
:label: firstp0_og2

y_{t+1} = f(y_t - \sigma(y_t)) \xi_{t+1},
\quad y_0 \text{ given}
```

This is the time path for output when we choose and stick with the policy $\sigma$.

We insert this process into the objective function to get

```{math}
:label: texss

\mathbb E
\left[ \,
\sum_{t = 0}^{\infty} \beta^t u(c_t) \,
\right] =
\mathbb E
\left[ \,
\sum_{t = 0}^{\infty} \beta^t u(\sigma(y_t)) \,
\right]
```

This is the total expected present value of following policy $\sigma$ forever,
given initial income $y_0$.

The aim is to select a policy that makes this number as large as possible.

The next section covers these ideas more formally.

### Optimality

The **policy value function** $v_{\sigma}$ associated with a given policy $\sigma$ is the mapping defined by

```{math}
:label: vfcsdp00

v_{\sigma}(y) = \mathbb E \left[ \sum_{t = 0}^{\infty} \beta^t u(\sigma(y_t)) \right]
```

when $\{y_t\}$ is given by {eq}`firstp0_og2` with $y_0 = y$.

In other words, it is the lifetime value of following policy $\sigma$
starting at initial condition $y$.

The **value function** is then defined as

```{math}
:label: vfcsdp0

v^*(y) := \sup_{\sigma \in \Sigma} \; v_{\sigma}(y)
```

The value function gives the maximal value that can be obtained from state $y$, after considering all feasible policies.

A policy $\sigma \in \Sigma$ is called **optimal** if it attains the supremum in {eq}`vfcsdp0` for all $y \in \mathbb R_+$.

### The Bellman Equation

With our assumptions on utility and production function, the value function as defined in {eq}`vfcsdp0` also satisfies a **Bellman equation**.

For this problem, the Bellman equation takes the form

```{math}
:label: fpb30

w(y) = \max_{0 \leq c \leq y}
    \left\{
        u(c) + \beta \int w(f(y - c) z) \phi(dz)
    \right\}
\qquad (y \in \mathbb R_+)
```

This is a *functional equation in* $w$.

The term $\int w(f(y - c) z) \phi(dz)$ can be understood as the expected next period value when

* $w$ is used to measure value
* the state is $y$
* consumption is set to $c$

As shown in [EDTC](http://johnstachurski.net/edtc.html), theorem 10.1.11 and a range of other texts.

> *The value function* $v^*$ *satisfies the Bellman equation*

In other words, {eq}`fpb30` holds when $w=v^*$.

The intuition is that maximal value from a given state can be obtained by optimally trading off

* current reward from a given action, vs
* expected discounted future value of the state resulting from that action

The Bellman equation is important because it gives us more information about the value function.

It also suggests a way of computing the value function, which we discuss below.

### Greedy policies

The primary importance of the value function is that we can use it to compute optimal policies.

The details are as follows.

Given a continuous function $w$ on $\mathbb R_+$, we say that $\sigma \in \Sigma$ is $w$-**greedy** if $\sigma(y)$ is a solution to

```{math}
:label: defgp20

\max_{0 \leq c \leq y}
    \left\{
    u(c) + \beta \int w(f(y - c) z) \phi(dz)
    \right\}
```

for every $y \in \mathbb R_+$.

In other words, $\sigma \in \Sigma$ is $w$-greedy if it optimally
trades off current and future rewards when $w$ is taken to be the value
function.

In our setting, we have the following key result

> *A feasible consumption  policy is optimal if and only if it is* $v^*$-*greedy*

The intuition is similar to the intuition for the Bellman equation, which was
provided after {eq}`fpb30`.

See, for example, theorem 10.1.11 of [EDTC](http://johnstachurski.net/edtc.html).

Hence, once we have a good approximation to $v^*$, we can compute the (approximately) optimal policy by computing the corresponding greedy policy.

The advantage is that we are now solving a much lower dimensional optimization
problem.

### The Bellman Operator

How, then, should we compute the value function?

One way is to use the so-called **Bellman operator**.

(An operator is a map that sends functions into functions)

The Bellman operator is denoted by $T$ and defined by

```{math}
:label: fcbell20_optgrowth

Tw(y) := \max_{0 \leq c \leq y}
\left\{
    u(c) + \beta \int w(f(y - c) z) \phi(dz)
\right\}
\qquad (y \in \mathbb R_+)
```

In other words, $T$ sends the function $w$ into the new function
$Tw$ defined {eq}`fcbell20_optgrowth`.

By construction, the set of solutions to the Bellman equation {eq}`fpb30` *exactly coincides with* the set of fixed points of $T$.

For example, if $Tw = w$, then, for any $y \geq 0$,

$$
w(y)
= Tw(y)
= \max_{0 \leq c \leq y}
\left\{
    u(c) + \beta \int v^*(f(y - c) z) \phi(dz)
\right\}
$$

which says precisely that $w$ is a solution to the Bellman equation.

It follows that $v^*$ is a fixed point of $T$.

### Review of Theoretical Results

```{index} single: Dynamic Programming; Theory
```

One can also show that $T$ is a contraction mapping on the set of continuous bounded functions on $\mathbb R_+$ under the supremum distance

$$
\rho(g, h) = \sup_{y \geq 0} |g(y) - h(y)|
$$

See  [EDTC](http://johnstachurski.net/edtc.html), lemma 10.1.18.

Hence it has exactly one fixed point in this set, which we know is equal to the value function.

It follows that

* The value function $v^*$ is bounded and continuous.
* Starting from any bounded and continuous $w$, the sequence $w, Tw, T^2 w, \ldots$ generated by iteratively applying $T$ converges uniformly to $v^*$.

This iterative method is called **value function iteration**.

We also know that a feasible policy is optimal if and only if it is $v^*$-greedy.

It's not too hard to show that a $v^*$-greedy policy exists (see  [EDTC](http://johnstachurski.net/edtc.html), theorem 10.1.11 if you get stuck).

Hence at least one optimal policy exists.

Our problem now is how to compute it.

### {index}`Unbounded Utility <single: Unbounded Utility>`

```{index} single: Dynamic Programming; Unbounded Utility
```

The results stated above assume that the utility function is bounded.

In practice economists often work with unbounded utility functions --- and so will we.

In the unbounded setting, various optimality theories exist.

Unfortunately, they tend to be case specific, as opposed to valid for a large range of applications.

Nevertheless, their main conclusions are usually in line with those stated for
the bounded case just above (as long as we drop the word "bounded").

Consult,  for example, section 12.2 of [EDTC](http://johnstachurski.net/edtc.html), {cite}`Kamihigashi2012` or {cite}`MV2010`.

## Computation

```{index} single: Dynamic Programming; Computation
```

Let's now look at computing the value function and the optimal policy.

### Fitted Value Iteration

```{index} single: Dynamic Programming; Value Function Iteration
```

The first step is to compute the value function by value function iteration.

In theory, the algorithm is as follows

1. Begin with a function $w$ --- an initial condition.
1. Solving {eq}`fcbell20_optgrowth`, obtain the function $T w$.
1. Unless some stopping condition is satisfied, set $w = Tw$ and go to step 2.

This generates the sequence $w, Tw, T^2 w, \ldots$.

However, there is a problem we must confront before we implement this procedure: The iterates can neither be calculated exactly nor stored on a computer.

To see the issue, consider {eq}`fcbell20_optgrowth`.

Even if $w$ is a known function, unless $Tw$ can be shown to have
some special structure, the only way to store it is to record the
value $Tw(y)$ for every $y \in \mathbb R_+$.

Clearly this is impossible.

What we will do instead is use **fitted value function iteration**.

The procedure is to record the value of the function $Tw$ at only finitely many "grid" points $y_1 < y_2 < \cdots < y_I$ and reconstruct it from this information when required.

More precisely, the algorithm will be

(fvi_alg)=
1. Begin with an array of values $\{ w_1, \ldots, w_I \}$ representing the values of some initial function $w$ on the grid points $\{ y_1, \ldots, y_I \}$.
1. Build a function $\hat w$ on the state space $\mathbb R_+$ by interpolation or approximation, based on these data points.
1. Obtain and record the value $T \hat w(y_i)$ on each grid point $y_i$ by repeatedly solving {eq}`fcbell20_optgrowth`.
1. Unless some stopping condition is satisfied, set $\{ w_1, \ldots, w_I \} = \{ T \hat w(y_1), \ldots, T \hat w(y_I) \}$ and go to step 2.

How should we go about step 2?

This is a problem of function approximation, and there are many ways to approach it.

What's important here is that the function approximation scheme must not only produce a good approximation to $Tw$, but also combine well with the broader iteration algorithm described above.

```{only} html
One good choice from both respects is continuous piecewise linear interpolation (see <a href=../_static/pdfs/3ndp.pdf download>this paper</a> for further discussion).
```

```{only} latex
One good choice from both respects is continuous piecewise linear interpolation (see [this paper](https://lectures.quantecon.org/_downloads/3ndp.pdf) for further discussion).
```

The next figure illustrates piecewise linear interpolation of an arbitrary function on grid points $0, 0.2, 0.4, 0.6, 0.8, 1$.


```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Statistics
using Plots, QuantEcon, Interpolations, NLsolve, Optim, Random, Parameters
using Optim: maximum, maximizer
```

```{code-cell} julia
---
tags: [remove-cell]
---
using Test
```

```{code-cell} julia
f(x) = 2 .* cos.(6x) .+ sin.(14x) .+ 2.5
c_grid = 0:.2:1
f_grid = range(0,  1, length = 150)

Af = LinearInterpolation(c_grid, f(c_grid))

plt = plot(xlim = (0,1), ylim = (0,6))
plot!(plt, f, f_grid, color = :blue, lw = 2, alpha = 0.8, label = "true function")
plot!(plt, f_grid, Af.(f_grid), color = :green, lw = 2, alpha = 0.8,
      label = "linear approximation")
plot!(plt, f, c_grid, seriestype = :sticks, linestyle = :dash, linewidth = 2, alpha = 0.5,
      label = "")
plot!(plt, legend = :top)
```

Another advantage of piecewise linear interpolation is that it preserves useful shape properties such as monotonicity and concavity / convexity.

### The Bellman Operator

Here's a function that implements the Bellman operator using linear interpolation

```{code-cell} julia
function T(w;p, tol = 1e-10)
    @unpack β, u, f, ξ, y = p # unpack parameters
    w_func = LinearInterpolation(y, w)

    Tw = similar(w)
    σ = similar(w)
    for (i, y_val) in enumerate(y)
        # solve maximization for each point in y, using y itself as initial condition.
        results = maximize(c -> u(c;p) + β * mean(w_func.(f(y_val - c;p) .* ξ)), tol, y_val) # solver result for each grid point        
        Tw[i] = maximum(results)
        σ[i] = maximizer(results)
    end
    return (;w = Tw, σ) # returns named tuple of results
end
```

Notice that the expectation in {eq}`fcbell20_optgrowth` is computed via Monte Carlo, using the approximation

$$
\int w(f(y - c) z) \phi(dz) \approx \frac{1}{n} \sum_{i=1}^n w(f(y - c) \xi_i)
$$

where $\{\xi_i\}_{i=1}^n$ are IID draws from $\phi$.

Monte Carlo is not always the most efficient way to compute integrals numerically but it does have some theoretical advantages in the present setting.

(For example, it preserves the contraction mapping property of the Bellman operator --- see, e.g., {cite}`pal2013`)

(benchmark_growth_mod)=
### An Example

Let's test out our operator when

* $f(k) = k^{\alpha}$
* $u(c) = \ln c$
* $\phi$ is the distribution of $\exp(\mu + \sigma \zeta)$ when $\zeta$ is standard normal

As is well-known (see {cite}`Ljungqvist2012`, section 3.1.2), for this particular problem an exact analytical solution is available, with

```{math}
:label: dpi_tv

v^*(y) =
\frac{\ln (1 - \alpha \beta) }{ 1 - \beta}
+
\frac{(\mu + \alpha \ln (\alpha \beta))}{1 - \alpha}
 \left[
     \frac{1}{1- \beta} - \frac{1}{1 - \alpha \beta}
 \right]
 +
 \frac{1}{1 - \alpha \beta} \ln y
```

The optimal consumption policy is

$$
\sigma^*(y) = (1 - \alpha \beta ) y
$$

Let's code this up now so we can test against it below

In addition to the model parameters, we need a grid and some shock draws for Monte Carlo integration.

```{code-cell} julia
Random.seed!(42) # for reproducible results
u(c;p) = log(c) # utility
f(k;p) = k^p.α # deterministic part of production function
OptimalGrowthModel = @with_kw (α = 0.4, β = 0.96, μ = 0.0, s = 0.1,
                  u = u, f = f, # defaults defined above
                  y = range(1e-5, 4.0, length = 200), # grid on y
                  ξ = exp.(μ .+ s * randn(250)) # monte carlo shocks
) # named tuples defaults

# True value and policy function
function v_star(y;p)
    @unpack α, μ, β = p
    c1 = log(1 - α * β) / (1 - β)
    c2 = (μ + α * log(α * β)) / (1 - α)
    c3 = 1 / (1 - β)
    c4 = 1 / (1 - α * β)
    return c1 + c2 * (c3 - c4) + c4 * log(y)
end
c_star(y;p) = (1 - p.α * p.β) * y
```


```{code-cell} julia
---
tags: [remove-cell]
---
# # Unused in code, can add back in as required.
#∂u∂c(c) = 1 / c
#f′(k;p) = p.α * k^(p.α - 1)

@testset "Primitives Tests" begin
    @test v_star(3.0;p = OptimalGrowthModel()) ≈ -25.245288867900843
end
```

### A First Test

To test our code, we want to see if we can replicate the analytical solution numerically, using fitted value function iteration.

Now let's do some tests.

As one preliminary test, let's see what happens when we apply our Bellman operator to the exact solution $v^*$.

In theory, the resulting function should again be $v^*$.

In practice we expect some small numerical error.

```{code-cell} julia
p = OptimalGrowthModel() # use all default parameters from named tuple
w_star = v_star.(p.y; p)  # evaluate closed form value along grid

w = T(w_star; p).w # evaluate operator, access Tw results

plt = plot(ylim = (-35,-24))
plot!(plt, p.y, w, linewidth = 2, alpha = 0.6, label = "T(v_star)")
plot!(plt, p.y, w_star, linewidth = 2, alpha=0.6, label = "v_star")
plot!(plt, legend = :bottomright)
```

The two functions are essentially indistinguishable, so we are off to a good start.

Now let's have a look at iterating with the Bellman operator, starting off
from an arbitrary initial condition.

The initial condition we'll start with is $w(y) = 5 \ln (y)$

```{code-cell} julia
w = 5 * log.(p.y)  # An initial condition -- fairly arbitrary
n = 35

plot(xlim = (extrema(p.y)), ylim = (-50, 10))
lb = "initial condition"
plt = plot(p.y, w, color = :black, linewidth = 2, alpha = 0.8, label = lb)
for i in 1:n
    w = T(w; p).w
    plot!(p.y, w, color = RGBA(i/n, 0, 1 - i/n, 0.8), linewidth = 2, alpha = 0.6,
          label = "")
end

lb = "true value function"
plot!(plt, y -> v_star(y; p), grid_y, color = :black, linewidth = 2, alpha = 0.8, label = lb)
plot!(plt, legend = :bottomright)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
    #test v_star(grid_y[2];OptimalGrowthModel()) ≈ -33.370496456772266
end
```

The figure shows

1. the first 36 functions generated by the fitted value function iteration algorithm, with hotter colors given to higher iterates
2. the true value function $v^*$ drawn in black

The sequence of iterates converges towards $v^*$.

We are clearly getting closer.

We can write a function that computes the exact fixed point

```{code-cell} julia
function solve_optgrowth(initial_w; p, iterations = 500, m = 3, show_trace = false) 
    results = fixedpoint(w -> T(w;p).w, initial_w; iterations, m, show_trace) # Anderson iteration
    v_star = results.zero
    σ = T(results.zero;p).σ
    return (;v_star, σ, results)
end
```

We can check our result by plotting it against the true value

```{code-cell} julia
initial_w = 5 * log.(p.y)
sol = solve_optgrowth(initial_w;p)
v_star_approx = sol.v_star
println("Converged in $(sol.results.iterations) to an ||residuals||_∞ of $(sol.results.residual_norm)")

plt = plot(ylim = (-35, -24))
plot!(plt, p.y, v_star_approx, linewidth = 2, alpha = 0.6,
      label = "approximate value function")
plot!(plt, p.y, v_star.(p.y;p), linewidth = 2, alpha = 0.6, label = "true value function")
plot!(plt, legend = :bottomright)
```

The figure shows that we are pretty much on the money

Note that this converges in fewer than the 36 iterations printed above because it is using Anderson iteration - where the $m=0$ parameter is naive fixed point iteration 

### The Policy Function

```{index} single: Optimal Growth; Policy Function
```

To compute an approximate optimal policy, we take the approximate value
function we just calculated and then compute the corresponding greedy policy.

The next figure compares the result to the exact solution, which, as mentioned
above, is $\sigma(y) = (1 - \alpha \beta) y$.

```{code-cell} julia
plt = plot(p.y, T(v_star_approx; p).σ, lw=2, alpha=0.6, label = "approximate policy function")
plot!(plt, p.y, c_star.(p.y; p), lw = 2, alpha = 0.6, label = "true policy function")
plot!(plt, legend = :bottomright)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
    #test c_star.(p.y; p)[102] ≈ 1.2505758978894472
end
```

The figure shows that we've done a good job in this instance of approximating
the true policy.

## Exercises

### Exercise 1

Once an optimal consumption policy $\sigma$ is given, income follows {eq}`firstp0_og2`.

The next figure shows a simulation of 100 elements of this sequence for three different discount factors (and hence three different policies).

```{figure} /_static/figures/solution_og_ex2.png
:width: 100%
```

In each sequence, the initial condition is $y_0 = 0.1$.

The discount factors are `discount_factors = (0.8, 0.9, 0.98)`.

We have also dialed down the shocks a bit.

```{code-cell} julia
---
tags: [remove-cell]
---
Random.seed!(42);
```

```{code-cell} julia
---
tags: [hide-output]
---
p = OptimalGrowthModel(s = 0.05)
p.ξ
```

Otherwise, the parameters and primitives are the same as the log linear model discussed earlier in the lecture.

Notice that more patient agents typically have higher wealth.

Replicate the figure modulo randomness.

## Solutions

### Exercise 1

Here's one solution (assuming as usual that you've executed everything above)

```{code-cell} julia
function simulate_og(σ, p, y0, ts_length)
    y = zeros(ts_length)
    y[1] = y0
    for t in 1:(ts_length-1)
        y[t+1] = (y[t] - σ(y[t]))^p.α * exp(p.μ + p.s * randn())
    end
    return y
end

β_vals = [0.9 0.94 0.98]
ts_length = 100
y0 = 0.1
plt = plot()

for β in β_vals
    p = OptimalGrowthModel(;β) # change β from default
    initial_w = 5 * log.(p.y)
    sol = solve_optgrowth(initial_w;p)
    σ_func = LinearInterpolation(p.y, sol.σ)
    y = simulate_og(σ_func, p,y0, ts_length)

    plot!(plt, 0:(ts_length-1), y, lw = 2, alpha = 0.6, label = "beta = $β")
end
plt
```