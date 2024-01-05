---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.10
---

(mass)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`Asset Pricing I: Finite State Models <single: Asset Pricing I: Finite State Models>`

```{index} single: Models; Markov Asset Pricing
```

```{contents} Contents
:depth: 2
```

```{epigraph}
"A little knowledge of geometric series goes a long way" -- Robert E. Lucas, Jr.
```

```{epigraph}
"Asset pricing is all about covariances" -- Lars Peter Hansen
```

## Overview

```{index} single: Markov Asset Pricing; Overview
```

An asset is a claim on one or more future payoffs.

The spot price of an asset depends primarily on

* the anticipated dynamics for the stream of income accruing to the owners
* attitudes to risk
* rates of time preference

In this lecture we consider some standard pricing models and dividend stream specifications.

We study how prices and dividend-price ratios respond in these different scenarios.

We also look at creating and pricing *derivative* assets by repackaging income streams.

Key tools for the lecture are

* formulas for predicting future values of functions of a Markov state
* a formula for predicting the discounted sum of future values of a Markov state

## {index}`Pricing Models <single: Pricing Models>`

```{index} single: Models; Pricing
```

In what follows let $\{d_t\}_{t \geq 0}$ be a stream of dividends.

* A time-$t$ **cum-dividend** asset is a claim to the stream $d_t, d_{t+1}, \ldots$.
* A time-$t$ **ex-dividend** asset is a claim to the stream $d_{t+1}, d_{t+2}, \ldots$.

Let's look at some equations that we expect to hold for prices of assets under ex-dividend contracts
(we will consider cum-dividend pricing in the exercises).

### Risk Neutral Pricing

```{index} single: Pricing Models; Risk Neutral
```

Our first scenario is risk-neutral pricing.

Let $\beta = 1/(1+\rho)$ be an intertemporal discount factor, where
$\rho$ is the rate at which agents discount the future.

The basic risk-neutral asset pricing equation for pricing one unit of an ex-dividend asset is.

(mass_pra)=
```{math}
:label: rnapex

p_t = \beta {\mathbb E}_t [d_{t+1} + p_{t+1}]
```

This is a simple "cost equals expected benefit" relationship.

Here ${\mathbb E}_t [y]$ denotes the best forecast of $y$, conditioned on information available at time $t$.

### Pricing with Random Discount Factor

```{index} single: Pricing Models; Risk Aversion
```

What happens if for some reason traders discount payouts differently depending on the state of the world?

Michael Harrison and David Kreps {cite}`HarrisonKreps1979` and Lars Peter Hansen
and Scott Richard {cite}`HansenRichard1987` showed that in quite general
settings the price of an ex-dividend asset obeys

```{math}
:label: lteeqs0

p_t = {\mathbb E}_t \left[ m_{t+1}  ( d_{t+1} + p_{t+1} ) \right]
```

for some  **stochastic discount factor** $m_{t+1}$.

The fixed discount factor $\beta$ in {eq}`rnapex` has been replaced by the random variable $m_{t+1}$.

The way anticipated future payoffs are evaluated can now depend on various random outcomes.

One example of this idea is that assets that tend to have good payoffs in bad states of the world might be regarded as more valuable.

This is because they pay well when the funds are more urgently needed.

We give examples of how the stochastic discount factor has been modeled below.

### Asset Pricing and Covariances

Recall that, from the definition of a conditional covariance ${\rm cov}_t (x_{t+1}, y_{t+1})$, we have

```{math}
:label: lteeqs101

{\mathbb E}_t (x_{t+1} y_{t+1}) = {\rm cov}_t (x_{t+1}, y_{t+1}) + {\mathbb E}_t x_{t+1} {\mathbb E}_t y_{t+1}
```

If we apply this definition to the asset pricing equation {eq}`lteeqs0` we obtain

```{math}
:label: lteeqs102

p_t = {\mathbb E}_t m_{t+1} {\mathbb E}_t (d_{t+1} + p_{t+1}) + {\rm cov}_t (m_{t+1}, d_{t+1}+ p_{t+1})
```

It is useful to regard equation {eq}`lteeqs102`   as a generalization of equation {eq}`rnapex`.

* In equation {eq}`rnapex`, the stochastic discount factor $m_{t+1} = \beta$,  a constant.
* In equation {eq}`rnapex`, the covariance term ${\rm cov}_t (m_{t+1}, d_{t+1}+ p_{t+1})$ is zero because $m_{t+1} = \beta$.

Equation {eq}`lteeqs102` asserts that the covariance of the stochastic discount factor with the one period payout $d_{t+1} + p_{t+1}$ is an important determinant of the price $p_t$.

We give examples of some models of stochastic discount factors that have been proposed later in this lecture and also in a {doc}`later lecture <../multi_agent_models/lucas_model>`.

### The Price-Dividend Ratio

Aside from prices, another quantity of interest is the **price-dividend ratio** $v_t := p_t / d_t$.

Let's write down an expression that this ratio should satisfy.

We can divide both sides of {eq}`lteeqs0` by $d_t$ to get

```{math}
:label: pdex

v_t = {\mathbb E}_t \left[ m_{t+1} \frac{d_{t+1}}{d_t} (1 + v_{t+1}) \right]
```

Below we'll discuss the implication of this equation.

## Prices in the Risk Neutral Case

What can we say about price dynamics on the basis of the models described above?

The answer to this question depends on

1. the process we specify for dividends
1. the stochastic discount factor and how it correlates with dividends

For now let's focus on the risk neutral case, where the stochastic discount factor is constant, and study how prices depend on the dividend process.

### Example 1: Constant dividends

The simplest case is risk neutral pricing in the face of a constant, non-random dividend stream $d_t = d > 0$.

Removing the expectation from {eq}`rnapex` and iterating forward gives

$$
\begin{aligned}
    p_t & = \beta (d + p_{t+1})
        \\
        & = \beta (d + \beta(d + p_{t+2}))
        \\
        & \quad \vdots
        \\
        & = \beta (d + \beta d + \beta^2 d +  \cdots + \beta^{k-2} d + \beta^{k-1} p_{t+k})
\end{aligned}
$$

Unless prices explode in the future, this sequence converges to

```{math}
:label: ddet

\bar p := \frac{\beta d}{1-\beta}
```

This price is the equilibrium price in the constant dividend case.

Indeed, simple algebra shows that setting $p_t = \bar p$ for all $t$
satisfies the equilibrium condition $p_t = \beta (d + p_{t+1})$.

### Example 2: Dividends with deterministic growth paths

Consider a growing, non-random dividend process $d_{t+1} = g d_t$
where $0 < g \beta < 1$.

While prices are not usually constant when dividends grow over time, the price
dividend-ratio might be.

If we guess this, substituting $v_t = v$ into {eq}`pdex` as well as our
other assumptions, we get $v = \beta g (1 + v)$.

Since $\beta g < 1$, we have a unique positive solution:

$$
v = \frac{\beta g}{1 - \beta g }
$$

The price is then

$$
p_t = \frac{\beta g}{1 - \beta g } d_t
$$

If, in this example, we take $g = 1+\kappa$ and let
$\rho := 1/\beta - 1$, then the price becomes

$$
p_t = \frac{1 + \kappa}{ \rho - \kappa} d_t
$$

This is called the *Gordon formula*.

(mass_mg)=
### Example 3: Markov growth, risk neutral pricing

Next we consider a dividend process

```{math}
:label: mass_fmce

d_{t+1} = g_{t+1} d_t
```

The stochastic growth factor $\{g_t\}$ is given by

$$
g_t = g(X_t), \quad t = 1, 2, \ldots
$$

where

1. $\{X_t\}$ is a finite Markov chain with state space $S$ and
   transition probabilities

$$
P(x, y) := \mathbb P \{ X_{t+1} = y \,|\, X_t = x \}
\qquad (x, y \in S)
$$

1. $g$ is a given function on $S$ taking positive values

You can think of

* $S$ as $n$ possible "states of the world" and $X_t$ as the
  current state
* $g$ as a function that maps a given state $X_t$ into a growth
  factor $g_t = g(X_t)$ for the endowment
* $\ln g_t = \ln (d_{t+1} / d_t)$ is the growth rate of dividends

(For a refresher on notation and theory for finite Markov chains see {doc}`this lecture <../introduction_dynamics/finite_markov>`)

The next figure shows a simulation, where

* $\{X_t\}$ evolves as a discretized AR1 process produced using {ref}`Tauchen's method <mc_ex3>`
* $g_t = \exp(X_t)$, so that $\ln g_t = X_t$ is the growth rate


```{code-cell} julia
---
tags: [remove-cell]
---
using Test
```

```{code-cell} julia
using LinearAlgebra, Statistics, Random
using LaTeXStrings, Plots, QuantEcon, NLsolve
```

```{code-cell} julia
---
tags: [remove-cell]
---
Random.seed!(42);
```

```{code-cell} julia
n = 25
mc = tauchen(n, 0.96, 0.25)
sim_length = 80

x_series = simulate(mc, sim_length; init = round(Int, n / 2))
g_series = exp.(x_series)
d_series = cumprod(g_series) # assumes d_0 = 1

series = [x_series g_series d_series log.(d_series)]
labels = [L"X_t" L"g_t" L"d_t" L"ln(d_t)"]
plot(series; layout = 4, labels)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
    @test x_series[4] ≈ -0.22321428571428567
    @test g_series[6] ≈ 1.2500884211272658
    @test d_series[9] ≈ 0.5118913634883987
end
```

#### Pricing

To obtain asset prices in this setting, let's adapt our analysis from the case of deterministic growth.

In that case we found that $v$ is constant.

This encourages us to guess that, in the current case, $v_t$ is constant given the state $X_t$.

In other words, we are looking for a fixed function $v$ such that the price-dividend ratio satisfies  $v_t = v(X_t)$.

We can substitute this guess into {eq}`pdex` to get

$$
v(X_t) = \beta {\mathbb E}_t [ g(X_{t+1}) (1 + v(X_{t+1})) ]
$$

If we condition on $X_t = x$, this becomes

$$
v(x) = \beta \sum_{y \in S}  g(y) (1 + v(y)) P(x, y)
$$

or

```{math}
:label: pstack

v(x) = \beta \sum_{y \in S}   K(x, y) (1 + v(y))
\quad \text{where} \quad
K(x, y) := g(y) P(x, y)
```

Suppose that there are $n$ possible states $x_1, \ldots, x_n$.

We can then think of {eq}`pstack` as $n$ stacked equations, one for each state, and write it in matrix form as

```{math}
:label: vcumrn

v = \beta K (\mathbb 1 + v)
```

Here

* $v$ is understood to be the column vector $(v(x_1), \ldots, v(x_n))'$
* $K$ is the matrix $(K(x_i, x_j))_{1 \leq i, j \leq n}$
* ${\mathbb 1}$ is a column vector of ones

When does {eq}`vcumrn` have a unique solution?

From the {ref}`Neumann series lemma <la_neumann>` and Gelfand's formula, this will be the case if $\beta K$ has spectral radius strictly less than one.

In other words, we require that the eigenvalues of $K$  be strictly less than $\beta^{-1}$ in modulus.

The solution is then

```{math}
:label: rned

v = (I - \beta K)^{-1} \beta K{\mathbb 1}
```

### Code

Let's calculate and plot the price-dividend ratio at a set of parameters.

As before, we'll generate $\{X_t\}$  as a {ref}`discretized AR1 process <mc_ex3>` and set $g_t = \exp(X_t)$.

Here's the code, including a test of the spectral radius condition

```{code-cell} julia
n = 25  # size of state space
beta = 0.9
mc = tauchen(n, 0.96, 0.02)

K = mc.p .* exp.(mc.state_values)'

v = (I - beta * K) \ (beta * K * ones(n, 1))

plot(mc.state_values, v; lw = 2, ylabel = "price-dividend ratio",
     xlabel = L"X_t", alpha = 0.7, label = L"v")
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
    @test v[2] ≈ 3.4594684257743284
    @test K[8] ≈ 8.887213530262768e-10
end
```

Why does the price-dividend ratio increase with the state?

The reason is that this Markov process is positively correlated, so high
current states suggest high future states.

Moreover, dividend growth is increasing in the state.

Anticipation of high future dividend growth leads to a high price-dividend ratio.

## Asset Prices under Risk Aversion

Now let's turn to the case where agents are risk averse.

We'll price several distinct assets, including

* The price of an endowment stream.
* A consol (a variety of bond issued by the UK government in the 19th century).
* Call options on a consol.

### Pricing a Lucas tree

```{index} single: Finite Markov Asset Pricing; Lucas Tree
```

Let's start with a version of the celebrated asset pricing model of Robert E. Lucas, Jr. {cite}`Lucas1978`.

As in {cite}`Lucas1978`, suppose that the stochastic discount factor takes the form

```{math}
:label: lucsdf

m_{t+1} = \beta \frac{u'(c_{t+1})}{u'(c_t)}
```

where $u$ is a concave utility function and $c_t$ is time $t$ consumption of a representative consumer.

(A derivation of this expression is given in a {doc}`later lecture <../multi_agent_models/lucas_model>`)

Assume the existence of an endowment that follows {eq}`mass_fmce`.

The asset being priced is a claim on the endowment process.

Following {cite}`Lucas1978`, suppose further that in equilibrium, consumption
is equal to the endowment, so that $d_t = c_t$ for all $t$.

For utility, we'll assume the **constant relative risk aversion** (CRRA)
specification

```{math}
:label: eqCRRA

u(c) = \frac{c^{1-\gamma}}{1 - \gamma} \ {\rm with} \ \gamma > 0
```

When $\gamma =1$ we let $u(c) = \ln c$.

Inserting the CRRA specification into {eq}`lucsdf` and using $c_t = d_t$ gives

```{math}
:label: lucsdf2

m_{t+1}
= \beta \left(\frac{c_{t+1}}{c_t}\right)^{-\gamma}
= \beta g_{t+1}^{-\gamma}
```

Substituting this into {eq}`pdex` gives the price-dividend ratio
formula

$$
v(X_t)
= \beta {\mathbb E}_t
\left[
    g(X_{t+1})^{1-\gamma} (1 + v(X_{t+1}) )
\right]
$$

Conditioning on $X_t = x$, we can write this as

$$
v(x)
= \beta \sum_{y \in S} g(y)^{1-\gamma} (1 + v(y) ) P(x, y)
$$

If we let

$$
J(x, y) := g(y)^{1-\gamma}  P(x, y)
$$

then we can rewrite in vector form as

$$
v = \beta J ({\mathbb 1} + v )
$$

Assuming that the spectral radius of $J$ is strictly less than $\beta^{-1}$, this equation has the unique solution

```{math}
:label: resolvent2

v = (I - \beta J)^{-1} \beta  J {\mathbb 1}
```

We will define a function `tree_price` to solve for $v$ given parameters stored in
the AssetPriceModel objects

The default Markov Chain for will be a discretized AR(1) with $\rho = 0.9, \sigma = 0.02$ and discretized into 25 states using Tauchen's method.

```{code-cell} julia
function AssetPriceModel(; beta = 0.96, gamma = 2.0, g = exp,
                         mc = tauchen(25, 0.9, 0.02))
    return (; beta, gamma, mc, g)
end

# price/dividend ratio of the Lucas tree
function tree_price(ap)
    (; beta, mc, gamma, g) = ap
    P = mc.p
    y = mc.state_values'
    J = P .* g.(y) .^ (1 - gamma)
    @assert maximum(abs, eigvals(J)) < 1 / beta # check stability

    # Compute v
    v = (I - beta * J) \ sum(beta * J, dims = 2)
    return v
end
```

Here's a plot of $v$ as a function of the state for several values of $\gamma$,
with a positively correlated Markov process and $g(x) = \exp(x)$

```{code-cell} julia
gammas = [1.2, 1.4, 1.6, 1.8, 2.0]
p = plot(title = "Price-dividend ratio as a function of the state",
         xlabel = L"X_t", ylabel = "price-dividend ratio")

for gamma in gammas
    ap = AssetPriceModel(; gamma)
    states = ap.mc.state_values
    plot!(states, tree_price(ap); label = L"\gamma = %$gamma")
end
p
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
    ap = AssetPriceModel()
    v = tree_price(ap)
    @test v[5] ≈ 74.54830456854316
    @test v[16] ≈ 29.426651157109678
end
```

Notice that $v$ is decreasing in each case.

This is because, with a positively correlated state process, higher states suggest higher future consumption growth.

In the stochastic discount factor {eq}`lucsdf2`, higher growth decreases the
discount factor, lowering the weight placed on future returns.

#### Special cases

In the special case $\gamma =1$, we have $J = P$.

Recalling that $P^i {\mathbb 1} = {\mathbb 1}$ for all $i$ and applying {ref}`Neumann's geometric series lemma <la_neumann>`, we are led to

$$
v = \beta(I-\beta P)^{-1} {\mathbb 1}
= \beta \sum_{i=0}^{\infty} \beta^i P^i {\mathbb 1}
= \beta \frac{1}{1 - \beta} {\mathbb 1}
$$

Thus, with log preferences, the price-dividend ratio for a Lucas tree is constant.

Alternatively, if $\gamma = 0$, then $J = K$ and we recover the
risk neutral solution {eq}`rned`.

This is as expected, since $\gamma = 0$ implies $u(c) = c$ (and hence agents are risk neutral).

### A Risk-Free Consol

Consider the same pure exchange representative agent economy.

A risk-free consol promises to pay a constant amount  $\zeta> 0$ each period.

Recycling notation, let $p_t$ now be the price of an  ex-coupon claim to the consol.

An ex-coupon claim to the consol entitles the owner at the end of period $t$ to

* $\zeta$ in period $t+1$, plus
* the right to sell the claim for $p_{t+1}$ next period

The price satisfies {eq}`lteeqs0` with $d_t = \zeta$, or

$$
p_t = {\mathbb E}_t \left[ m_{t+1}  ( \zeta + p_{t+1} ) \right]
$$

We maintain the stochastic discount factor {eq}`lucsdf2`, so this becomes

```{math}
:label: consolguess1

p_t
= {\mathbb E}_t \left[ \beta g_{t+1}^{-\gamma}  ( \zeta + p_{t+1} ) \right]
```

Guessing a solution of the form $p_t = p(X_t)$ and conditioning on
$X_t = x$, we get

$$
p(x)
= \beta \sum_{y \in S}  g(y)^{-\gamma} (\zeta + p(y)) P(x, y)
$$

Letting $M(x, y) = P(x, y) g(y)^{-\gamma}$ and rewriting in vector notation
yields the solution

```{math}
:label: consol_price

p = (I - \beta M)^{-1} \beta M \zeta {\mathbb 1}
```

The above is implemented in the function `consol_price`

```{code-cell} julia
function consol_price(ap, zeta)
    (; beta, gamma, mc, g) = ap
    P = mc.p
    y = mc.state_values'
    M = P .* g.(y) .^ (-gamma)
    @assert maximum(abs, eigvals(M)) < 1 / beta

    # Compute price
    return (I - beta * M) \ sum(beta * zeta * M, dims = 2)
end
```

Note that the `sum(Q, dims=2)`  is equivalent to `Q * ones(size(Q)[1], 1)`.

### Pricing an Option to Purchase the Consol

Let's now price options of varying maturity that give the right to purchase a consol at a price $p_S$.

#### An infinite horizon call option

We want to price an infinite horizon  option to purchase a consol at a price $p_S$.

The option entitles the owner at the beginning of a period either to

1. purchase the bond at price $p_S$ now, or
1. Not to exercise the option now but to retain the right to exercise it later

Thus, the owner either *exercises* the option now, or chooses *not to exercise* and wait until next period.

This is termed an infinite-horizon *call option* with *strike price* $p_S$.

The owner of the option is entitled to purchase the consol at the price $p_S$ at the beginning of any period, after the coupon has been paid to the previous owner of the bond.

The fundamentals of the economy are identical with the one above, including the stochastic discount factor and the process for consumption.

Let $w(X_t, p_S)$ be the value of the option when the time $t$ growth state is known to be $X_t$ but *before* the owner has decided whether or not to exercise the option
at time $t$ (i.e., today).

Recalling that $p(X_t)$ is the value of the consol when the initial growth state is $X_t$, the value of the option satisfies

$$
w(X_t, p_S)
= \max \left\{
    \beta \, {\mathbb E}_t \frac{u'(c_{t+1})}{u'(c_t)} w(X_{t+1}, p_S), \;
         p(X_t) - p_S
\right\}
$$

The first term on the right is the value of waiting, while the second is the value of exercising now.

We can also write this as

```{math}
:label: FEoption0

w(x, p_S)
= \max \left\{
    \beta \sum_{y \in S} P(x, y) g(y)^{-\gamma}
    w (y, p_S), \;
    p(x) - p_S
\right\}
```

With $M(x, y) = P(x, y) g(y)^{-\gamma}$ and $w$ as the vector of
values $(w(x_i), p_S)_{i = 1}^n$, we can express {eq}`FEoption0` as the nonlinear vector equation

```{math}
:label: FEoption

w = \max \{ \beta M w, \; p - p_S {\mathbb 1} \}
```

To solve {eq}`FEoption`, form the operator $T$ mapping vector $w$
into vector $Tw$ via

$$
T w
= \max \{ \beta M w,\; p - p_S {\mathbb 1} \}
$$

Start at some initial $w$ and iterate to convergence with $T$, or use a fixed point algorithm.

We can find the solution with the following function call_option

```{code-cell} julia
# price of perpetual call on consol bond
function call_option(ap, zeta, p_s)
    (; beta, gamma, mc, g) = ap
    P = mc.p
    y = mc.state_values'
    M = P .* g.(y) .^ (-gamma)
    @assert maximum(abs, eigvals(M)) < 1 / beta

    # Find consol prices
    p = consol_price(ap, zeta)

    # Operator for fixed point, using consol prices
    T(w) = max.(beta * M * w, p .- p_s)

    # Compute option price as fixed point
    sol = fixedpoint(T, zeros(length(y), 1))
    converged(sol) || error("Failed to converge in $(sol.iterations) iter")
    return sol.zero
end
```

Here's a plot of $w$ compared to the consol price when $P_S = 40$

```{code-cell} julia
ap = AssetPriceModel(; beta = 0.9)
zeta = 1.0
strike_price = 40.0

x = ap.mc.state_values
p = consol_price(ap, zeta)
w = call_option(ap, zeta, strike_price)

plot(x, p, color = "blue", lw = 2, xlabel = L"X_t", label = "consol price")
plot!(x, w, color = "green", lw = 2,
      label = "value of call option with strike at $strike_price")
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
    @test p[17] ≈ 9.302197030956606
    @test w[20] ≈ 0.4610168409491948
end
```

In large states the value of the option is close to zero.

This is despite the fact the Markov chain is irreducible and low states ---
where the consol prices is high --- will eventually be visited.

The reason is that $\beta=0.9$, so the future is discounted relatively rapidly


### Risk Free Rates

Let's look at risk free interest rates over different periods.

#### The one-period risk-free interest rate

As before, the stochastic discount factor is $m_{t+1} = \beta g_{t+1}^{-\gamma}$.

It follows that the reciprocal $R_t^{-1}$ of the gross risk-free interest rate $R_t$ in state $x$ is

$$
{\mathbb E}_t m_{t+1} = \beta \sum_{y \in S} P(x, y) g(y)^{-\gamma}
$$

We can write this as

$$
m_1 = \beta M {\mathbb 1}
$$

where the $i$-th  element of $m_1$ is the reciprocal of the one-period gross risk-free interest rate in state $x_i$.

#### Other terms

Let $m_j$ be an $n \times 1$ vector whose $i$ th component is the reciprocal of the $j$ -period gross risk-free interest rate in state $x_i$.

Then $m_1 = \beta M$, and $m_{j+1} = M m_j$ for $j \geq 1$.

## Exercises

### Exercise 1

In the lecture, we considered **ex-dividend assets**.

A **cum-dividend** asset is a claim to the stream $d_t, d_{t+1}, \ldots$.

Following {eq}`rnapex`, find the risk-neutral asset pricing equation for
one unit of a cum-dividend asset.

With a constant, non-random dividend stream $d_t = d > 0$, what is the equilibrium
price of a cum-dividend asset?

With a growing, non-random dividend process $d_t = g d_t$ where $0 < g \beta < 1$,
what is the equilibrium price of a cum-dividend asset?

### Exercise 2

Consider the following primitives

```{code-cell} julia
n = 5
P = fill(0.0125, n, n) + (0.95 - 0.0125)I
s = [1.05, 1.025, 1.0, 0.975, 0.95]
gamma = 2.0
beta = 0.94
zeta = 1.0
```

Let $g$ be defined by $g(x) = x$  (that is, $g$ is the identity map).

Compute the price of the Lucas tree.

Do the same for

* the price of the risk-free consol when $\zeta = 1$
* the call option on the consol when $\zeta = 1$ and $p_S = 150.0$

### Exercise 3

Let's consider finite horizon call options, which are more common than the
infinite horizon variety.

Finite horizon options obey functional equations closely related to {eq}`FEoption0`.

A $k$ period option expires after $k$ periods.

If we view today as date zero, a $k$ period option gives the owner the right to exercise the option to purchase the risk-free consol at the strike price $p_S$ at dates $0, 1, \ldots , k-1$.

The option expires at time $k$.

Thus, for $k=1, 2, \ldots$, let $w(x, k)$ be the value of a $k$-period option.

It obeys

$$
w(x, k)
= \max \left\{
    \beta \sum_{y \in S} P(x, y) g(y)^{-\gamma}
    w (y, k-1), \;
    p(x) - p_S
\right\}
$$

where $w(x, 0) = 0$ for all $x$.

We can express the preceding as the sequence of nonlinear vector equations

$$
w_k = \max \{ \beta M w_{k-1}, \; p - p_S {\mathbb 1} \}
  \quad k =1, 2, \ldots
  \quad \text{with } w_0 = 0
$$

Write a function that computes $w_k$ for any given $k$.

Compute the value of the option with `k = 5` and `k = 25` using parameter values as in Exercise 1.

Is one higher than the other?  Can you give intuition?

## Solutions

### Exercise 2

```{code-cell} julia
n = 5
P = fill(0.0125, n, n) + (0.95 - 0.0125) * I
s = [0.95, 0.975, 1.0, 1.025, 1.05]  # state values
mc = MarkovChain(P, s)
g = x -> x # identity
gamma = 2.0
beta = 0.94
zeta = 1.0
p_s = 150.0
```

Next we'll create an instance of AssetPriceModel to feed into the functions.

```{code-cell} julia
ap = AssetPriceModel(; beta, mc, gamma, g)
```

Lucas tree prices are 
```{code-cell} julia
tree_price(ap)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
    v = tree_price(ap)
    @test v[2] ≈  21.935706611219704
end
```

Consol Bond Prices
```{code-cell} julia
consol_price(ap, 1.0)
```

```{code-cell} julia
w = call_option(ap, zeta, p_s)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
    @test consol_price(ap, 1.0)[1] ≈ 753.8710047641985
    @test w[2] ≈ 176.83933430191294
end
```

### Exercise 3

Here's a suitable function:

```{code-cell} julia
function finite_horizon_call_option(ap, zeta, p_s, k)
    (; beta, gamma, mc) = ap
    P = mc.p
    y = mc.state_values'
    M = P .* ap.g.(y) .^ (-gamma)
    @assert maximum(abs, eigvals(M)) < 1 / beta

    # Compute option price
    p = consol_price(ap, zeta)

    w = zeros(length(y), 1)
    for i in 1:k
        # Maximize across columns
        w = max.(beta * M * w, p .- p_s)
    end

    return w
end
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset begin
    @test finite_horizon_call_option(ap, zeta, p_s, 5)[3] ≈ 31.798938780647198
end
```

```{code-cell} julia
p = plot(title = "Value Finite Horizon Call Option", xlabel = L"t",
         ylabel = "value")
for k in [5, 25]
    w = finite_horizon_call_option(ap, zeta, p_s, k)
    plot!(w; label = L"k = %$k")
end
p
```

Not surprisingly, the option has greater value with larger $k$.
This is because the owner has a longer time horizon over which he or she
may exercise the option.

