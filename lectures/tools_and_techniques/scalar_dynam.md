---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.7
---

(scalar_dynam)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`Dynamics in One Dimension <single: Dynamics in One Dimension>`

```{contents} Contents
:depth: 2
```

## Overview

In this lecture we give a quick introduction to discrete time dynamics in one
dimension.

In one-dimensional models, the state of the system is described by a single variable.

Although most interesting dynamic models have two or more state variables, the
one-dimensional setting is a good place to learn the foundations of dynamics and build
intuition.

We'll use the following packages:

```{code-cell} julia
using LaTeXStrings, LinearAlgebra, Plots
```

## Some Definitions

This section sets out the objects of interest and the kinds of properties we study.

### Difference Equations

A **time homogeneous first order difference equation** is an equation of the
form

```{math}
:label: sdsod

x_{t+1} = g(x_t)
```

where $g$ is a function from some subset $S$ of $\mathbb R$ to itself.

Here $S$ is called the **state space** and $x$ is called the **state variable**.

In the definition,

* time homogeneity means that $g$ is the same at each time $t$
* first order means dependence on only one lag (i.e., earlier states such as $x_{t-1}$ do not enter into {eq}`sdsod`).

If $x_0 \in S$ is given, then {eq}`sdsod` recursively defines the sequence

```{math}
:label: sdstraj

x_0, \quad
x_1 = g(x_0), \quad
x_2 = g(x_1) = g(g(x_0)), \quad \text{etc.}
```

This sequence is called the **trajectory** of $x_0$ under $g$.

If we define $g^n$ to be $n$ compositions of $g$ with itself, then we can write the trajectory more simply as $x_t = g^t(x_0)$ for $t \geq 0$.

### Example: A Linear Model

One simple example is the **linear difference equation**

$$
x_{t+1} = a x_t + b, \qquad S = \mathbb R
$$

where $a, b$ are fixed constants.

In this case, given $x_0$, the trajectory {eq}`sdstraj` is

```{math}
:label: sdslinmodpath

x_0, \quad
a x_0 + b, \quad
a^2 x_0 + a b + b, \quad \text{etc.}
```

Continuing in this way, and using our knowledge of {doc}`geometric series <../tools_and_techniques/geom_series>`, we find that, for any $t \geq 0$,

```{math}
:label: sdslinmod

x_t = a^t x_0 + b \frac{1 - a^t}{1 - a}
```

This is about all we need to know about the linear model.

We have an exact expression for $x_t$ for all $t$ and hence a full
understanding of the dynamics.

Notice in particular that $|a| < 1$, then, by {eq}`sdslinmod`, we have

```{math}
:label: sdslinmodc

x_t \to  \frac{b}{1 - a} \text{ as } t \to \infty
```

regardless of $x_0$

This is an example of what is called global stability, a topic we return to
below.

### Example: A Nonlinear Model

In the linear example above, we obtained an exact analytical expression for $x_t$
in terms of arbitrary $t$ and $x_0$.

This made analysis of dynamics very easy.

When models are nonlinear, however, the situation can be quite different.

For example, consider the law of motion for the Solow growth model, a simplified version of which is

```{math}
:label: solow_lom2

k_{t+1} = s z k_t^{\alpha} + (1 - \delta) k_t
```

Here $k$ is capital stock and $s, z, \alpha, \delta$ are positive
parameters with $0 < \alpha, \delta < 1$.

If you try to iterate like we did in {eq}`sdslinmodpath`, you will find that
the algebra gets messy quickly.

Analyzing the dynamics of this model requires a different method (see below).

### Stability

A **steady state** of the difference equation $x_{t+1} = g(x_t)$ is a
point $x^*$ in $S$ such that $x^* = g(x^*)$.

In other words, $x^*$ is a **fixed point** of the function $g$ in
$S$.

For example, for the linear model $x_{t+1} = a x_t + b$, you can use the
definition to check that

* $x^* := b/(1-a)$ is a steady state whenever $a \not= 1$.
* if $a = 1$ and $b=0$, then every $x \in \mathbb R$ is a
  steady state.
* if $a = 1$ and $b \not= 0$, then the linear model has no steady
  state in $\mathbb R$.

A steady state $x^*$ of $x_{t+1} = g(x_t)$ is called
**globally stable** if, for all $x_0 \in S$,

$$
x_t = g^t(x_0) \to x^* \text{ as } t \to \infty
$$

For example, in the linear model $x_{t+1} = a x_t + b$ with $a
\not= 1$, the steady state $x^*$

* is globally stable if $|a| < 1$ and
* fails to be globally stable otherwise.

This follows directly from {eq}`sdslinmod`.

A steady state $x^*$ of $x_{t+1} = g(x_t)$ is called
**locally stable** if there exists an $\epsilon > 0$ such that

$$
| x_0 - x^* | < \epsilon
\; \implies \;
x_t = g^t(x_0) \to x^* \text{ as } t \to \infty
$$

Obviously every globally stable steady state is also locally stable.

We will see examples below where the converse is not true.

## Graphical Analysis

As we saw above, analyzing the dynamics for nonlinear models is nontrivial.

There is no single way to tackle all nonlinear models.

However, there is one technique for one-dimensional models that provides a
great deal of intuition.

This is a graphical approach based on **45 degree diagrams**.

Let's look at an example: the Solow model with dynamics given in {eq}`solow_lom2`.

We begin with some plotting code that you can ignore at first reading.

The function of the code is to produce 45 degree diagrams and time series
plots.

```{code-cell} julia
function plot45(g, xmin, xmax, x0, num_arrows = 6)

    xgrid = LinRange(xmin, xmax, 200)
    
    plt = plot(xgrid, xgrid, xlim = (xmin, xmax), ylim = (xmin, xmax), linecolor = :black, lw = 2)
    plot!(plt, xgrid, g.(xgrid), linecolor = :blue, lw = 2)
    
    x = x0

    arrow_iterator = 0:(num_arrows-1)

    arrow_kwargs = (arrow = :closed, linecolor = :black, alpha = 0.5)
    dash_kwargs = (linestyle = :dash, linecolor = :black, alpha = 0.5)

    xticks = zeros(num_arrows) # Is there a better style than this?
 
    for (i, j) in enumerate(arrow_iterator)

        xticks[i] = x

        if i == 1
            plot!([x, x], [0, g.(x)]; arrow_kwargs...)
            plot!([x, g.(x)], [g.(x), g.(x)]; arrow_kwargs...)
        else
            plot!([x, x], [x, g.(x)]; arrow_kwargs...)
            plot!([x, x], [0, x]; dash_kwargs...)
            plot!([x, g.(x)], [g.(x), g.(x)]; arrow_kwargs...)

        end

        x = g.(x)
        
        plot!([x, x], [0, x], xticks = (xticks, [L"k_%$j" for j in arrow_iterator]), yticks = (xticks, [L"k_%$j" for j in arrow_iterator]); dash_kwargs...)
    end

    plot!([x, x], [0, x], legend = false; dash_kwargs...) # superfluous line of code, trying to add the final xtick
end    

function ts_plot(g, xmin, xmax, x0; ts_length=6)

    x = zeros(ts_length)

    x[1] = x0

    for t in 1:(ts_length-1)
        x[t+1] = g.(x[t])
    end

    plot(1:ts_length, x, ylim = (xmin, xmax), linecolor = :blue, lw = 2, alpha = 0.7)
    scatter!(x, mc = :blue, alpha = 0.7, legend = false)

end
```

Let's create a 45 degree diagram for the Solow model with a fixed set of
parameters

```{code-cell} julia
p = (A = 2, s = 0.3, alpha = 0.3, delta = 0.4, xmin = 0, xmax = 4)
```

Here's the update function corresponding to the model.

```{code-cell} julia
g = k -> p.A * p.s * k ^ p.alpha + (1 - p.delta) * k
```

Here is the 45 degree plot.

```{code-cell} julia
plot45(g, p.xmin, p.xmax, 0, num_arrows=0)
```

The plot shows the function $g$ and the 45 degree line.

Think of $k_t$ as a value on the horizontal axis.

To calculate $k_{t+1}$, we can use the graph of $g$ to see its
value on the vertical axis.

Clearly,

* If $g$ lies above the 45 degree line at this point, then we have $k_{t+1} > k_t$.
* If $g$ lies below the 45 degree line at this point, then we have $k_{t+1} < k_t$.
* If $g$ hits the 45 degree line at this point, then we have $k_{t+1} = k_t$, so $k_t$ is a steady state.

For the Solow model, there are two steady states when $S = \mathbb R_+ =
[0, \infty)$.

* the origin $k=0$
* the unique positive number such that $k = s z k^{\alpha} + (1 - \delta) k$.

By using some algebra, we can show that in the second case, the steady state is

$$
k^* = \left( \frac{sz}{\delta} \right)^{1/(1-\alpha)}
$$

### Trajectories

By the preceding discussion, in regions where $g$ lies above the 45 degree line, we know that the trajectory is increasing.

The next figure traces out a trajectory in such a region so we can see this more clearly.

The initial condition is $k_0 = 0.25$.

```{code-cell} julia
k0 = 0.25

plot45(g, p.xmin, p.xmax, k0, num_arrows=5)
```

We can plot the time series of capital corresponding to the figure above as
follows:

```{code-cell} julia
ts_plot(g, p.xmin, p.xmax, k0)
```

Here's a somewhat longer view:

```{code-cell} julia
ts_plot(g, p.xmin, p.xmax, k0, ts_length=20)
```

When capital stock is higher than the unique positive steady state, we see that
it declines:

```{code-cell} julia
k0 = 2.95

plot45(g, p.xmin, p.xmax, k0, num_arrows=5)
```

Here is the time series:

```{code-cell} julia
ts_plot(g, p.xmin, p.xmax, k0)
```

