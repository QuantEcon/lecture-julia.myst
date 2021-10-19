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

(mc)=
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

Let's start with by importing some basic packages:

```{code-cell} julia
using LinearAlgebra, Statistics
using Distributions, Plots, Random
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

Continuing in this way, and using our knowledge of {doc}`geometric series <geom_series>`, we find that, for any $t \geq 0$,

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

For example, the law of motion for the Solow growth model is

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