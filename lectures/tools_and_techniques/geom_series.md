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

# {index}`Geometric Series for Elementary Economics <single: Geometric Series for Elementary Economics>`

```{contents} Contents
:depth: 2
```

## Overview

The lecture describes important ideas in economics that use the mathematics of geometric series.

Among these are

- the Keynesian **multiplier**
- the money **multiplier** that prevails in fractional reserve banking
  systems
- interest rates and present values of streams of payouts from assets

(As we shall see below, the term **multiplier** comes down to meaning **sum of a convergent geometric series**)

These and other applications prove the truth of the wise crack that

```{epigraph}
"in economics, a little knowledge of geometric series goes a long way "
```

Below we'll use the following packages:

[//]: # Remember to check this packages that will be used. These are the standard ones that are usually called. 

[//]: # Do we want to use `Symbolics.jl` or `Sympy.jl`?

```{code-cell} julia
using LinearAlgebra, Statistics
using Distributions, Plots, Printf, QuantEcon, Random, Symbolics

```

## Key Formulas

To start, let $c$ be a real number that lies strictly between
$-1$ and $1$.

- We often write this as $c \in (-1,1)$.
- Here $(-1,1)$ denotes the collection of all real numbers that
  are strictly less than $1$ and strictly greater than $-1$.
- The symbol $\in$ means *in* or *belongs to the set after the symbol*.

We want to evaluate geometric series of two types -- infinite and finite.





