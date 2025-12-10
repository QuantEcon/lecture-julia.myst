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

(auto_differentiation)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Automatic Differentiation

```{contents} Contents
:depth: 2
```

## Overview

In this lecture we will discuss auto-differentiation in Julia, and introduce some key packages.



```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Statistics
using ForwardDiff
```

## Introduction to Differentiable Programming

The promise of differentiable programming is that we can move towards taking the derivatives of almost arbitrarily
complicated computer programs, rather than simply thinking about the derivatives of mathematical functions.  Differentiable
programming is the natural evolution of automatic differentiation (AD, sometimes called algorithmic differentiation).

Stepping back, there are three ways to calculate the gradient or Jacobian

* Analytic derivatives / Symbolic differentiation
    * You can sometimes calculate the derivative on pen-and-paper, and potentially simplify the expression.
    * In effect, repeated applications of the chain rule, product rule, etc.
    * It is sometimes, though not always, the most accurate and fastest option if there are algebraic simplifications.
    * Sometimes symbolic integration on the computer a good solution, if the package can handle your functions. Doing algebra by hand is tedious and error-prone, but
      is sometimes invaluable.
* Finite differences
    * Evaluate the function at least $N+1$ times to get the gradient -- Jacobians are even worse.
    * Large $\Delta$ is numerically stable but inaccurate, too small of $\Delta$ is numerically unstable but more accurate.
    * Choosing the $\Delta$ is hard, so use packages such as [DiffEqDiffTools.jl](https://github.com/JuliaDiffEq/DiffEqDiffTools.jl).
    * If a function is $R^N \to R$ for a large $N$, this requires $O(N)$ function evaluations.

$$
\partial_{x_i}f(x_1,\ldots x_N) \approx \frac{f(x_1,\ldots x_i + \Delta,\ldots x_N) - f(x_1,\ldots x_i,\ldots x_N)}{\Delta}
$$

* Automatic Differentiation
    * The same as analytic/symbolic differentiation, but where the **chain rule** is calculated **numerically** rather than symbolically.
    * Just as with analytic derivatives, can establish rules for the derivatives of individual functions (e.g. $d\left(sin(x)\right)$ to $cos(x) dx$) for intrinsic derivatives.

AD has two basic approaches, which are variations on the order of evaluating the chain rule: reverse and forward mode (although mixed mode is possible).

1. If a function is $R^N \to R$, then **reverse-mode** AD can find the gradient in $O(1)$ sweep (where a "sweep" is $O(1)$ function evaluations).
1. If a function is $R \to R^N$, then **forward-mode** AD can find the jacobian in $O(1)$ sweeps.

We will explore two types of automatic differentiation in Julia (and discuss a few packages which implement them).  For both, remember the [chain rule](https://en.wikipedia.org/wiki/Chain_rule)

$$
\frac{dy}{dx} = \frac{dy}{dw} \cdot \frac{dw}{dx}
$$

Forward-mode starts the calculation from the left with $\frac{dy}{dw}$ first, which then calculates the product with $\frac{dw}{dx}$.  On the other hand, reverse mode starts on the right hand side with $\frac{dw}{dx}$ and works backwards.

Take an example a function with fundamental operations and known analytical derivatives

$$
f(x_1, x_2) = x_1 x_2 + \sin(x_1)
$$

And rewrite this as a function which contains a sequence of simple operations and temporaries.

```{code-cell} julia
function f(x_1, x_2)
    w_1 = x_1
    w_2 = x_2
    w_3 = w_1 * w_2
    w_4 = sin(w_1)
    w_5 = w_3 + w_4
    return w_5
end
```

Here we can identify all of the underlying functions (`*, sin, +`), and see if each has an
intrinsic derivative.  While these are obvious, with Julia we could come up with all sorts of differentiation rules for arbitrarily
complicated combinations and compositions of intrinsic operations.  In fact, there is even [a package](https://github.com/JuliaDiff/ChainRules.jl) for registering more.

### Forward-Mode Automatic Differentiation

In forward-mode AD, you first fix the variable you are interested in (called "seeding"), and then evaluate the chain rule in left-to-right order.

For example, with our $f(x_1, x_2)$ example above, if we wanted to calculate the derivative with respect to $x_1$ then
we can seed the setup accordingly.  $\frac{\partial  w_1}{\partial  x_1} = 1$ since we are taking the derivative of it, while $\frac{\partial  w_2}{\partial  x_1} = 0$.

Following through with these, redo all of the calculations for the derivative in parallel with the function itself.

$$
\begin{array}{l|l}
f(x_1, x_2) &
\frac{\partial f(x_1,x_2)}{\partial x_1}
\\
\hline
w_1 = x_1 &
\frac{\partial  w_1}{\partial  x_1} = 1 \text{ (seed)}\\
w_2 = x_2 &
\frac{\partial   w_2}{\partial  x_1} = 0 \text{ (seed)}
\\
w_3 = w_1 \cdot w_2 &
\frac{\partial  w_3}{\partial x_1} = w_2 \cdot \frac{\partial   w_1}{\partial  x_1} + w_1 \cdot \frac{\partial   w_2}{\partial  x_1}
\\
w_4 = \sin w_1 &
\frac{\partial   w_4}{\partial x_1} = \cos w_1 \cdot \frac{\partial  w_1}{\partial x_1}
\\
w_5 = w_3 + w_4 &
\frac{\partial  w_5}{\partial x_1} = \frac{\partial  w_3}{\partial x_1} + \frac{\partial  w_4}{\partial x_1}
\end{array}
$$

Since these two could be done at the same time, we say there is "one pass" required for this calculation.

Generalizing a little, if the function was vector-valued, then that single pass would get the entire row of the Jacobian in that single pass.  Hence for a $R^N \to R^M$ function, requires $N$ passes to get a dense Jacobian using forward-mode AD.

How can you implement forward-mode AD?  It turns out to be fairly easy with a generic programming language to make a simple example (while the devil is in the details for
a high-performance implementation).

### Forward-Mode with Dual Numbers

One way to implement forward-mode AD is to use [dual numbers](https://en.wikipedia.org/wiki/Dual_number).

Instead of working with just a real number, e.g. $x$, we will augment each with an infinitesimal $\epsilon$ and use $x + \epsilon$.

From Taylor's theorem,

$$
f(x + \epsilon) = f(x) + f'(x)\epsilon + O(\epsilon^2)
$$

where we will define the infinitesimal such that $\epsilon^2 = 0$.

With this definition, we can write a general rule for differentiation of $g(x,y)$ as the chain rule for the total derivative

$$
g(x + \epsilon, y + \epsilon) = g(x, y) + (\partial_x g(x,y) + \partial_y g(x,y))\epsilon
$$

But, note that if we keep track of the constant in front of the $\epsilon$ terms (e.g. a $x'$ and $y'$)

$$
g(x + x'\epsilon, y + y'\epsilon) = g(x, y) + (\partial_x g(x,y)x' + \partial_y g(x,y)y')\epsilon
$$

This is simply the chain rule.  A few more examples

$$
\begin{aligned}
        (x + x'\epsilon) + (y + y'\epsilon) &= (x + y) + (x' + y')\epsilon\\
(x + x'\epsilon)\times(y + y'\epsilon) &= (xy) + (x'y + y'x)\epsilon\\
\exp(x + x'\epsilon) &= \exp(x) + (x'\exp(x))\epsilon\\
        \end{aligned}
$$

Using the generic programming in Julia, it is easy to define a new dual number type which can encapsulate the pair $(x, x')$ and provide a definitions for
all of the basic operations.  Each definition then has the chain-rule built into it.

With this approach, the "seed" process is simple the creation of the $\epsilon$ for the underlying variable.

So if we have the function $f(x_1, x_2)$ and we wanted to find the derivative $\partial_{x_1} f(3.8, 6.9)$ then then we would seed them with the dual numbers $x_1 \to (3.8, 1)$ and $x_2 \to (6.9, 0)$.

If you then follow all of the same scalar operations above with a seeded dual number, it will calculate both the function value and the derivative in a single "sweep" and without modifying any of your (generic) code.

### ForwardDiff.jl

Dual-numbers are at the heart of one of the AD packages we have already seen.

```{code-cell} julia
h(x) = sin(x[1]) + x[1] * x[2] + sinh(x[1] * x[2]) # multivariate.
x = [1.4 2.2]
@show ForwardDiff.gradient(h, x) # use AD, seeds from x

#Or, can use complicated functions of many variables
f(x) = sum(sin, x) + prod(tan, x) * sum(sqrt, x)
g = (x) -> ForwardDiff.gradient(f, x); # g() is now the gradient
g(rand(5)) # gradient at a random point
# ForwardDiff.hessian(f,x') # or the hessian
```

We can even auto-differentiate complicated functions with embedded iterations.

```{code-cell} julia
function squareroot(x) #pretending we don't know sqrt()
    z = copy(x) # Initial starting point for Newton’s method
    while abs(z * z - x) > 1e-13
        z = z - (z * z - x) / (2z)
    end
    return z
end
squareroot(2.0)
```

```{code-cell} julia
dsqrt(x) = ForwardDiff.derivative(squareroot, x)
dsqrt(2.0)
```

### Reverse-Mode AD

Unlike forward-mode auto-differentiation, reverse-mode is very difficult to implement efficiently, and there are many variations on the best approach.

Many reverse-mode packages are connected to machine-learning packages, since the efficient gradients of $R^N \to R$ loss functions are necessary for the gradient descent optimization algorithms used in machine learning.

At this point, Julia does not have a single consistently usable reverse-mode AD package without rough edges, but a few key ones to consider are:

- [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl), a relatively dependable but limited package.  Not really intended for standard ML-pipline usage
- [Zygote.jl](https://github.com/FluxML/Zygote.jl), which is flexible but buggy and less reliable.  In a slow process of deprecation, but often the primary alternative.
- [Enzyme.jl](https://enzyme.mit.edu/julia/stable/), which is the most promising (and supports both forward and reverse mode).  However, as it works at a lower level of the compiler, it cannot support all Julia code.  In particular, it prefers in-place rather than "pure" functions.

## Exercises

### Exercise 1

Doing a simple implementation of forward-mode auto-differentiation is very easy in Julia since it is generic.  In this exercise, you
will fill in a few of the operations required for a simple AD implementation.

First, we need to provide a type to hold the dual.

```{code-cell} julia
struct DualNumber{T} <: Real
    val::T
    ϵ::T
end
```

Here we have made it a subtype of `Real` so that it can pass through functions expecting Reals.

We can add on a variety of chain rule definitions by importing in the appropriate functions and adding DualNumber versions.  For example

```{code-cell} julia
import Base: +, *, -, ^, exp
+(x::DualNumber, y::DualNumber) = DualNumber(x.val + y.val, x.ϵ + y.ϵ)  # dual addition
+(x::DualNumber, a::Number) = DualNumber(x.val + a, x.ϵ)  # i.e. scalar addition, not dual
+(a::Number, x::DualNumber) = DualNumber(x.val + a, x.ϵ)  # i.e. scalar addition, not dual
```

With that, we can seed a dual number and find simple derivatives,

```{code-cell} julia
f(x, y) = 3.0 + x + y

x = DualNumber(2.0, 1.0)  # x -> 2.0 + 1.0\epsilon
y = DualNumber(3.0, 0.0)  # i.e. y = 3.0, no derivative

# seeded calculates both the function and the d/dx gradient!
f(x, y)
```

For this assignment:

1. Add in AD rules for the other operations: `*, -, ^, exp`.
1. Come up with some examples of univariate and multivariate functions combining those operations and use your AD implementation to find the derivatives.

