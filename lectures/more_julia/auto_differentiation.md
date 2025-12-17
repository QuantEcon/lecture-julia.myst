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
using ForwardDiff, Enzyme, Test
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

- [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl), a relatively dependable but limited package.  Not really intended for standard ML-pipeline usage
- [Zygote.jl](https://github.com/FluxML/Zygote.jl), which is flexible but buggy and less reliable.  In a slow process of deprecation, but often the primary alternative.
- [Enzyme.jl](https://enzyme.mit.edu/julia/stable/), which is the most promising (and supports both forward and reverse mode).  However, as it works at a lower level of the compiler, it cannot support all Julia code.  In particular, it prefers in-place rather than "pure" functions.

## Introduction to Enzyme

[Enzyme.jl](https://enzyme.mit.edu/julia/stable/) is a high-performance automatic differentiation (AD) tool that operates at the LLVM level (compiler IR) rather than the source code level. This allows it to differentiate through low-level optimizations, mutation, and foreign function calls where other AD tools might fail.

```{note}
**Caution** : Enzyme.jl is under active development.  Some of the patterns shown here may change in future releases.  In practice, you may find using an LLM to be very valuable for navigating the perplexing error messages of Enzyme.jl.  Compilation times can be very slow, and performance intuition is not always straightforward.
```

However, with this power it makes usage a little more challenging as you need to ensure the compiler code generated by Julia conforms to certain patterns.

It supports both **Forward Mode** (best for $N \to \text{Many}$ derivatives) and **Reverse Mode** (best for gradients of scalar loss functions, $N \to 1$), and nested differentiation (e.g., Hessians).

While `ForwardDiff.jl` is often easier for simple problems, Enzyme is
capable of high-performance differentiation typically used in scientific computing and scientific machine learning (e.g., differentiable simulations and sensitivity analysis of differential equations).

### Comparison to the Python Ecosystem

Relative to JAX and PyTorch, Enzyme is often faster and more flexible for specialized algorithms in scientific computing (e.g., ODE solvers, physical simulations) because it differentiates at the LLVM level rather than the operation level. However, Enzyme itself is a low-level tool; it lacks the high-level layers for managing neural network state and batching found in PyTorch.

```{note}
One current advantage of Enzyme is that it has traditionally been difficult to write mutating code in JAX, which is essential in many scientific computing applications. The JAX ecosystem has been making progress on this limitation through mechanisms like "hijax" support in [JAX NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html). This mechanism allows users to write standard Python classes with **in-place mutation**, which the library then automatically transforms into JAX-compatible functional code during compilation. However, this remains experimental, whereas Enzyme handles mutation directly.
```



### Lux.jl: Neural Networks in Julia

While not designed for "bread-and-butter" deep learning pipelines out of the box, Enzyme can be paired with frameworks like [Lux.jl](https://github.com/LuxDL/Lux.jl) to handle deep learning tasks and implement neural networks.

Unlike Pytorch and the default behavior of JAX's [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html), the Lux.jl framework does not use implicit differentiable arguments within the neural network layers.

Instead, the differentiable parameters are explicitly passed and separated from non-differentiable arguments.  This is consistent with the split/merge pattern in [JAX NNX](https://flax.readthedocs.io/en/latest/guides/performance.html#functional-training-loop).

### Reactant.jl: The Bridge to TPUs and XLA
A recent addition to this ecosystem is [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl).

While Enzyme optimizes code for the CPU (and CUDA via LLVM), Reactant is designed to compile Julia code for high-performance accelerators like TPUs, as well as multi-node GPU clusters.

In particular, Reactant is a compiler frontend that lowers Julia code (and Enzyme gradients) into MLIR (Multi-Level Intermediate Representation) and StableHLO.  This targets the exact same compiler stack as JAX. In particular, JAX traces Python code to generate HLO/StableHLO, which is then compiled by XLA (Accelerated Linear Algebra). Reactant does the same for Julia and allows it to emit the same IR that JAX generates.

This allows Julia users to leverage the XLA compiler's massive optimizations for linear algebra and convolution, run natively on Google TPUs, and execute on large-scale distributed clusters, all while writing standard Julia code.  However, this is in early stages of development and should not be considered as a general replacement for standard Python ML frameworks and use-cases.

See [the Lux.jl documentation](https://github.com/LuxDL/Lux.jl?tab=readme-ov-file#reactant--enzyme) for an example combining Lux, Reactant, and Enzyme for a single Neural Network.

### Writing Enzyme-Differentiable Code

Enzyme-friendly code looks like ordinary Julia with a few discipline rules: mutate into preallocated buffers, avoid hidden allocations, and keep inputs type-stable.

In general, these are also good practices for achieving very high-performance Julia code in many cases, so there is no trade-off in making code high-performance vs. Enzyme-differentiable.  However, these tend to be more
advanced patterns than an introductory Julia user might be used to.

A common pattern is `f!(out, inputs..., cache)` where `cache` holds temporary work arrays passed last.

In many cases the biggest change is to use in-place linear algebra, many of which have corresponding highly optimized BLAS/LAPACK routines.  For example,
- `mul!(y, A, x)` implements the out-of-place math `y = A * x` without allocating;
- Use the 5-arg form `mul!(Y, A, B, α, β)` to compute `Y = α * A * B + β * Y` in-place (see the [mul! docs](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.mul!))
- Use `ldiv!(y, A, x)` for the in-place solve corresponding to `y = A \ x` (see [ldiv! docs](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.ldiv!))
- Use column-wise access or `@views` when slicing to avoid copies, and write into existing storage with `copy!` or `.=` rather than rebinding.
- Pass buffers explicitly to keep the function allocation-free; even a simple `cache = (; tmp = similar(x))` helps for temporary workspaces.

```{code-cell} julia
# in-place linear step with an explicit workspace
function step!(x_next, x, A, cache)
    @views mul!(cache.tmp, A, x) # cache.tmp = A * x
    @inbounds for i in eachindex(cache.tmp)
        # loops, are encouraged, but careful to avoid temp allocations
        cache.tmp[i] += 0.1 * i
    end
    copy!(x_next, cache.tmp) # x_next = cache.tmp
    return nothing
end

A = [0.9 0.1 0.0
     0.0 0.8 0.1
     0.0 0.0 0.7]
x = randn(3)
x_next = similar(x)
# Pass preallocated cache/buffers as individual arguments
# as named tuples, or as typesafe structs.
cache = (; tmp = zeros(3))

step!(x_next, x, A, cache)
x_next
```

Even without Enzyme, checking type-stability and allocations helps catch AD pain points early.

```{code-cell} julia
# warm-up to avoid counting compilation
step!(x_next, x, A, cache)

# Should have no type instability warnings
@show @inferred step!(x_next, x, A, cache)

# Should allocate zero bytes
@show @allocated step!(x_next, x, A, cache);
```

The same patterns apply to more complex routines: keep buffers explicit, avoid temporary slices, and rely on in-place linear algebra to minimize allocations that can break reverse-mode AD.


### Core Concepts & Terminology

To use Enzyme effectively, you must manually annotate your function arguments to tell the compiler how to handle them for a particular derivative.

### The Primal vs. The Shadow
* **Primal:** The values or variables used in the main computation (e.g., your input vector `x` or buffer `b`).
* **Shadow:** The separate memory location where derivatives are accumulated (e.g., `dx` or `db`).


### Fundamental Rules of Activity
When calling `autodiff`, every argument needs an "Activity" wrapper. To determine how you annotate them, consider:

1.  **Are you differentiating with respect to this argument?**
    * **No:** Use `Const(x)`. This tells Enzyme the value is constant and its derivative is zero.
    * **Yes (Scalar):** Use `Active(x)`. Enzyme will return the derivative directly.
    * **Yes (Array/Struct):** Use `Duplicated(x, dx)`. You must provide the shadow memory `dx`.

2.  **Is the argument mutated (written to) inside the function?**
    * **Yes, and I need to know the value:** Use `Duplicated(x, dx)`. Enzyme needs the shadow `dx` to store intermediate adjoints during the reverse pass.
    * **Yes, and I do not need to access the value:** You could still use `Duplicated(x, dx)` or `DuplicatedNoNeed(x, dx)` which may avoid some calculations required to calculate `x` if you are not using it directly.

This last point is important.  As your functions will be non-allocating, you need to ensure there is a "shadow" for any arguments that are used, even if they are just temporary buffers.

### Argument Wrappers (Quick Reference)

| Wrapper | Usage | Primal | Shadow | Docs |
| :--- | :--- | :--- | :--- | :--- |
| `Const(x)` | Constants / Config | Read-only | None | [Ref](https://enzyme.mit.edu/julia/stable/api/#EnzymeCore.Const) |
| `Active(x)` | Scalars (Float64) | Read-only | Returned | [Ref](https://enzyme.mit.edu/julia/stable/api/#EnzymeCore.Active) |
| `Duplicated(x, dx)` | Arrays / Mutated Buffers | Read/Write | **Explicit** | [Ref](https://enzyme.mit.edu/julia/stable/api/#EnzymeCore.Duplicated) |
| `DuplicatedNoNeed(x, dx)` | Mutated Buffers, ignoring `x` | Read/Write | **Explicit** | [Ref](https://enzyme.mit.edu/julia/stable/api/#EnzymeCore.DuplicatedNoNeed) |
| `BatchDuplicated` | Vectorized Derivatives | Read/Write | Tuple of Shadows | [Ref](https://enzyme.mit.edu/julia/stable/api/#EnzymeCore.BatchDuplicated) |

---

### Examples

### Convenience Functions
Enzyme provides [convenience functions](https://enzymead.github.io/Enzyme.jl/dev/#Convenience-functions-(gradient,-jacobian,-hessian)) to create zero-initialized shadows for you and to call `autodiff` with common patterns.

Following that documentation, we define a scalar valued function of a vector input.

When using Reverse-mode AD the forward pass needs to calculate the "primal" value, and we can request the function to return both.

```{code-cell} julia
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

# Use Reverse-mode  AD
x = [1.0, 2.0]
@show gradient(Reverse, rosenbrock, x)
# Return a tuple with the "primal" (i.e., function value) and grad
@show gradient(ReverseWithPrimal, rosenbrock, x);
```

With a preallocated gradient vector, we can use an in-place version
```{code-cell} julia
dx = Enzyme.make_zero(x) # or just zeros(size(x)), but this is more bulletproof
gradient!(Reverse, dx, rosenbrock, x)
@show dx;
```

Similarly, we can execute forward-mode AD to get the gradient.  Unlike Reverse mode, this will call the `autodiff` for each input dimension.

```{code-cell} julia
@show gradient(Forward, rosenbrock, [1.0, 2.0])
@show gradient(ForwardWithPrimal, rosenbrock, [1.0, 2.0]);
```

In the case of vector-valued functions, we can fill a Jacobian matrix.  If calling with `Forward`, each column of the Jacobian is filled in a separate pass.  If calling with `Reverse`, each row is filled in a separate pass.

```{code-cell} julia
f(x) = [(1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2;
        x[1] * x[2]]
@show jacobian(Forward, f, x)
@show jacobian(Reverse, f, x)
@show jacobian(ReverseWithPrimal, f, x);
```

See [here](https://enzymead.github.io/Enzyme.jl/dev/#Hessian-Vector-Product-Convenience-functions) for more examples such as Hessian-vector products.

### Simple Forward Mode
Forward mode propagates derivatives alongside the primal calculation. It is ideal for low-dimensional inputs.

```{code-cell} julia
using Enzyme, LinearAlgebra

f(x, y) = sum(x .* y)

x = [1.0, 2.0, 3.0]
y = [0.5, 1.0, 1.5]

# We want ∂f/∂x (holding y constant).
# 1. Create Shadow for x (seed with 1.0s to get full gradient)
dx = ones(size(x))

# 2. Call autodiff with respect to x
# Note: y is Const, x is Duplicated (Array)
autodiff(Forward, f, Duplicated(x, dx), Const(y))

# Result: Returns nothing (void), but the function executed.
print("∂f/∂x = ", dx)
```
Note that in Forward mode for scalar outputs, you often look at the return value.  However, for array inputs, Enzyme conventions usually focus on Reverse mode.


### Reverse Mode
Reverse mode computes the gradient of the output with respect to *all* inputs in a single pass.

```{code-cell} julia
# Define function
calc(x, y) = sum(x .^ 2 .+ y)

x = [1.0, 2.0, 3.0]
y = [0.5, 0.5, 0.5]

# 1. Initialize Shadows with Zeros
dx = Enzyme.make_zero(x) # ∂f/∂x will accumulate here
dy = Enzyme.make_zero(y) # ∂f/∂y will accumulate here

# 2. Call autodiff
# In this case we find the gradient for all arguments
autodiff(Reverse, calc, Duplicated(x, dx), Duplicated(y, dy))

@show dx  # Should be 2*x -> [2.0, 4.0, 6.0]
@show dy;  # Should be 1.0 -> [1.0, 1.0, 1.0]
```

As it requires calculation of the primal in the forward pass, as with the convenience functions you can request it to be returned, and then examine the shadows.

```{code-cell} julia
dx = Enzyme.make_zero(x)
dy = Enzyme.make_zero(y)
autodiff(ReverseWithPrimal, calc, Duplicated(x, dx), Duplicated(y, dy))
```

### Handling Mutation and Buffers
This is the most common pitfall. If a function modifies an argument (like `out` or a workspace buffer), both the Primal and Shadow must be valid.

```{code-cell} julia
function axpy!(y, A, x)
    mul!(y, A, x)
    return nothing
end

# 2. Loss calls the mutating function, but returns a scalar (the loss).
function compute_loss!(y, A, x)
    axpy!(y, A, x)
    return sum(y) # We want the gradient of THIS scalar
end

# Setup Data
A = [2.0 0.0; 0.0 3.0]
x = [1.0, 1.0]
y = zeros(2)

# Setup Shadows
# Enzyme will calculate ∂(sum)/∂y and populate dy for us.
dx = Enzyme.make_zero(x)
dA = Enzyme.make_zero(A)
dy = Enzyme.make_zero(y)

# 3. Calculate Gradient
#    We differentiate 'compute_loss!'.
#    Since it returns a scalar (Active), Enzyme automatically seeds it with 1.0.
autodiff(Reverse, compute_loss!,
         Duplicated(y, dy),   # Intermediate buffer (Enzyme handles the backprop!)
         Duplicated(A, dA),   # Parameter we want gradient for
         Duplicated(x, dx))

@show dx   # ∂(sum(y))/∂x
@show dA;  # ∂(sum(y))/∂A
```

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
