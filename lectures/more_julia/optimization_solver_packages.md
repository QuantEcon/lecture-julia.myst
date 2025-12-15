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

(optimization_solver_packages)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Optimization and Solving Systems of Equations

```{contents} Contents
:depth: 2
```

## Overview

In this lecture we introduce a few of the Julia libraries for solving optimization problems, systems of equations, and finding fixed-points.

See {doc}`auto-differentiation <../more_julia/auto_differentiation>` for more on calculating gradients and Jacobians for these types of algorithms.


```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Statistics
using ForwardDiff, Optim, Roots, NLsolve
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions
```

## Optimization

There are a large number of packages intended to be used for optimization in Julia.

Part of the reason for the diversity of options is that Julia makes it possible to efficiently implement a large number of variations on optimization routines.

The other reason is that different types of optimization problems require different algorithms.

### Optim.jl

A good pure-Julia solution for the (unconstrained or box-bounded) optimization of
univariate and multivariate function is the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) package.

By default, the algorithms in `Optim.jl` target minimization rather than
maximization, so if a function is called `optimize` it will mean minimization.

#### Univariate Functions on Bounded Intervals

[Univariate optimization](http://julianlsolvers.github.io/Optim.jl/stable/user/minimization/#minimizing-a-univariate-function-on-a-bounded-interval)
defaults to a robust hybrid optimization routine called [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method).

```{code-cell} julia
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions

result = optimize(x -> x^2, -2.0, 1.0)
```

Always check if the results converged, and throw errors otherwise

```{code-cell} julia
converged(result) || error("Failed to converge in $(iterations(result)) iterations")
xmin = result.minimizer
result.minimum
```

The first line is a logical OR between `converged(result)` and `error("...")`.

If the convergence check passes, the logical sentence is true, and it will proceed to the next line; if not, it will throw the error.

#### Unconstrained Multivariate Optimization

There are a variety of [algorithms and options](http://julianlsolvers.github.io/Optim.jl/stable/user/minimization/#_top) for multivariate optimization.

From the documentation, the simplest version is

```{code-cell} julia
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x_iv = [0.0, 0.0]
results = optimize(f, x_iv) # i.e. optimize(f, x_iv, NelderMead())
```

The default algorithm in `NelderMead`, which is derivative-free and hence requires many function evaluations.

To change the algorithm type to [L-BFGS](http://julianlsolvers.github.io/Optim.jl/stable/algo/lbfgs/)

```{code-cell} julia
results = optimize(f, x_iv, LBFGS())
println("minimum = $(results.minimum) with argmin = $(results.minimizer) in " *
        "$(results.iterations) iterations")
```

Note that this has fewer iterations.

As no derivative was given, it used [finite differences](https://en.wikipedia.org/wiki/Finite_difference) to approximate the gradient of `f(x)`.

However, since most of the algorithms require derivatives, you will often want to use auto differentiation or pass analytical gradients if possible.

```{code-cell} julia
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x_iv = [0.0, 0.0]
results = optimize(f, x_iv, LBFGS(), autodiff = :forward) # i.e. use ForwardDiff.jl
println("minimum = $(results.minimum) with argmin = $(results.minimizer) in " *
        "$(results.iterations) iterations")
```

Note that we did not need to use `ForwardDiff.jl` directly, as long as our `f(x)` function was written to be generic (see the {doc}`generic programming lecture <../more_julia/generic_programming>` ).

Alternatively, with an analytical gradient

```{code-cell} julia
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x_iv = [0.0, 0.0]
function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

results = optimize(f, g!, x_iv, LBFGS()) # or ConjugateGradient()
println("minimum = $(results.minimum) with argmin = $(results.minimizer) in " *
        "$(results.iterations) iterations")
```

For derivative-free methods, you can change the algorithm -- and have no need to provide a gradient

```{code-cell} julia
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x_iv = [0.0, 0.0]
results = optimize(f, x_iv, SimulatedAnnealing()) # or ParticleSwarm() or NelderMead()
```

However, you will note that this did not converge, as stochastic methods typically require many more iterations as a tradeoff for their global-convergence properties.

See the [maximum likelihood](http://julianlsolvers.github.io/Optim.jl/stable/examples/generated/maxlikenlm/)
example and the accompanying [Jupyter notebook](https://nbviewer.jupyter.org/github/JuliaNLSolvers/Optim.jl/blob/gh-pages/v0.15.3/examples/generated/maxlikenlm.ipynb).

### JuMP.jl

The [JuMP.jl](https://github.com/JuliaOpt/JuMP.jl) package is an ambitious implementation of a modelling language for optimization problems in Julia.

In that sense, it is more like an AMPL (or Pyomo) built on top of the Julia
language with macros, and able to use a variety of different commerical and open source solvers.

If you have a linear, quadratic, conic, mixed-integer linear, etc. problem then this will likely be the ideal "meta-package" for calling various solvers.

For nonlinear problems, the modelling language may make things difficult for complicated functions (as it is not designed to be used as a general-purpose nonlinear optimizer).

See the [quick start guide](http://www.juliaopt.org/JuMP.jl/0.18/quickstart.html) for more details on all of the options.

The following is an example of calling a linear objective with a nonlinear constraint (provided by an external function).

Here `Ipopt` stands for `Interior Point OPTimizer`, a [nonlinear solver](https://github.com/JuliaOpt/Ipopt.jl) in Julia

```{code-block} julia
using JuMP, Ipopt
# solve
# max( x[1] + x[2] )
# st sqrt(x[1]^2 + x[2]^2) <= 1

function squareroot(x) # pretending we don't know sqrt()
    z = x # Initial starting point for Newton’s method
    while abs(z * z - x) > 1e-13
        z = z - (z * z - x) / (2z)
    end
    return z
end
m = Model(Ipopt.Optimizer)
# need to register user defined functions for AD
JuMP.register(m, :squareroot, 1, squareroot, autodiff = true)

@variable(m, x[1:2], start=0.5) # start is the initial condition
@objective(m, Max, sum(x))
@NLconstraint(m, squareroot(x[1]^2 + x[2]^2)<=1)
@show JuMP.optimize!(m)
```

And this is an example of a quadratic objective

```{code-block} julia
# solve
# min (1-x)^2 + (100(y-x^2)^2)
# st x + y >= 10

using JuMP, Ipopt
m = Model(Ipopt.Optimizer) # settings for the solver
@variable(m, x, start=0.0)
@variable(m, y, start=0.0)

@NLobjective(m, Min, (1 - x)^2+100(y - x^2)^2)

JuMP.optimize!(m)
println("x = ", value(x), " y = ", value(y))

# adding a (linear) constraint
@constraint(m, x + y==10)
JuMP.optimize!(m)
println("x = ", value(x), " y = ", value(y))
```

### BlackBoxOptim.jl

Another package for doing global optimization without derivatives is [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl).


An example for [parallel execution](https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/examples/rosenbrock_parallel.jl) of the objective is provided.

## Systems of Equations and Least Squares

### Roots.jl

A root of a real function $f$ on $[a,b]$ is an $x \in [a, b]$ such that $f(x)=0$.

For example, if we plot the function

```{math}
:label: root_f

f(x) = \sin(4 (x - 1/4)) + x + x^{20} - 1
```

with $x \in [0,1]$ we get

```{figure} /_static/figures/sine-screenshot-2.png

```

The unique root is approximately 0.408.

The [Roots.jl](https://github.com/JuliaLang/Roots.jl) package offers `fzero()` to find roots

```{code-cell} julia
using Roots
f(x) = sin(4 * (x - 1 / 4)) + x + x^20 - 1
fzero(f, 0, 1)
```

### NLsolve.jl

The [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl/) package provides functions to solve for multivariate systems of equations and fixed points.

From the documentation, to solve for a system of equations without providing a Jacobian

```{code-cell} julia
using NLsolve

f(x) = [(x[1] + 3) * (x[2]^3 - 7) + 18
        sin(x[2] * exp(x[1]) - 1)] # returns an array

results = nlsolve(f, [0.1; 1.2])
```

In the above case, the algorithm used finite differences to calculate the Jacobian.

Alternatively, if `f(x)` is written generically, you can use auto-differentiation with a single setting.

```{code-cell} julia
results = nlsolve(f, [0.1; 1.2], autodiff = :forward)

println("converged=$(NLsolve.converged(results)) at root=$(results.zero) in " *
        "$(results.iterations) iterations and $(results.f_calls) function calls")
```

Providing a function which operates inplace (i.e., modifies an argument) may help performance for large systems of equations (and hurt it for small ones).

```{code-cell} julia
function f!(F, x) # modifies the first argument
    F[1] = (x[1] + 3) * (x[2]^3 - 7) + 18
    F[2] = sin(x[2] * exp(x[1]) - 1)
end

results = nlsolve(f!, [0.1; 1.2], autodiff = :forward)

println("converged=$(NLsolve.converged(results)) at root=$(results.zero) in " *
        "$(results.iterations) iterations and $(results.f_calls) function calls")
```

## LeastSquaresOptim.jl

Many optimization problems can be solved using linear or nonlinear least squares.

Let $x \in R^N$ and $F(x) : R^N \to R^M$ with $M \geq N$, then the nonlinear least squares problem is

$$
\min_x F(x)^T F(x)
$$

While $F(x)^T F(x) \to R$, and hence this problem could technically use any nonlinear optimizer, it is useful to exploit the structure of the problem.

In particular, the Jacobian of $F(x)$, can be used to approximate the Hessian of the objective.

As with most nonlinear optimization problems, the benefits will typically become evident only when analytical or automatic differentiation is possible.

If $M = N$ and we know a root $F(x^*) = 0$ to the system of equations exists, then NLS is the defacto method for solving large **systems of equations**.

An implementation of NLS is given in [LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl).
