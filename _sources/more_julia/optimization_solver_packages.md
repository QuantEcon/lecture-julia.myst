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

# Optimization and Nonlinear Solvers

```{contents} Contents
:depth: 2
```

## Overview

In this lecture we introduce a few of the Julia libraries for solving optimization problems, systems of equations, and finding fixed points.

See {doc}`auto-differentiation <../more_julia/auto_differentiation>` for more on calculating gradients and Jacobians for these types of algorithms.


```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Statistics, BenchmarkTools
using ForwardDiff, Optim, Roots, NLsolve
using FixedPointAcceleration, NonlinearSolve
using Optimization, OptimizationOptimJL, ForwardDiff, Enzyme
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions
```

## Optimization

There are a large number of packages intended to be used for optimization in Julia.

Part of the reason for the diversity of options is that Julia makes it possible to efficiently implement a large number of variations on optimization routines.

The other reason is that different types of optimization problems require different algorithms.

### Optim.jl

A good pure-Julia solution for the (unconstrained or box-bounded) optimization of
univariate and multivariate functions is the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) package.

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
converged(result) ||
    error("Failed to converge in $(iterations(result)) iterations")
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

The default algorithm is `NelderMead`, which is derivative-free and hence requires many function evaluations.

To change the algorithm type to [L-BFGS](http://julianlsolvers.github.io/Optim.jl/stable/algo/lbfgs/)

```{code-cell} julia
results = optimize(f, x_iv, LBFGS())
println("minimum = $(results.minimum) with argmin = $(results.minimizer) in " *
        "$(results.iterations) iterations")
```

Note that this has fewer iterations.

As no derivative was given, it used [finite differences](https://en.wikipedia.org/wiki/Finite_difference) to approximate the gradient of `f(x)`.

However, since most of the algorithms require derivatives, you will often want to use automatic differentiation or pass analytical gradients if possible.

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

However, you will note that this did not converge, as stochastic methods typically require many more iterations as a tradeoff for their global convergence properties.

See the [maximum likelihood](http://julianlsolvers.github.io/Optim.jl/stable/examples/generated/maxlikenlm/)
example and the accompanying [Jupyter notebook](https://nbviewer.jupyter.org/github/JuliaNLSolvers/Optim.jl/blob/gh-pages/v0.15.3/examples/generated/maxlikenlm.ipynb).

### JuMP.jl

The [JuMP.jl](https://github.com/JuliaOpt/JuMP.jl) package is an ambitious implementation of a modelling language for optimization problems in Julia.

In that sense, it is more like an AMPL (or Pyomo) built on top of the Julia
language with macros, and able to use a variety of different commercial and open-source solvers.

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

## Optimization.jl Meta-Package
The [Optimization.jl](https://github.com/SciML/Optimization.jl) package provides a common interface to a variety of optimization packages in Julia.  As part of the [SciML](https://sciml.ai/) ecosystem, it is designed to work seamlessly with other SciML tools, and to provide differentiable optimization routines that can be used in conjunction with automatic differentiation packages.

Algorithms require loading additional packages for one of the [supported optimizers](https://docs.sciml.ai/Optimization/stable/#Overview-of-the-solver-packages-in-alphabetical-order), such as `OptimizationOptimJL.jl`, which wraps the `Optim.jl` package.

From the [documentation](https://docs.sciml.ai/Optimization/stable/getting_started/):

```{code-cell} julia
rosenbrock(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
u0 = zeros(2)
p = [1.0, 100.0]
prob = OptimizationProblem(rosenbrock, u0, p)
sol = solve(prob, NelderMead())
sol
```

The separation of the argument, `u`, and the parameters, `p`, is common in SciML and provides methods to cleanly handle parameterized problems.

Function wrappers also provide easy integration with automatic differentiation such as `ForwardDiff.jl`.

```{code-cell} julia
f_fd = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f_fd, u0, p)
sol = solve(prob, BFGS())
```

Or with `Enzyme.jl`, which has slower compilation times but provides another AD backend

```{code-cell} julia
f_enzyme = OptimizationFunction(rosenbrock, Optimization.AutoEnzyme())
prob = OptimizationProblem(f_enzyme, u0, p)
sol = solve(prob, BFGS())
```

Finally, [the documentation](https://docs.sciml.ai/Optimization/stable/) for a variety of other examples and features, such as constrained optimization.


## NonlinearSolve.jl Meta-Package
Within the [SciML](https://sciml.ai/) ecosystem, the [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) package provides a unified interface for solving nonlinear systems of equations. It builds on top of other SciML packages and offers advanced features such as automatic differentiation, various solver algorithms, and support for large-scale problems.  Furthermore, as with `Optimization.jl`, it has convenient integration with automatic differentiation (as well as implementing AD for the solver itself, with respect to parameters).

In general, we suggest using this meta-package where possible as it provides a well-maintained interface amenable to switching out solvers in the future. 

### Basic Examples

Here we adapt examples directly from the [documentation](https://docs.sciml.ai/NonlinearSolve/stable/).

First we can solve a system of equations as a closure over a constant `c`.

```{code-cell} julia
c = 2.0
f(u, p) = u .* u .- c # p ignored
u0 = [1.0, 1.0]
prob = NonlinearProblem(f, u0) # defaults to p = nothing
sol = solve(prob, NewtonRaphson())
```
In this case, the `p` argument (which can hold parameters) is ignored, but needs to be present in SciML problems.

The `NewtonRaphson()` method is a built-in solver, and not an external package.  We can access the results through `sol.u` and see other information such as the return code

```{code-cell} julia
@show sol.u
@show sol.retcode
sol.stats
```


We can see further details on the algorithm itself, and see that it uses `ForwardDiff.jl` by default since `NewtonRaphson()` requires a Jacobian.

```{code-cell} julia
sol.alg
```

You can see the performance of this algorithm in this context, which uses a large number of allocations relative to the simplicity of the problem.

```{code-cell} julia
@benchmark solve($prob, NewtonRaphson())
```

### Using Parameters and Inplace Functions
SciML interfaces prefer to cleanly separate a function argument and a parameter argument, which makes it easier to handle parameterized problems and ensure flexible code.

To use this, we rewrite our example above to use the previously ignored parameter `p`.

```{code-cell} julia
f(u, p) = u .* u .- p
u0 = [1.0, 1.0]
p = 2.0
prob = NonlinearProblem(f, u0, p) # pass the `p`
```

Note that the `prob` shows In-place: true.

Benchmarking this version.

```{code-cell} julia
@btime solve($prob, NewtonRaphson())
```

This may or may not have better performance than the previous version which used the closure.

Regardless, it will still have many allocations, which can be a significant fraction of the runtime for problems small or large.

To support this, the SciML ecosystem has a bias towards in-place functions.  Here we add another version of the function `f!(du, u, p)` to be in-place, modifying the first argument, which the solver detects.

```{code-cell} julia
function f!(du, u, p)
    du .= u .* u .- p
end
prob = NonlinearProblem(f!, u0, p)
@btime solve($prob, NewtonRaphson())
```

Note the decrease in allocations, and possibly time depending on your system.

### Patterns for Parameters

The `p` argument can be anything, including named tuples, vectors, or a single value as above.

A common pattern appropriate for more complicated code is to use a named tuple packing and unpacking.

```{code-cell} julia
function f_nt(u, p)
    (; c) = p # unpack any arguments in the named tuple
    return u .* u .- c
end

p_nt = (c = 2.0,) # named tuple
prob_nt = NonlinearProblem(f_nt, u0, p_nt)
@btime solve($prob_nt, NewtonRaphson())
```


### Small Systems
For small systems, the solver is able to use [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) which can further improve performance.  These fixed-size arrays require using the out-of-place formulation, but otherwise require no special handling.

```{code-cell} julia
using StaticArrays
f_SA(u, p) = SA[u[1] * u[1] - p, u[2] * u[2] - p] # rewrote without broadcasting
u0_static = SA[1.0, 1.0] # static array
prob = NonlinearProblem(f_SA, u0_static, p)
```

Note that the problem shows that this is `In-place: false` and operates on the `SVector{2, Float64}` type.

Next we can benchmark this version, where we can use a simpler non-allocating solver designed for small systems, `SimpleNewtonRaphson()`.  See [the docs](https://docs.sciml.ai/NonlinearSolve/stable/native/simplenonlinearsolve/) for more details on solvers optimized for non-allocating algorithms.

```{code-cell} julia
@btime solve($prob, SimpleNewtonRaphson())
```

Depending on your system, you may find this is 100 to 1000x faster than the original version using closures and allocations.

```{note}
In principle it should be equivalent to use our previous `f` instead of `f_SA`, but depending on package and compiler versions it may not achieve the full 0 allocations.  Regardless, try the generic function first before porting over to a new function.  The `SimpleNewtonRaphson()` function is also specialized to avoid any internal allocations for small systems, but you will even see some benefits with `NewtonRaphson()`.
```

### Using Other Solvers
While it has some pre-built solvers, as we used above, it is primarily intended as a meta-package.  To use one of the many [external solver packages](https://docs.sciml.ai/NonlinearSolve/stable/solvers/nonlinear_system_solvers/) you simply need to ensure the package is in your Project file and then include it.

For example, we can use the Anderson Acceleration in [FixedPointAcceleration.jl's wrapper](https://docs.sciml.ai/NonlinearSolve/stable/api/fixedpointacceleration/)

```{code-cell} julia
using FixedPointAcceleration
prob = NonlinearProblem(f, u0, p)
@btime solve($prob, FixedPointAccelerationJL())
@btime solve($prob, FixedPointAccelerationJL(; algorithm = :Simple))
```

The default algorithm in this case is `:Anderson`, but we can also use `:Simple` fixed-point iteration.

In this case, the fixed-point iteration algorithm is slower than the gradient-based Newton-Raphson, but shows the flexibility in cases where gradients are not available.

```{note}
While there is a [NLsolve.jl wrapper](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/), it seems to have a less robust implementation of Anderson than some of the other packages.
```

### Bracketing Solvers and Rootfinding Problems

The `NonlinearProblem` type is intended for algorithms which use an initial condition (and may or may not use derivatives).

For some one-dimensional problems it is more convenient to use one of the [bracketing methods](https://docs.sciml.ai/NonlinearSolve/stable/solvers/bracketing_solvers/#bracketing)

For example, from the [documentation](https://docs.sciml.ai/NonlinearSolve/stable/tutorials/getting_started/#Problem-Type-2:-Solving-Interval-Rootfinding-Problems-with-Bracketing-Methods),

```{code-cell} julia
f_bracketed(u, p) = u * u - p
uspan = (1.0, 2.0) # brackets
p = 2.0
prob = IntervalNonlinearProblem(f_bracketed, uspan, p)
sol = solve(prob)
```

In general, this should be preferred to using the `Roots.jl` package below, as it may provide a more consistent interface.


## Systems of Equations and Fixed Points

Julia has a variety of packages you can directly use for solving systems of equations and finding fixed points.  Many of these packages can also be used as backends for the `NonlinearSolve.jl` meta-package described above.


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
