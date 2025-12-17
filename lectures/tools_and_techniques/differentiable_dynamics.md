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

(lssm)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Differentiating Models of Economic Dynamics

```{index} single: Differentiating Models of Economic Dynamics
```

```{contents} Contents
:depth: 2
```


## Overview


This lecture provides an introduction to differentiable simulation of dynamic systems in Julia using Enzyme.jl.  See {doc}`auto-differentiation <../more_julia/auto_differentiation>` for background on in-place patterns and Enzyme wrappers.

It builds on the **linear state space** models we introduced in {doc}`linear models <../introduction_dynamics/linear_models>` and the {doc}`kalman filter <../introduction_dynamics/kalman>`. 


Example applications of these methods include:

* calibration
* simulated method of moments
* estimation


**Caution** : The code in this section is significantly more advanced than some of the other lectures, and requires some experience with both auto-differentiation concepts and a more detailed understanding of type-safety and memory management in Julia.

[Enzyme.jl](https://enzyme.mit.edu/julia/stable/) is under active development and while state-of-the-art, it is often bleeding-edge. Some of the patterns shown here may change in future releases. See {doc}`auto-differentiation <../more_julia/auto_differentiation>` for the latest best practices.

In practice, you may find using an LLM valuable for navigating the perplexing error messages of Enzyme.jl. Compilation times can be very slow, and performance intuition is not always straightforward.


```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Random, Plots, Test, Enzyme, Statistics
using Optimization, OptimizationOptimJL, EnzymeTestUtils
```

(simm_lss)=
## Simulating a Linear State Space Model

Take the following parameterization of a linear state space model

$$
x_{t+1} = A x_t + C w_{t+1}, \qquad
y_t = G x_t + H v_t
$$

where $w_{t+1}$ and $v_t$ are i.i.d. standard normal shocks. States $x_t \in \mathbb R^N$ and observations $y_t \in \mathbb R^M$ are stored column-wise.

See the {doc}`auto-differentiation <../more_julia/auto_differentiation>`  lecture for more on efficient in-place operations and `mul!`.

```{code-cell} julia
function simulate_lss!(x, y, model, x_0, w, v)
    (; A, C, G, H) = model
    N, T1 = size(x)
    M, T1y = size(y)
    T = size(w, 2)

    @assert T1 == T + 1 == T1y
    @assert size(v, 2) == T + 1
    @assert size(A) == (N, N)
    @assert size(G) == (M, N)
    @assert size(C, 1) == N
    @assert size(H, 1) == M
    @assert length(x_0) == N

    # Enzyme has challenges with activity analysis on broadcasting assignments
    @inbounds for i in 1:N
        x[i, 1] = x_0[i]
    end
    # Apply evolution and observation equations
    @inbounds for t in 1:T
        @views mul!(x[:, t + 1], A, x[:, t])             # x_{t+1} = A x_t
        @views mul!(x[:, t + 1], C, w[:, t], 1.0, 1.0)   # + C w_{t+1}

        @views mul!(y[:, t], G, x[:, t])                 # y_t = G x_t
        @views mul!(y[:, t], H, v[:, t], 1.0, 1.0)       # + H v_t
    end
    # Apply observation equation at T+1
    @views mul!(y[:, T + 1], G, x[:, T + 1])
    @views mul!(y[:, T + 1], H, v[:, T + 1], 1.0, 1.0)

    return nothing
end
```

Crucially, this function modifies the preallocated `x` and `y` arrays in place without any allocations.

We can use this function to simulate from example matrices and a sequence of shocks

```{code-cell} julia
Random.seed!(1234)

N, M, K, L = 3, 2, 2, 2
T = 10

A = [0.8 0.1 0.0
     0.0 0.7 0.1
     0.0 0.0 0.6]
C = 0.1 .* randn(N, K)
G = [1.0 0.0 0.0
     0.0 1.0 0.3]
H = 0.05 .* randn(M, L)
model = (; A, C, G, H)

x_0 = randn(N)
w = randn(K, T)
v = randn(L, T + 1)

x = zeros(N, T + 1)
y = zeros(M, T + 1)

simulate_lss!(x, y, model, x_0, w, v)

time = 0:T
plot(time, x', lw = 2, xlabel = "t", ylabel = "state", label = ["x1" "x2" "x3"],
     title = "State Paths")
```

```{code-cell} julia
plot(time, y', lw = 2, xlabel = "t", ylabel = "observation",
     label = ["y1" "y2"], title = "Observation Paths")
```

(simm_lss_diff)=
### Differentiating the Simulation

Forward-mode in Enzyme is convenient for impulse-style effects: for example, here we perturb only the $w_1$ leaving everything else fixed and can see the change in the $x$ and $y$ paths

```{code-cell} julia
# forward-mode on w[1]
x = zeros(N, T + 1)
y = zeros(M, T + 1)
dx = Enzyme.make_zero(x)
dy = Enzyme.make_zero(y)
dw = Enzyme.make_zero(w)
dw[1] = 1.0                         # unit perturbation to first shock

autodiff(Forward,
         simulate_lss!,
         Duplicated(x, dx),
         Duplicated(y, dy),
         Const(model), # leaving model fixed
         Const(x_0), # leaving initial state fixed
         Duplicated(w, dw), # perturbing w
         Const(v))

dx[:, 1:3]   # early-state sensitivities (impulse response flavor)
```

Batch tangents let us reuse one primal evaluation while seeding multiple partials. Below we differentiate with respect to two entries of $A$ in one call; Enzyme accumulates the tangents into separate shadow arrays.

```{code-cell} julia
dx_batch = (Enzyme.make_zero(x), Enzyme.make_zero(x))
dy_batch = (Enzyme.make_zero(y), Enzyme.make_zero(y))
dmodels = (Enzyme.make_zero(model), Enzyme.make_zero(model))
dmodels[1].A[1] = 1.0
dmodels[2].A[2] = 1.0

autodiff(Forward,
         simulate_lss!,
         BatchDuplicated(x, dx_batch), # batch duplicated to match dmodels
         BatchDuplicated(y, dy_batch),
         BatchDuplicated(model, dmodels),
         Const(x_0),
         Const(w),
         Const(v))
@show dy_batch[1], dy_batch[2];
```


### Checking Type Stability and Allocations

With complicated code, we first need to ensure the code is type-stable.  The following call is silent, which indicates there are no type-stability issues.

```{code-cell} julia
@inferred simulate_lss!(x, y, model, x_0, w, v)
```

Next we check that it does not allocate any memory during execution.

```{code-cell} julia
function count_allocs()
    return @allocated simulate_lss!(x, y, model, x_0, w, v)
end
@test count_allocs() == 0
```

Note that `@allocated` can be misleading if used in a Jupyter notebook and must be wrapped in a function to get reliable results, or used with `@btime` from `BenchmarkTools.jl`.

Finally, for complicated functions such as simulations, we cannot
assume that Enzyme (or any AD system) will necessarily be correct.

To aid in this, the `EnzymeTestUtils` provides utilities for this purpose which automatically check against finite-difference approximations using the appropriate seeding.

When using in-place functions mutating the arguments `test_forward` requires that you pass any mutated arguments as output of the function itself, which can be done by a small wrapper.

```{code-cell} julia
function test_forward_simulate_lss!(x, y, model, x_0, w, v)
    simulate_lss!(x, y, model, x_0, w, v)
    return x, y
end
test_forward(
    test_forward_simulate_lss!,
    Duplicated,
    (x, Duplicated),
    (y, Duplicated),
    (model, Const),
    (x_0, Duplicated),
    (w, Const),
    (v, Const) 
)
```

Unlike `test_forward`, the automatic checks on reverse-mode AD with `test_reverse` require a scalar output, which we discuss below in the calibration section.

### Differentiating Functions of Simulation

Often we care about scalars of the simulated paths. For example, the average of the first observable is

$$
g(w, v, \theta) = \frac{1}{T+1} \sum_{t=0}^T y_{1,t}.
$$

Reverse-mode gives the gradient with respect to all shocks in one sweep while holding parameters fixed.

```{code-cell} julia
function mean_first_observation(y)
    return mean(@view y[1, :]) # view to avoid allocation
end

function g(x, y, model, x_0, w, v)
    simulate_lss!(x, y, model, x_0, w, v)
    return mean_first_observation(y)
end

x_rev = zeros(N, T + 1)
y_rev = zeros(M, T + 1)
dx_rev = Enzyme.make_zero(x_rev)
dy_rev = Enzyme.make_zero(y_rev)
dw_rev = Enzyme.make_zero(w)
dv_rev = Enzyme.make_zero(v)

autodiff(Reverse,
         g,
         Duplicated(x_rev, dx_rev), # output/buffer
         Duplicated(y_rev, dy_rev), # output/buffer
         Const(model),
         Const(x_0),
         Duplicated(w, dw_rev),   # active shocks
         Duplicated(v, dv_rev))

@show g(x, y, model, x_0, w, v)

dw_rev # sensitivity wrt evolution shock
```

These examples mirror the larger workflow: write allocation-free, in-place simulations, seed tangents with `Duplicated`, and use forward or reverse mode depending on whether you want many outputs per input (forward) or many inputs to one scalar (reverse).

As before, for complicated functions we may want to check that the gradients are correct using finite differences.

```{code-cell} julia
test_reverse(g,
    Active,
    (x_rev, Duplicated),
    (y_rev, Duplicated),
    (model, Const),
    (x_0, Const),
    (w, Duplicated),
    (v, Duplicated)
)

# Or with a different set of arguments
test_reverse(g,
    Active,
    (x_rev, Duplicated),
    (y_rev, Duplicated),
    (model, Duplicated),
    (x_0, Duplicated),
    (w, Const),
    (v, Const)
)
```
In all cases we must ensure the mutated arguments are passed as `Duplicated`.

Functions which internally use buffers can allocate them, but will need to ensure that the buffers are of the appropriate type (i.e., duplicated if they are active).  This can be achieved with the `eltype`

```{code-cell} julia
function g2(model, x_0, w, v)
    x = zeros(eltype(x_0), N, T + 1)
    y = zeros(eltype(x_0), M, T + 1)
    simulate_lss!(x, y, model, x_0, w, v)
    return mean_first_observation(y)
end
g2(model, x_0, w, v)
gradient(Reverse, g2, model, x_0, w, v)
```


### Calibration

Reverse-mode in particular can be useful for calibration with simulated dynamics.

For example, consider if the our $A$ matrix was parameterized by a scalar $a$ in its upper left corner, and we wanted to calibrate $a$ so that the time average of the first observation matched a target value $y^* = 0$.


For some technical reasons discussed below, we rewrite the simulation to take the model parameters individually rather than as a named tuple. A version that works with the original function is given below.

```{code-cell} julia
function simulate_lss!(x, y, A, C, G, H, x_0, w, v)
    N, T1 = size(x)
    M, T1y = size(y)
    T = size(w, 2)

    @assert T1 == T + 1 == T1y
    @assert size(v, 2) == T + 1
    @assert size(A) == (N, N)
    @assert size(G) == (M, N)
    @assert size(C, 1) == N
    @assert size(H, 1) == M
    @assert length(x_0) == N

    # Enzyme has challenges with activity analysis on broadcasting assignments
    @inbounds for i in 1:N
        x[i, 1] = x_0[i]
    end
    # Apply evolution and observation equations
    @inbounds for t in 1:T
        @views mul!(x[:, t + 1], A, x[:, t])             # x_{t+1} = A x_t
        @views mul!(x[:, t + 1], C, w[:, t], 1.0, 1.0)   # + C w_{t+1}

        @views mul!(y[:, t], G, x[:, t])                 # y_t = G x_t
        @views mul!(y[:, t], H, v[:, t], 1.0, 1.0)       # + H v_t
    end
    # Apply observation equation at T+1
    @views mul!(y[:, T + 1], G, x[:, T + 1])
    @views mul!(y[:, T + 1], H, v[:, T + 1], 1.0, 1.0)

    return nothing
end

function parameterized_A(a)
    return [a 0.1 0.0; 0.0 0.7 0.1; 0.0 0.0 0.6]
end

function loss(u, p)
    (; x_0, w, v, y_target, C, G, H) = p
    T = size(w, 2)
    a = u[1]

    A = parameterized_A(a)

    # Allocate buffers and simulate
    x = zeros(eltype(A), length(x_0), T + 1)
    y = zeros(eltype(A), size(G, 1), T + 1)
    simulate_lss!(x, y, A, C, G, H, x_0, w, v)

    y_mean = mean_first_observation(y)
    return (y_mean - y_target)^2
end
```


There are a few tricks to note here, which work around challenges with using Enzyme in its current state.

- With the SciML packages, such as [Optimization.jl](https://github.com/SciML/Optimization.jl), the `AutoEnzyme()` automatically determines which variables to mark as `Active` following certain patterns
- In particular, `p` holds parameters assumed to be constant during optimization, while `u` holds the optimization variables.
- Enzyme generally avoids allocations, but allocation is necessary here since we need to create a new model with the appropriate `a` value.
- For the buffers, note the use of `eltype(A)` to ensure the correct type, since the `A` matrix itself will be differentiable.


Using this setup, we can create the `p` parameter named tuple, none of which will be differentiable, and then use the LBFGS optimizer in OptimizationOptimJL to find the optimal `a` value.

```{code-cell} julia 
y_target = 0.0

# Bundle parameters into a named tuple
p = (; x_0, w, v, y_target, C, G, H)

# Initial Guess for 'a'
u_0 = [0.8]

println("Initial a=$(u_0[1]), Loss: $(loss(u_0, p))")

# Define the problem type, associate initial condition and parameters
# AutoEnzyme() will handle the AD setup
optf = OptimizationFunction(loss, AutoEnzyme())
prob = OptimizationProblem(optf, u_0, p)

sol = solve(prob, OptimizationOptimJL.LBFGS())
println("Final a=$(sol.u[1]), Loss: $(loss(sol.u, p))")
```

### Calibration with More Complicated Types

```{code-cell} julia

function loss_2(u, p)

    # Unpack constants
    (; x_0, w, v, y_target, C, G, H) = p
    T = size(w, 2)
    a = u[1]

    A = parameterized_A(a) # A is Active (depends on u)

    # Trick: "Launder" the constants by copying them.
    # Creates new locals which Enzyme sees these as "Local Active Variables"
    # Now build the struct with homogeneous (all local) variables
    model = (; A, C = copy(C), G = copy(G), H = copy(H))

    # Allocate buffers and simulate
    x = zeros(eltype(A), length(x_0), T + 1)
    y = zeros(eltype(A), size(G, 1), T + 1)
    simulate_lss!(x, y, model, x_0, w, v)

    y_mean = mean_first_observation(y)
    return (y_mean - y_target)^2
end
```

This version uses our original `simulate_lss!` function, which takes a named tuple for the model parameters. Note that it contains a trick where the constant parameters `C`, `G`, and `H` are "laundered" by copying them into new local variables before building the named tuple. This ensures that Enzyme can properly analyze which variables are active versus constant when creating the named tuple.

The `simulate_lss!` approach that splits out the `A` matrix might be more efficient for large $C, G, H$ since it avoids the copies, but it may not matter in practice.

The other code is identical with this new loss.

```{code-cell} julia 
y_target = 0.0

# Bundle parameters into a named tuple
p = (; x_0, w, v, y_target, C, G, H)

# Initial Guess for 'a'
u_0 = [0.8]

println("Initial a=$(u_0[1]), Loss: $(loss_2(u_0, p))")

# Define the problem type, associate initial condition and parameters
# AutoEnzyme() will handle the AD setup
optf = OptimizationFunction(loss_2, AutoEnzyme())
prob = OptimizationProblem(optf, u_0, p)

sol = solve(prob, OptimizationOptimJL.LBFGS())
println("Final a=$(sol.u[1]), Loss: $(loss_2(sol.u, p))")
```
