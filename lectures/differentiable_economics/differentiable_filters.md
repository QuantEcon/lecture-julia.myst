---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia
---

(diff_filters)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Differentiable Filters

```{index} single: Differentiable Filters
```

```{contents} Contents
:depth: 2
```

## Overview

Many problems in economics involve state-space models with unobserved states and noisy observations. For example, when estimating dynamic models an econometrician might have an unobserved TFP process and only observe noisy functions of aggregates such as capital and consumption from national accounts. Or within a model, an agent might have an unobserved state such as a belief about the state of the world, or a hidden match quality they are trying to learn -- as in {cite}`Jovanovic1979matching`'s model of job matching and turnover, {cite}`Jovanovic1982`'s model of firm selection and industry dynamics, and {cite}`Moscarini2005`'s equilibrium search model with learning about match quality.

The mathematical problem of estimating these unobserved states is called "filtering", where the {doc}`Kalman filter <../introduction_dynamics/kalman>` is the optimal filter given certain assumptions (e.g., a linear state-space model with Gaussian shocks).

One immediate application is estimation: the Kalman filter also computes the likelihood of a linear state-space model given observed data. If we can differentiate through the filter, we unlock gradient-based methods for estimation -- maximum likelihood via algorithms such as L-BFGS, Bayesian inference via [Hamiltonian Monte Carlo (HMC)](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) or [variational inference](https://en.wikipedia.org/wiki/Variational_inference), and sensitivity analysis of filtered states with respect to model parameters. These techniques are used in {cite}`dssm`.

This lecture builds on {doc}`differentiable dynamics <differentiable_dynamics>` (differentiable simulation) and the {doc}`Kalman filter <../introduction_dynamics/kalman>` (filter theory). See {doc}`auto-differentiation <../more_julia/auto_differentiation>` for background on Enzyme.

Along the way, you will learn several patterns for writing high-performance numerical code that is both generic and AD-compatible:

* The **bang-bang (`!!`) convention** -- a single algorithm that works with both heap-allocated mutable arrays (for large problems) and stack-allocated `StaticArrays` (for dramatic speedups on small problems)
* **Prototype-based allocation** for writing generic code that infers array types from inputs
* A **nonlinear state-space simulator** with callback-based transitions and observations, compatible with Enzyme
* A **high-performance zero-allocation Kalman filter** with forward- and reverse-mode AD, enabling gradient-based estimation of all model parameters in a single sweep
* Examples of auto-differentiation of more complicated functions, such as in-place solutions to linear systems (i.e., `ldiv!`) and matrix factorizations (i.e., `cholesky!`), which are common in many likelihoods

```{caution}
The code in this lecture is significantly more advanced than some of the other lectures, and requires some experience with both auto-differentiation concepts and a more detailed understanding of type-safety and memory management in Julia.

Enzyme.jl is under active development and while state-of-the-art, it is often bleeding-edge. Some of the patterns shown here may change in future releases. See {doc}`auto-differentiation <../more_julia/auto_differentiation>` for the latest best practices.
```

```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Random, Plots, Test, Enzyme, Statistics, RecursiveArrayTools
using BenchmarkTools, EnzymeTestUtils, StaticArrays
```

```{code-cell} julia
---
tags: [remove-cell]
---
using Test
```



## The Bang-Bang Pattern for Generic Arrays

### Motivation

The best coding strategy for numerical algorithms depends on the size of the problem. Different array types and mutation patterns are appropriate at different scales.

For large vectors and matrices, allocation of intermediate arrays can be a major bottleneck. In these cases, we want to write **in-place code** that mutates pre-allocated buffers to achieve zero allocations and maximum performance -- see {ref}`in-place-functions` for background. Writing zero-allocation code is also essential in practice for compatibility with Enzyme.

For small fixed-size problems, `StaticArrays` (stack-allocated, no heap) can be dramatically faster than standard `Vector`/`Matrix`. The [rule of thumb](https://juliaarrays.github.io/StaticArrays.jl/stable/) is that StaticArrays are beneficial when the **total number of elements** is under about 100 -- so an `SVector{10}` or an `SMatrix{3,3}` are good candidates, while an `SMatrix{20,20}` (400 elements) is not. The biggest wins (10x--100x) come at very small sizes (under ~20 elements); beyond 100 elements, compilation costs explode and the stack/register advantages disappear.

The challenge is that these two approaches seem incompatible: standard in-place operations (`mul!`, `ldiv!`, `copyto!`) only work with mutable arrays, while StaticArrays are immutable and require returning new values. Ideally, we want to write a **single algorithm** that works with both array types -- choosing the optimal strategy for each problem size without duplicating code.

### The `!!` Convention

One approach is a pattern called the "bang-bang" convention.

A function `f!!` **always returns its result** and **tries to mutate in-place when possible**.

It checks if a type can be modified directly with `ismutable(Y)` and then modifies in place if possible. For example, if mutable it might call `mul!(Y, A, B)` and return `Y`; if immutable it would simply return `A * B`.

This makes the natural data structure **arrays of arrays** (e.g., `Vector{SVector{N}}`) rather than 2D matrices, since each element can be either mutable or immutable.

### Key `!!` Utilities

| Function | Mutable path | Immutable path | Use case |
|----------|-------------|----------------|----------|
| `mul!!(Y, A, B)` | `mul!(Y,A,B); return Y` | `return A*B` | Matrix/vector multiply |
| `mul!!(Y, A, B, α, β)` | `mul!(Y,A,B,α,β); return Y` | `return α*(A*B) + β*Y` | Accumulate: `Y = αAB + βY` |
| `muladd!!(Y, A, B)` | `mul!(Y,A,B,1.0,1.0); return Y` | `return Y + A*B` | `Y += A*B` |
| `copyto!!(Y, X)` | `copyto!(Y,X); return Y` | `return X` | Copy data |
| `assign!!(Y, X)` | loop `Y[i]=X[i]; return Y` | `return X` | Enzyme-safe copy (avoids `Base.copyto!`) |

```{code-cell} julia
@inline function mul!!(Y, A, B)
    if ismutable(Y)
        mul!(Y, A, B)
        return Y
    else
        return A * B
    end
end

@inline function mul!!(Y, A, B, α, β)
    if ismutable(Y)
        mul!(Y, A, B, α, β)
        return Y
    else
        return α * (A * B) + β * Y
    end
end

@inline function muladd!!(Y, A, B)
    if ismutable(Y)
        mul!(Y, A, B, 1.0, 1.0)
        return Y
    else
        return Y + A * B
    end
end
```

We also define no-op specializations for `nothing` arguments. This lets us write generic code that handles optional components -- for example, a model with or without observation noise -- using a single code path. If `H = nothing`, then `muladd!!(y, H, v)` simply returns `y` unchanged without any branching at runtime.

```{code-cell} julia
@inline muladd!!(Y, ::Nothing, B) = Y
@inline muladd!!(Y, A, ::Nothing) = Y
@inline muladd!!(Y, ::Nothing, ::Nothing) = Y

@inline function copyto!!(Y, X)
    if ismutable(Y)
        copyto!(Y, X)
        return Y
    else
        return X
    end
end

@inline function assign!!(Y, X)
    if ismutable(Y)
        @inbounds for i in eachindex(X)
            Y[i] = X[i]
        end
        return Y
    else
        return X
    end
end
```

`assign!!` in this pattern helps to avoid `copyto!!` since Enzyme can have issues with runtime activity analysis errors when overwriting an initial condition in an output buffer.

### Prototype-Based Allocation

When writing generic code, we often need to allocate workspace arrays that match the type family of the inputs -- mutable `Vector`/`Matrix` or immutable `SVector`/`SMatrix`. Rather than maintaining separate allocation functions for each case, we use **prototype-based allocation**: pass an existing array as a template, and `alloc_like` creates a new zeroed array of the same type. The `SVector`/`SMatrix` overloads are necessary because Julia's `similar` returns a mutable `MVector`/`MMatrix` for static arrays, which would lose immutability and stack allocation.

```{code-cell} julia
# Same shape as prototype
@inline alloc_like(x::AbstractArray) = similar(x)
@inline alloc_like(::SVector{N, T}) where {N, T} = zeros(SVector{N, T})
@inline alloc_like(::SMatrix{N, M, T}) where {N, M, T} = zeros(SMatrix{N, M, T})

# Different dimensions, same type family
@inline alloc_like(x::AbstractArray, dims::Int...) = similar(x, dims...)
@inline alloc_like(::SVector{<:Any, T}, n::Int) where {T} = zeros(SVector{n, T})
@inline alloc_like(::SMatrix{<:Any, <:Any, T}, n::Int,
                   m::Int) where {T} = zeros(SMatrix{n, m, T})
@inline alloc_like(::SMatrix{<:Any, <:Any, T},
                   n::Int) where {T} = zeros(SVector{n, T})
```

Similarly, `fill_zero!!` zeroes an array generically -- in-place for mutable, returning a new zeroed value for immutable. This is needed to reset workspace caches between AD calls.

```{code-cell} julia
@inline fill_zero!!(::SVector{N, T}) where {N, T} = zeros(SVector{N, T})
@inline fill_zero!!(::SMatrix{N, M, T}) where {N, M,
                                               T} = zeros(SMatrix{N, M, T})
@inline function fill_zero!!(x::AbstractArray{T}) where {T}
    fill!(x, zero(T))
    return x
end
```

## High-Performance Nonlinear State-Space Simulation

Here we will make a variation on the simulation in {doc}`differentiable dynamics <differentiable_dynamics>`, but with a more flexible interface that supports arbitrary nonlinear state transitions and observation functions via callbacks, and is compatible with Enzyme.jl for AD.

### Arrays-of-Arrays Storage

With the `!!` pattern, the natural data structure is `Vector{Vector{Float64}}` or `Vector{SVector{N,Float64}}`. This enables a single simulation function for both mutable and static arrays.

### The `simulate_ssm!` Function

The model is

$$
x_{t+1} = f(x_t, w_{t+1}, p, t), \qquad y_t = g(x_t, p, t) + H v_t, \quad v_t \sim N(0, I)
$$

where $f$ and $g$ can be **arbitrary nonlinear functions**, but the observation noise is assumed **additive Gaussian** with $v_t \sim N(0, I)$.

To implement this for both static and preallocated arrays, the state transition callback `f!!(x_next, x, w, p, t)` implements the full $f(\cdot)$ (including process noise), while the observation callback `g!!(y, x, p, t)` implements only the deterministic component $g(\cdot)$. The simulator adds the Gaussian observation noise $H v_t$ separately via `muladd!!`. Both callbacks follow the `!!` convention -- they attempt to mutate the first argument in-place but always return the result. Passing `H = nothing` drops the observation noise entirely thanks to the no-op specializations above.

```{code-cell} julia
@inline function simulate_ssm!(x, y, f!!, g!!, x_0, w, v, H, p)
    T = length(w)

    # Initialize first state
    x[1] = assign!!(x[1], x_0)

    @inbounds for t in 1:T
        x[t + 1] = f!!(x[t + 1], x[t], w[t], p, t - 1) 
        y[t] = g!!(y[t], x[t], p, t - 1)
        y[t] = muladd!!(y[t], H, v[t])
    end

    # Final observation
    y[T + 1] = g!!(y[T + 1], x[T + 1], p, T)
    y[T + 1] = muladd!!(y[T + 1], H, v[T + 1])

    return nothing
end
```

### Linear State-Space Callbacks

As a concrete example, we define callbacks for the linear state-space model. The state transition callback implements $f(x_t, w_{t+1}) = Ax_t + Cw_{t+1}$, while the observation callback implements only the noiseless part $g(x_t) = Gx_t$. The full observation is $y_t = g(x_t) + Hv_t = Gx_t + Hv_t$, where the $Hv_t$ term is added by `simulate_ssm!`.

```{code-cell} julia
@inline function f_lss!!(x_p, x, w, p, t)
    x_p = mul!!(x_p, p.A, x)
    return muladd!!(x_p, p.C, w)
end

@inline function g_lss!!(y, x, p, t)
    return mul!!(y, p.G, x)
end
```

### Example: Small Linear State-Space Model

We set up a small $N=2$ model with two coupled states and two observables. The transition matrix $A$ has off-diagonal terms, so both states interact and move jointly. Both states receive independent shocks through $C$, and both are directly observed through $G = I$ with small measurement noise $H$.

```{code-cell} julia
Random.seed!(42)

A = [0.9 0.1; -0.1 0.8]    # coupled states: cross-feedback
C = [0.1 0.0; 0.0 0.1]     # independent shocks to each state
G = [1.0 0.0; 0.0 1.0]     # both states observed directly
H = [0.05 0.0; 0.0 0.05]   # small observation noise
model = (; A, C, G, H)

N = size(A, 1)     # state dimension
M = size(G, 1)     # observation dimension
K = size(C, 2)     # state noise dimension
L = size(H, 2)     # observation noise dimension
T = 50

x_0 = zeros(N)

# Generate noise sequences (arrays of arrays)
w = [randn(K) for _ in 1:T]
v = [randn(L) for _ in 1:(T + 1)]

# Allocate output arrays using prototypes
x = [alloc_like(x_0) for _ in 1:(T + 1)]
y = [alloc_like(x_0, M) for _ in 1:(T + 1)]

simulate_ssm!(x, y, f_lss!!, g_lss!!, x_0, w, v, H, model)

time = 0:T
plot(time, [x[t][1] for t in 1:(T + 1)], lw = 2, label = "x₁",
     xlabel = "t", ylabel = "state", title = "State Paths")
plot!(time, [x[t][2] for t in 1:(T + 1)], lw = 2, label = "x₂")
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Simulation regression values" begin
    @test x[end][1] ≈ 0.10265193580241357
    @test x[end][2] ≈ 0.02444692249574301
    @test y[end][1] ≈ 0.028265020918149895
    @test y[end][2] ≈ -0.05342589068431139
end
```

### RecursiveArrayTools.jl

With this pattern, accessing and slicing arrays of arrays can be burdensome.

The `RecursiveArrayTools.jl` package helps by wrapping the arrays in a `VectorOfArray` type, which provides convenient indexing and slicing while preserving the underlying array types.

```{code-cell} julia
x_arr = VectorOfArray(x)
y_arr = VectorOfArray(y)

@show x_arr[:,1], x_arr[1,1]
@show x_arr[1, 5:10];
```

### Running with StaticArrays

The same model wrapped in `SMatrix`/`SVector` produces identical results but runs on the stack.

```{code-cell} julia
A_s = SMatrix{N, N}(A)
C_s = SMatrix{N, K}(C)
G_s = SMatrix{M, N}(G)
H_s = SMatrix{M, L}(H)
model_s = (; A = A_s, C = C_s, G = G_s, H = H_s)

x_0_s = SVector{N}(x_0)
w_s = [SVector{K}(w[t]) for t in 1:T]
v_s = [SVector{L}(v[t]) for t in 1:(T + 1)]
x_s = [alloc_like(x_0_s) for _ in 1:(T + 1)]
y_s = [alloc_like(x_0_s, M) for _ in 1:(T + 1)]

simulate_ssm!(x_s, y_s, f_lss!!, g_lss!!, x_0_s, w_s, v_s, H_s, model_s)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Mutable vs Static simulation consistency" begin
    @test all(isapprox(x_s[t], x[t]; rtol = 1e-12) for t in 1:(T + 1))
    @test all(isapprox(y_s[t], y[t]; rtol = 1e-12) for t in 1:(T + 1))
end
```

### Benchmarks

Depending on your system, you will find a significant speedup for this small model when using `StaticArrays` -- often 10x or more, and neither version should allocate any memory.

```{code-cell} julia
@btime simulate_ssm!($x, $y, f_lss!!, g_lss!!, $x_0, $w, $v, $H, $model)
```

```{code-cell} julia
@btime simulate_ssm!($x_s, $y_s, f_lss!!, g_lss!!, $x_0_s, $w_s, $v_s, $H_s,
                     $model_s)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Simulation zero allocations" begin
    simulate_ssm!(x, y, f_lss!!, g_lss!!, x_0, w, v, H, model)
    @test (@allocated simulate_ssm!(x, y, f_lss!!, g_lss!!, x_0, w, v, H, model)) == 0
end
```

### Forward-Mode AD: Perturbing the Shocks

For the linear model $x_{t+1} = Ax_t + Cw_{t+1}$ starting from $x_1 = x_0$, a unit perturbation $\delta w_1 = e_k$ (the $k$-th standard basis vector) propagates through the dynamics as

$$
\frac{\partial x_{t+1}}{\partial w_{1,k}} = A^{t-1} C\, e_k, \qquad t \geq 1
$$

This is the **impulse response function** -- it shows how a one-time shock decays through the system. Forward-mode AD computes exactly these derivatives: by seeding $dw_1 = e_k$ and propagating tangents forward, the output `dx[t]` gives $\partial x_t / \partial w_{1,k}$ at every horizon.

Every argument to `autodiff` must be annotated to tell Enzyme how to handle it (see {ref}`enzyme-activity-rules`). Here `w` is `Duplicated` because we are differentiating with respect to it, and `dw` holds the seed tangent. The buffers `x` and `y` are also `Duplicated` because `simulate_ssm!` **mutates** them -- Enzyme needs shadow arrays `dx` and `dy` to propagate tangents through those writes, even though `x` and `y` are intermediate storage rather than parameters of interest. Arguments that are neither differentiated nor mutated (like the model coefficients) are wrapped in `Const`.

```{code-cell} julia
dx = Enzyme.make_zero(x)
dy = Enzyme.make_zero(y)
dw = Enzyme.make_zero(w)
dw[1][1] = 1.0  # unit perturbation to first shock element

autodiff(Forward, simulate_ssm!,
         Duplicated(x, dx), Duplicated(y, dy),
         Const(f_lss!!), Const(g_lss!!), Const(x_0),
         Duplicated(w, dw), Const(v), Const(H), Const(model))

plot(time, [dx[t][1] for t in 1:(T + 1)], lw = 2, label = "dx₁/dw₁₁",
     xlabel = "t", ylabel = "sensitivity",
     title = "Impulse Response via Forward AD")
plot!(time, [dx[t][2] for t in 1:(T + 1)], lw = 2, label = "dx₂/dw₁₁")
```

### Reverse-Mode AD: Sensitivity of the Terminal Observation

Reverse mode is best when computing the gradient of a **scalar** output with respect to **all** inputs in a single sweep, regardless of input dimension. Here the scalar is the first element of the final observation $y_{T+1,1}$ -- a "terminal observable" whose sensitivity tells us how each initial condition, shock, and model parameter affects this single outcome.

We wrap the simulation in a self-contained function that allocates its own workspace, calls `simulate_ssm!`, and returns $y_{T+1,1}$. The callbacks `f!!` and `g!!` are passed as keyword arguments to keep them out of the differentiation:

```{code-cell} julia
function terminal_observable(x_0, w, v, H, p; f!! = f_lss!!, g!! = g_lss!!)
    T_s = length(w)
    y_proto = alloc_like(x_0, size(H, 1))
    x = [alloc_like(x_0) for _ in 1:(T_s + 1)]
    y = [alloc_like(y_proto) for _ in 1:(T_s + 1)]
    simulate_ssm!(x, y, f!!, g!!, x_0, w, v, H, p)
    return y[end][1]
end
```

Because `terminal_observable` returns an `Active` scalar, Enzyme's reverse mode will back-propagate through `simulate_ssm!` and accumulate gradients in the `Duplicated` shadow arrays we provide:

```{code-cell} julia
dx_0 = Enzyme.make_zero(x_0)
dw = Enzyme.make_zero(w)
dmodel = Enzyme.make_zero(model)

autodiff(Reverse, terminal_observable, Active,
         Duplicated(x_0, dx_0), Duplicated(w, dw),
         Const(v), Const(H), Duplicated(model, dmodel))

println("∂y[T+1]₁/∂x_0:       ", dx_0)
println("∂y[T+1]₁/∂w[1]:      ", dw[1])
println("∂y[T+1]₁/∂A:         ", dmodel.A)
println("∂y[T+1]₁/∂G:         ", dmodel.G)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Reverse AD terminal observable" begin
    @test dx_0[1] ≈ -7.010566685527678e-5
    @test dx_0[2] ≈ -0.0004130964112403158
    @test dmodel.A[1, 1] ≈ -1.442574419229336
    @test dmodel.G[1, 1] ≈ 0.10265193580241357
end
```

## The Kalman Filter

### Additional `!!` Utilities

Before the filter, we need utilities for Cholesky factorization, linear solves, transposition, symmetrization, and log-determinants. Each serves a specific purpose in the filter:

- **`cholesky!!(A, :U)`** -- Cholesky factorization of innovation covariance $S_t$
- **`ldiv!!(y, F, x)`** and **`ldiv!!(F, x)`** -- solving $S_t^{-1} \nu_t$ for the log-likelihood and Kalman gain (the 2-arg form avoids internal allocation)
- **`transpose!!(Y, X)`** -- computing $K_t$ via $S_t K_t^{\top} = (\hat\Sigma_t G^{\top})^{\top}$
- **`symmetrize_upper!!(L, A, epsilon)`** -- enforcing exact symmetry before Cholesky (numerical drift in $\hat\Sigma_t$ can break it); the `epsilon` diagonal perturbation ensures positive definiteness
- **`logdet_chol(F)`** -- allocation-free log-determinant from Cholesky factor

| Function | Mutable path | Immutable path |
|----------|-------------|----------------|
| `cholesky!!(A, :U)` | `cholesky!(Symmetric(A,:U), NoPivot())` | `cholesky(Symmetric(A,:U))` |
| `ldiv!!(y, F, x)` | `ldiv!(y,F,x); return y` | `return F \ x` |
| `ldiv!!(F, x)` | `ldiv!(F,x); return x` | `return F \ x` |
| `transpose!!(Y, X)` | `transpose!(Y,X); return Y` | `return transpose(X)` |
| `symmetrize_upper!!(L, A, epsilon)` | element-wise loop, zero lower | `(A+A')/2 + epsilon*I` |
| `logdet_chol(F)` | `2*sum(log(diag(F.U)))` | same |

```{code-cell} julia
@inline function cholesky!!(A, uplo::Symbol = :U)
    if ismutable(A)
        return cholesky!(Symmetric(A, uplo), NoPivot(); check = false)
    else
        return cholesky(Symmetric(A, uplo))
    end
end

@inline function ldiv!!(y, F, x)
    if ismutable(y)
        ldiv!(y, F, x)
        return y
    else
        return F \ x
    end
end

@inline function ldiv!!(F, x)
    if ismutable(x)
        ldiv!(F, x)
        return x
    else
        return F \ x
    end
end

@inline function transpose!!(Y, X)
    if ismutable(Y)
        transpose!(Y, X)
        return Y
    else
        return transpose(X)
    end
end

@inline function logdet_chol(F)
    U = F.U
    result = zero(eltype(U))
    @inbounds for i in axes(U, 1)
        result += log(U[i, i])
    end
    return 2 * result
end

@noinline function symmetrize_upper!!(L, A, epsilon = 0.0)
    if ismutable(L)
        @inbounds for j in axes(A, 2)
            for i in 1:j
                v = (A[i, j] + A[j, i]) * 0.5
                L[i, j] = (i == j) ? v + epsilon : v
            end
            for i in (j + 1):size(A, 1)
                L[i, j] = 0
            end
        end
        return L
    else
        sym = (A + A') / 2
        if epsilon != 0
            return sym + epsilon * one(A)
        else
            return sym
        end
    end
end
```

### State-Space Model

The linear state-space model (see {doc}`linear models <../introduction_dynamics/linear_models>`) is

$$
x_{t+1} = A x_t + C w_{t+1}, \quad w_t \sim N(0, I)
$$
$$
y_t = G x_t + H v_t, \quad v_t \sim N(0, I)
$$

### Kalman Filter Recursion

The Kalman filter (see the {doc}`Kalman filter <../introduction_dynamics/kalman>` lecture) estimates the hidden state $x_t$ from noisy observations $\{y_1, \ldots, y_T\}$. Written in predict-update form matching our code:

**Predict:**

$$
\hat\mu_t = A \mu_t, \qquad \hat\Sigma_t = A \Sigma_t A^{\top} + CC^{\top}
$$

**Innovation:**

$$
\nu_t = y_t - G\hat\mu_t, \qquad S_t = G\hat\Sigma_t G^{\top} + HH^{\top}
$$

**Update:**

$$
K_t = \hat\Sigma_t G^{\top} S_t^{-1}, \quad \mu_{t+1} = \hat\mu_t + K_t\nu_t, \quad \Sigma_{t+1} = \hat\Sigma_t - K_t G \hat\Sigma_t
$$

### Log-Likelihood

Under the Gaussian assumption, the innovation $\nu_t = y_t - G\hat\mu_t$ is distributed as

$$
\nu_t \sim N(0, S_t), \qquad S_t = G\hat\Sigma_t G^{\top} + HH^{\top}
$$

so the log-density of $\nu_t$ under the multivariate normal $N(0, S_t)$ is

$$
\log p(\nu_t) = -\frac{1}{2}\bigl(M \log 2\pi + \log|S_t| + \nu_t^{\top} S_t^{-1} \nu_t\bigr)
$$

Summing over observations gives the log-likelihood:

$$
\ell = \sum_{t=1}^T \log p(\nu_t) = -\frac{1}{2}\sum_{t=1}^T \bigl(M\log 2\pi + \log|S_t| + \nu_t^{\top} S_t^{-1}\nu_t\bigr)
$$

In the code, we compute $\log|S_t|$ from its Cholesky factor $S_t = U^{\top}U$ via `logdet_chol(F)`, which sums $2\sum_i \log U_{ii}$ without any allocation. The quadratic form $\nu_t^{\top} S_t^{-1} \nu_t$ is computed by first solving $S_t^{-1}\nu_t$ with `ldiv!!`, then taking the inner product with `dot`.

### Workspace Cache

Zero-allocation code requires preallocating **all** intermediate buffers. This is especially important for AD since mutations must be tracked. Using `alloc_like`, a single allocation function handles both mutable and static arrays -- it infers all types and dimensions from the prototypes `mu_0`, `Sigma_0`, and `model.G`.

```{code-cell} julia
function alloc_kalman_cache(mu_0, Sigma_0, model, T)
    N = size(Sigma_0, 1)
    M = size(model.G, 1)
    return (;
            mu_pred = [alloc_like(mu_0) for _ in 1:T],
            sigma_pred = [alloc_like(Sigma_0) for _ in 1:T],
            A_sigma = [alloc_like(Sigma_0) for _ in 1:T],
            sigma_Gt = [alloc_like(Sigma_0, N, M) for _ in 1:T],
            innovation = [alloc_like(mu_0, M) for _ in 1:T],
            innovation_cov = [alloc_like(Sigma_0, M, M) for _ in 1:T],
            S_chol = [alloc_like(Sigma_0, M, M) for _ in 1:T],
            innovation_solved = [alloc_like(mu_0, M) for _ in 1:T],
            gain_rhs = [alloc_like(model.G) for _ in 1:T],
            gain = [alloc_like(Sigma_0, N, M) for _ in 1:T],
            gainG = [alloc_like(Sigma_0) for _ in 1:T],
            KgSigma = [alloc_like(Sigma_0) for _ in 1:T],
            mu_update = [alloc_like(mu_0) for _ in 1:T])
end
```

Since the cache is reused across calls, it must be zeroed at the start of each filter run to prevent gradient accumulation during AD. Using `fill_zero!!`, the same loop handles both mutable and immutable arrays without branching.

```{code-cell} julia
function zero_kalman_cache!!(cache)
    @inbounds for t in 1:length(cache.mu_pred)
        cache.mu_pred[t] = fill_zero!!(cache.mu_pred[t])
        cache.sigma_pred[t] = fill_zero!!(cache.sigma_pred[t])
        cache.A_sigma[t] = fill_zero!!(cache.A_sigma[t])
        cache.sigma_Gt[t] = fill_zero!!(cache.sigma_Gt[t])
        cache.innovation[t] = fill_zero!!(cache.innovation[t])
        cache.innovation_cov[t] = fill_zero!!(cache.innovation_cov[t])
        cache.S_chol[t] = fill_zero!!(cache.S_chol[t])
        cache.innovation_solved[t] = fill_zero!!(cache.innovation_solved[t])
        cache.gain_rhs[t] = fill_zero!!(cache.gain_rhs[t])
        cache.gain[t] = fill_zero!!(cache.gain[t])
        cache.gainG[t] = fill_zero!!(cache.gainG[t])
        cache.KgSigma[t] = fill_zero!!(cache.KgSigma[t])
        cache.mu_update[t] = fill_zero!!(cache.mu_update[t])
    end
    return cache
end
```

### The `kalman!` Function

The full Kalman filter implementation. Inline comments reference the math equations above.

```{code-cell} julia
function kalman!(mu, Sigma, y, mu_0, Sigma_0, model, cache;
                 perturb_diagonal = 1e-8)
    (; A, C, G, H) = model
    T = length(y)

    # Zero cache buffers for Enzyme AD compatibility
    zero_kalman_cache!!(cache)

    # Initialize: μ₁ = μ₀, Σ₁ = Σ₀
    # Use assign!! (not copyto!!) to avoid aliasing between Sigma[1] and Sigma_0,
    # which would prevent Enzyme from differentiating through Sigma_0.
    mu[1] = assign!!(mu[1], mu_0)
    Sigma[1] = assign!!(Sigma[1], Sigma_0)

    loglik = zero(eltype(mu[1]))
    is_mutable = ismutable(mu[1])

    @inbounds for t in 1:T
        μt = mu[t]
        Σt = Sigma[t]

        # Unpack cache buffers for this timestep
        μp = cache.mu_pred[t]
        Σp = cache.sigma_pred[t]
        AΣ = cache.A_sigma[t]
        ΣGt = cache.sigma_Gt[t]
        ν = cache.innovation[t]
        S = cache.innovation_cov[t]
        S_chol_buf = cache.S_chol[t]
        ν_solved = cache.innovation_solved[t]
        rhs = cache.gain_rhs[t]
        K = cache.gain[t]
        KG = cache.gainG[t]
        KGS = cache.KgSigma[t]
        μu = cache.mu_update[t]

        # Predict mean: μ̂_t = A μ_t
        μp = mul!!(μp, A, μt)

        # Predict covariance: Σ̂_t = A Σ_t A' + CC'
        AΣ = mul!!(AΣ, A, Σt)
        Σp = mul!!(Σp, AΣ, transpose(A))
        Σp = muladd!!(Σp, C, transpose(C))

        # Innovation: ν_t = y_t - G μ̂_t
        ν = copyto!!(ν, y[t])
        ν = mul!!(ν, G, μp, -1.0, 1.0)

        # Innovation covariance: S_t = G Σ̂_t G' + HH'
        ΣGt = mul!!(ΣGt, Σp, transpose(G))
        S = mul!!(S, G, ΣGt)
        S = muladd!!(S, H, transpose(H))

        # Symmetrize and Cholesky factorize S_t
        S_chol_buf = symmetrize_upper!!(S_chol_buf, S, perturb_diagonal)
        F = cholesky!!(S_chol_buf, :U)

        # Kalman gain: K_t = Σ̂_t G' S_t⁻¹
        # Solve S_t K_t' = (Σ̂_t G')' for K_t
        rhs = transpose!!(rhs, ΣGt)
        rhs = ldiv!!(F, rhs)
        K = transpose!!(K, rhs)

        # Update mean: μ_{t+1} = μ̂_t + K_t ν_t
        # Mutable: mu[t+1] aliases a heap array -- write elements directly.
        # Immutable: μp and μu are new SVector values on the stack returned by
        # the !! functions. Write them back to the cache so Enzyme's reverse
        # pass can track them (otherwise the cache still holds stale zeros).
        μu = mul!!(μu, K, ν)
        if is_mutable
            for i in eachindex(μp)
                mu[t + 1][i] = μp[i] + μu[i]
            end
        else
            cache.mu_pred[t] = μp
            cache.mu_update[t] = μu
            mu[t + 1] = μp + μu
        end

        # Update covariance: Σ_{t+1} = Σ̂_t - K_t G Σ̂_t
        # Same mutable/immutable cache-writeback logic as mean update above.
        KG = mul!!(KG, K, G)
        KGS = mul!!(KGS, KG, Σp)
        if is_mutable
            for i in eachindex(Σp)
                Sigma[t + 1][i] = Σp[i] - KGS[i]
            end
        else
            cache.sigma_pred[t] = Σp
            cache.KgSigma[t] = KGS
            Sigma[t + 1] = Σp - KGS
        end

        # Log-likelihood contribution
        ν_solved = ldiv!!(ν_solved, F, ν)
        logdetS = logdet_chol(F)
        M_obs = length(ν)
        quad = dot(ν_solved, ν)
        loglik -= 0.5 * (M_obs * log(2π) + logdetS + quad)
    end

    return loglik
end
```

### Running the Filter

We generate synthetic observations from the model and run `kalman!`.

```{code-cell} julia
# Generate synthetic observations using simulate_ssm!
Random.seed!(123)
T_kf = 20
mu_0 = zeros(N)
Sigma_0 = Matrix{Float64}(I, N, N)

x_0_kf = mu_0 + cholesky(Sigma_0).L * randn(N)
w_kf = [randn(K) for _ in 1:T_kf]
v_kf = [randn(L) for _ in 1:(T_kf + 1)]

x_true = [alloc_like(mu_0) for _ in 1:(T_kf + 1)]
y_sim = [alloc_like(mu_0, M) for _ in 1:(T_kf + 1)]
simulate_ssm!(x_true, y_sim, f_lss!!, g_lss!!, x_0_kf, w_kf, v_kf, H, model)
y_obs = y_sim[1:T_kf]

# Run our kalman! filter
mu_kf = [alloc_like(mu_0) for _ in 1:(T_kf + 1)]
Sigma_kf = [alloc_like(Sigma_0) for _ in 1:(T_kf + 1)]
cache_kf = alloc_kalman_cache(mu_0, Sigma_0, model, T_kf)

loglik = kalman!(mu_kf, Sigma_kf, y_obs, mu_0, Sigma_0, model, cache_kf)
println("Log-likelihood: ", loglik)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Kalman filter regression values" begin
    @test loglik ≈ 26.518326073316906
    @test mu_kf[end][1] ≈ -0.06300502070665298
    @test mu_kf[end][2] ≈ 0.05513998852768426
end
```

### Plot: Filtered vs True States

```{code-cell} julia
time_kf = 0:T_kf
plot(time_kf, [x_true[t][1] for t in 1:(T_kf + 1)], lw = 2,
     label = "true state x₁",
     xlabel = "t", ylabel = "state", title = "Filtered vs True States",
     ls = :dash)
plot!(time_kf, [mu_kf[t][1] for t in 1:(T_kf + 1)], lw = 2,
      label = "filtered μ₁")
plot!(time_kf, [x_true[t][2] for t in 1:(T_kf + 1)], lw = 2,
      label = "true state x₂",
      ls = :dash)
plot!(time_kf, [mu_kf[t][2] for t in 1:(T_kf + 1)], lw = 2,
      label = "filtered μ₂")
```

## Performance

In this section we check important performance characteristics of our Kalman filter implementation, including type stability, zero allocations, and speed benchmarks. We also compare the mutable version against a static version using `StaticArrays` to see the tradeoffs in speed and flexibility.

### Type Stability

```{code-cell} julia
@inferred kalman!(mu_kf, Sigma_kf, y_obs, mu_0, Sigma_0, model, cache_kf)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Kalman zero allocations" begin
    kalman!(mu_kf, Sigma_kf, y_obs, mu_0, Sigma_0, model, cache_kf)
    @test (@allocated kalman!(mu_kf, Sigma_kf, y_obs, mu_0, Sigma_0, model, cache_kf)) == 0
end
```

### Benchmarks: Small Static vs Mutable

```{code-cell} julia
@btime kalman!($mu_kf, $Sigma_kf, $y_obs, $mu_0, $Sigma_0, $model, $cache_kf)
```

```{code-cell} julia
# Static version -- alloc_like infers SVector/SMatrix from prototypes
mu_0_s = SVector{N}(mu_0)
Sigma_0_s = SMatrix{N, N}(Sigma_0)
y_obs_s = [SVector{M}(y_obs[t]) for t in 1:T_kf]
mu_kf_s = [alloc_like(mu_0_s) for _ in 1:(T_kf + 1)]
Sigma_kf_s = [alloc_like(Sigma_0_s) for _ in 1:(T_kf + 1)]
cache_kf_s = alloc_kalman_cache(mu_0_s, Sigma_0_s, model_s, T_kf)

@btime kalman!($mu_kf_s, $Sigma_kf_s, $y_obs_s, $mu_0_s, $Sigma_0_s, $model_s,
               $cache_kf_s)
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Static vs mutable Kalman consistency" begin
    loglik_mut = kalman!(mu_kf, Sigma_kf, y_obs, mu_0, Sigma_0, model, cache_kf)
    loglik_sta = kalman!(mu_kf_s, Sigma_kf_s, y_obs_s, mu_0_s, Sigma_0_s, model_s, cache_kf_s)
    @test loglik_mut ≈ loglik_sta rtol = 1e-10
    for t in 1:(T_kf + 1)
        @test mu_kf[t] ≈ Vector(mu_kf_s[t]) rtol = 1e-10
    end
end
```

### Benchmark: Larger Model

```{code-cell} julia
Random.seed!(42)
N_big, M_big, K_big, L_big, T_big = 5, 2, 5, 2, 100

A_big_raw = randn(N_big, N_big)
A_big = 0.5 * A_big_raw / maximum(abs.(eigvals(A_big_raw)))
C_big = 0.1 * randn(N_big, K_big)
G_big = randn(M_big, N_big)
H_big = 0.1 * randn(M_big, L_big)
model_big = (; A = A_big, C = C_big, G = G_big, H = H_big)

mu_0_big = zeros(N_big)
Sigma_0_big = Matrix{Float64}(I, N_big, N_big)

y_big = [randn(M_big) for _ in 1:T_big]
mu_big = [alloc_like(mu_0_big) for _ in 1:(T_big + 1)]
Sigma_big = [alloc_like(Sigma_0_big) for _ in 1:(T_big + 1)]
cache_big = alloc_kalman_cache(mu_0_big, Sigma_0_big, model_big, T_big)

@btime kalman!($mu_big, $Sigma_big, $y_big, $mu_0_big, $Sigma_0_big,
               $model_big, $cache_big)
```

## Differentiating the Kalman Filter

### Forward Mode: Sensitivity of Prior Mean

We perturb the first element of $\mu_0$ by a unit tangent and observe how the filtered means respond over time. As with the simulation, every mutated buffer (`mu`, `Sigma`, `cache`, `y`) needs a shadow array for Enzyme to propagate tangents through the writes.

```{code-cell} julia
mu_fwd = [alloc_like(mu_0) for _ in 1:(T_kf + 1)]
Sigma_fwd = [alloc_like(Sigma_0) for _ in 1:(T_kf + 1)]
cache_fwd = alloc_kalman_cache(mu_0, Sigma_0, model, T_kf)
dmu_fwd = Enzyme.make_zero(mu_fwd)
dSigma_fwd = Enzyme.make_zero(Sigma_fwd)
dcache_fwd = Enzyme.make_zero(cache_fwd)
dy_fwd = Enzyme.make_zero(y_obs)

dmu_0_fwd = zeros(N)
dmu_0_fwd[1] = 1.0  # seed: perturb first state element

autodiff(Forward, kalman!,
         Duplicated(mu_fwd, dmu_fwd),
         Duplicated(Sigma_fwd, dSigma_fwd),
         Duplicated(y_obs, dy_fwd),
         Duplicated(mu_0, dmu_0_fwd),
         Const(Sigma_0),
         Const(model),
         Duplicated(cache_fwd, dcache_fwd))

plot(time_kf, [dmu_fwd[t][1] for t in 1:(T_kf + 1)], lw = 2, label = "dμ₁/dμ₀₁",
     xlabel = "t", ylabel = "sensitivity",
     title = "Sensitivity of Filtered Mean to Prior μ₀[1]")
plot!(time_kf, [dmu_fwd[t][2] for t in 1:(T_kf + 1)], lw = 2,
      label = "dμ₂/dμ₀₁")
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Forward AD Kalman" begin
    @test dmu_fwd[2][1] ≈ 0.0026986699332962605
    @test dmu_fwd[2][2] ≈ -0.00033662535441628116
end
```

### Reverse Mode: Gradient of Log-Likelihood

Reverse mode gives gradients of the log-likelihood with respect to **all** model parameters and initial conditions in a single sweep, regardless of parameter dimension. This is what makes gradient-based MLE practical.

As with `terminal_observable`, we wrap `kalman!` in a self-contained function that allocates its own workspace and returns the scalar log-likelihood:

```{code-cell} julia
function kalman_loglik(y, mu_0, Sigma_0, model)
    T_k = length(y)
    mu = [alloc_like(mu_0) for _ in 1:(T_k + 1)]
    Sigma = [alloc_like(Sigma_0) for _ in 1:(T_k + 1)]
    cache = alloc_kalman_cache(mu_0, Sigma_0, model, T_k)
    return kalman!(mu, Sigma, y, mu_0, Sigma_0, model, cache)
end
```

Because `kalman_loglik` returns an `Active` scalar, Enzyme back-propagates through `kalman!` and accumulates gradients in the `Duplicated` shadow arrays. We use the larger $N=5$ model for this demonstration.

```{code-cell} julia
Random.seed!(456)
y_big_obs = [randn(M_big) for _ in 1:T_big]

dmu_0_rev = Enzyme.make_zero(mu_0_big)
dmodel_rev = Enzyme.make_zero(model_big)

autodiff(Reverse, kalman_loglik, Active,
         Const(y_big_obs), Duplicated(mu_0_big, dmu_0_rev),
         Const(Sigma_0_big), Duplicated(model_big, dmodel_rev))

println("∂ℓ/∂μ₀:             ", round.(dmu_0_rev; digits = 4))
println("∂ℓ/∂A (first row):   ", round.(dmodel_rev.A[1, :]; digits = 4))
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Reverse AD Kalman" begin
    @test dmu_0_rev[1] ≈ 1.6721832674300914
    @test dmu_0_rev[2] ≈ 0.2633814453617159
    @test dmodel_rev.A[1, 1] ≈ 682.6310040254984
end
```

### AD Correctness: EnzymeTestUtils

We validate forward and reverse modes against finite differences using `test_forward` and `test_reverse` on a small ($N=2$, $T=2$) model.

Enzyme is in active development and bugs in AD can occur.

```{code-cell} julia
N_test, M_test, T_test = 2, 2, 2

A_test = [0.8 0.1; -0.1 0.7]
C_test = [0.1 0.0; 0.0 0.1]
G_test = [1.0 0.0; 0.0 1.0]
H_test = [0.1 0.0; 0.0 0.1]
model_test = (; A = A_test, C = C_test, G = G_test, H = H_test)

mu_0_test = zeros(N_test)
Sigma_0_test = Matrix{Float64}(I, N_test, N_test)
y_test = [[0.5, 0.3], [0.2, 0.1]]

mu_et = [alloc_like(mu_0_test) for _ in 1:(T_test + 1)]
Sigma_et = [alloc_like(Sigma_0_test) for _ in 1:(T_test + 1)]
cache_et = alloc_kalman_cache(mu_0_test, Sigma_0_test, model_test, T_test)

test_forward(kalman!, Const,
             (mu_et, Duplicated),
             (Sigma_et, Duplicated),
             (y_test, Duplicated),
             (mu_0_test, Duplicated),
             (copy(Sigma_0_test), Const),
             (model_test, Duplicated),
             (cache_et, Duplicated))
```

```{code-cell} julia
test_reverse(kalman!, Const,
             (mu_et, Duplicated),
             (Sigma_et, Duplicated),
             (y_test, Duplicated),
             (mu_0_test, Duplicated),
             (copy(Sigma_0_test), Const),
             (model_test, Duplicated),
             (cache_et, Duplicated))
```
