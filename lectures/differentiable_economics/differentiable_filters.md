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

This lecture builds on {doc}`differentiable dynamics <differentiable_dynamics>` (simulation + AD) and {doc}`the Kalman filter <../introduction_dynamics/kalman>` (filter theory). See {doc}`auto-differentiation <../more_julia/auto_differentiation>` for background on Enzyme.

Here we introduce **advanced techniques for high-performance computing** -- in particular, working with both in-place mutable operations and immutable StaticArrays, then applying these to a generic **nonlinear** state-space simulator and a high-performance Kalman filter, both compatible with Enzyme.jl.

**Unlike other lectures, this focuses on performance rather than simple clarity.**

The simulation framework supports arbitrary nonlinear state transitions and observation functions via callbacks, though the Kalman filter itself applies to the linear case.

```{caution}
The code in this section is significantly more advanced than some of the other lectures, and requires some experience with both auto-differentiation concepts and a more detailed understanding of type-safety and memory management in Julia.

[Enzyme.jl](https://enzyme.mit.edu/julia/stable/) is under active development and while state-of-the-art, it is often bleeding-edge. Some of the patterns shown here may change in future releases. See {doc}`auto-differentiation <../more_julia/auto_differentiation>` for the latest best practices.
```

```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Random, Plots, Test, Enzyme, Statistics
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

For small fixed-size problems, `StaticArrays` (stack-allocated, no heap) can be dramatically faster than standard `Vector`/`Matrix`. The [rule of thumb](https://juliaarrays.github.io/StaticArrays.jl/stable/) is that StaticArrays are beneficial when the **total number of elements** is under about 100 -- so an `SVector{50}` or an `SMatrix{8,8}` (64 elements) are good candidates, while an `SMatrix{20,20}` (400 elements) is not. The biggest wins (10x--100x) come at very small sizes (under ~20 elements); beyond 100 elements, compilation costs explode and the stack/register advantages disappear.

The challenge is that standard in-place operations (`mul!`, `ldiv!`, `copyto!`) only work with mutable arrays. StaticArrays are immutable -- you must return new values.

Our goal is to write algorithms that work with **both** array types via a single code path.

### The `!!` Convention

A function `f!!` **always returns its result** and **tries to mutate in-place when possible**.

It dispatches on `ismutable(Y)`: if mutable, call `mul!(Y, A, B)` and return `Y`; if immutable, return `A * B`.

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

We also define **no-op specializations** for `nothing` arguments. This lets us write generic code that handles optional components -- for example, a model with or without observation noise -- using a single code path. If `H = nothing`, then `muladd!!(y, H, v)` simply returns `y` unchanged without any branching at runtime.

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

# Enzyme-safe alternative to copyto!! -- uses explicit loops to avoid
# Base.copyto! which can trigger runtime activity analysis errors.
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

`assign!!` is the safer choice for code differentiated with Enzyme; `copyto!!` delegates to `Base.copyto!`, which can trigger runtime activity analysis errors in some contexts.

### Prototype-Based Allocation

When writing generic code, we need to allocate workspace arrays that match the type family of the inputs -- mutable `Vector`/`Matrix` or immutable `SVector`/`SMatrix`. Rather than maintaining separate allocation functions for each case, we use **prototype-based allocation**: pass an existing array as a template, and `alloc_like` creates a new zeroed array of the same type. The `SVector`/`SMatrix` overloads are necessary because Julia's `similar` returns a mutable `MVector`/`MMatrix` for static arrays, which would lose immutability and stack allocation.

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

### Arrays-of-Arrays Storage

With the `!!` pattern, the natural data structure is `Vector{Vector{Float64}}` or `Vector{SVector{N,Float64}}`. This enables a **single simulation function** for both mutable and static arrays.

### The `simulate_ssm!` Function

The model is

$$
x_{t+1} = f(x_t, w_{t+1}, p, t), \qquad y_t = g(x_t, p, t) + H v_t, \quad v_t \sim N(0, I)
$$

where $f$ and $g$ can be **arbitrary nonlinear functions**, but the observation noise is assumed **additive Gaussian** with $v_t \sim N(0, I)$. The state transition callback `f!!(x_next, x, w, p, t)` implements the full $f(\cdot)$ (including process noise), while the observation callback `g!!(y, x, p, t)` implements only the noiseless $g(\cdot)$. The simulator adds the Gaussian observation noise $Hv_t$ separately via `muladd!!`. Both callbacks follow the `!!` convention -- they attempt to mutate the first argument in-place but always return the result. Passing `H = nothing` drops the observation noise entirely thanks to the no-op specializations above.

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

### Forward-Mode AD: Impulse Response

For the linear model $x_{t+1} = Ax_t + Cw_{t+1}$ starting from $x_1 = x_0$, a unit perturbation $\delta w_1 = e_k$ (the $k$-th standard basis vector) propagates through the dynamics as

$$
\frac{\partial x_{t+1}}{\partial w_{1,k}} = A^{t-1} C\, e_k, \qquad t \geq 1
$$

This is the **impulse response function** -- it shows how a one-time shock decays through the system. Forward-mode AD computes exactly these derivatives: by seeding $dw_1 = e_k$ and propagating tangents forward, the output `dx[t]` gives $\partial x_t / \partial w_{1,k}$ at every horizon.

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

### Reverse-Mode AD: Gradient of Scalar Summary

Reverse mode gives the gradient with respect to **all** inputs in a single sweep, regardless of input dimension.

To use `Enzyme.autodiff` in reverse mode we need a scalar-valued function and shadow (gradient) arrays for every `Duplicated` argument. First we define a scalar summary:

```{code-cell} julia
function scalar_ssm(x, y, f!!, g!!, x_0, w, v, H, p)
    simulate_ssm!(x, y, f!!, g!!, x_0, w, v, H, p)
    total = zero(eltype(x[1]))
    @inbounds for t in eachindex(x)
        total += mean(x[t])
    end
    @inbounds for t in eachindex(y)
        total += mean(y[t])
    end
    return total
end
```

The raw `autodiff` call requires allocating shadow arrays for each `Duplicated` argument:

```{code-cell} julia
x_rev = [alloc_like(x_0) for _ in 1:(T + 1)]
y_rev = [alloc_like(x_0, M) for _ in 1:(T + 1)]
dx_rev = Enzyme.make_zero(x_rev)
dy_rev = Enzyme.make_zero(y_rev)
dx_0_rev = Enzyme.make_zero(x_0)
dw_rev = Enzyme.make_zero(w)

autodiff(Reverse, scalar_ssm,
         Duplicated(x_rev, dx_rev), Duplicated(y_rev, dy_rev),
         Const(f_lss!!), Const(g_lss!!),
         Duplicated(x_0, dx_0_rev), Duplicated(w, dw_rev),
         Const(v), Const(H), Const(model))

println("Gradient w.r.t. x_0: ", dx_0_rev)
println("Gradient w.r.t. w[1]: ", dw_rev[1])
```

In practice, it is convenient to wrap this boilerplate into a gradient function that handles all the shadow allocation internally:

```{code-cell} julia
function gradient_ssm(x_0, w; f!! = f_lss!!, g!! = g_lss!!, v, H, model)
    T_s = length(w)
    y_proto = alloc_like(x_0, size(H, 1))  # observation vector prototype
    x = [alloc_like(x_0) for _ in 1:(T_s + 1)]
    y = [alloc_like(y_proto) for _ in 1:(T_s + 1)]
    dx = Enzyme.make_zero(x)
    dy = Enzyme.make_zero(y)
    dx_0 = Enzyme.make_zero(x_0)
    dw = Enzyme.make_zero(w)

    autodiff(Reverse, scalar_ssm,
             Duplicated(x, dx), Duplicated(y, dy),
             Const(f!!), Const(g!!),
             Duplicated(x_0, dx_0), Duplicated(w, dw),
             Const(v), Const(H), Const(model))

    return (; grad_x_0 = dx_0, grad_w = dw)
end

grads = gradient_ssm(x_0, w; v, H, model)
println("Gradient w.r.t. x_0: ", grads.grad_x_0)
println("Gradient w.r.t. w[1]: ", grads.grad_w[1])
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Reverse AD simulation" begin
    @test grads.grad_x_0 ≈ dx_0_rev rtol = 1e-10
    @test grads.grad_w[1] ≈ dw_rev[1] rtol = 1e-10
    grad_sum = sum(abs, grads.grad_x_0) + sum(sum(abs, g) for g in grads.grad_w)
    @test grad_sum > 0
    @test !isnan(grad_sum)
end
```

## The Kalman Filter

### Additional `!!` Utilities

Before the filter, we need utilities for Cholesky factorization, linear solves, transposition, symmetrization, and log-determinants. Each serves a specific purpose in the filter:

- **`cholesky!!(A, :U)`** -- Cholesky factorization of innovation covariance $S_t$
- **`ldiv!!(y, F, x)`** and **`ldiv!!(F, x)`** -- solving $S_t^{-1} \nu_t$ for the log-likelihood and Kalman gain (the 2-arg form avoids internal allocation)
- **`transpose!!(Y, X)`** -- computing $K_t$ via $S_t K_t' = (\hat\Sigma_t G')'$
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
\hat\mu_t = A \mu_t, \qquad \hat\Sigma_t = A \Sigma_t A' + CC'
$$

**Innovation:**

$$
\nu_t = y_t - G\hat\mu_t, \qquad S_t = G\hat\Sigma_t G' + HH'
$$

**Update:**

$$
K_t = \hat\Sigma_t G' S_t^{-1}, \quad \mu_{t+1} = \hat\mu_t + K_t\nu_t, \quad \Sigma_{t+1} = \hat\Sigma_t - K_t G \hat\Sigma_t
$$

### Log-Likelihood

Under the Gaussian assumption, the innovation $\nu_t = y_t - G\hat\mu_t$ is distributed as

$$
\nu_t \sim N(0, S_t), \qquad S_t = G\hat\Sigma_t G' + HH'
$$

so the log-density of $\nu_t$ under the multivariate normal $N(0, S_t)$ is

$$
\log p(\nu_t) = -\frac{1}{2}\bigl(M \log 2\pi + \log|S_t| + \nu_t' S_t^{-1} \nu_t\bigr)
$$

Summing over observations gives the log-likelihood:

$$
\ell = \sum_{t=1}^T \log p(\nu_t) = -\frac{1}{2}\sum_{t=1}^T \bigl(M\log 2\pi + \log|S_t| + \nu_t' S_t^{-1}\nu_t\bigr)
$$

In the code, we compute $\log|S_t|$ from its Cholesky factor $S_t = U'U$ via `logdet_chol(F)`, which sums $2\sum_i \log U_{ii}$ without any allocation. The quadratic form $\nu_t' S_t^{-1} \nu_t$ is computed by first solving $S_t^{-1}\nu_t$ with `ldiv!!`, then taking the inner product with `dot`.

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
    mu[1] = copyto!!(mu[1], mu_0)
    Sigma[1] = copyto!!(Sigma[1], Sigma_0)

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
# Generate synthetic observations
Random.seed!(123)
T_kf = 20
mu_0 = zeros(N)
Sigma_0 = Matrix{Float64}(I, N, N)

x_true = [alloc_like(mu_0) for _ in 1:(T_kf + 1)]
y_obs = [alloc_like(mu_0, M) for _ in 1:T_kf]

# Simulate true states and noisy observations
x_true[1] = mu_0 + cholesky(Sigma_0).L * randn(N)
for t in 1:T_kf
    w_t = randn(K)
    v_t = randn(L)
    x_true[t + 1] = A * x_true[t] + C * w_t
    y_obs[t] = G * x_true[t] + H * v_t
end

# Run our kalman! filter
mu_kf = [alloc_like(mu_0) for _ in 1:(T_kf + 1)]
Sigma_kf = [alloc_like(Sigma_0) for _ in 1:(T_kf + 1)]
cache_kf = alloc_kalman_cache(mu_0, Sigma_0, model, T_kf)

loglik = kalman!(mu_kf, Sigma_kf, y_obs, mu_0, Sigma_0, model, cache_kf)
println("Log-likelihood: ", loglik)
```

### Validation: ControlSystems.jl Steady-State Comparison

We run `kalman!` for $T=200$ steps and compare the steady-state Kalman gain against `dkalman` from ControlSystems.jl.

```{code-cell} julia
using ControlSystems: dkalman

R1 = C * C'  # process noise covariance
R2 = H * H'  # measurement noise covariance

T_long = 200
Random.seed!(789)

# Generate long observation sequence
x_long = [alloc_like(mu_0) for _ in 1:(T_long + 1)]
y_long = [alloc_like(mu_0, M) for _ in 1:T_long]
x_long[1] = randn(N)
for t in 1:T_long
    x_long[t + 1] = A * x_long[t] + C * randn(K)
    y_long[t] = G * x_long[t] + H * randn(L)
end

# Run our filter
mu_long = [alloc_like(mu_0) for _ in 1:(T_long + 1)]
Sigma_long = [alloc_like(Sigma_0) for _ in 1:(T_long + 1)]
cache_long = alloc_kalman_cache(mu_0, Sigma_0, model, T_long)
kalman!(mu_long, Sigma_long, y_long, mu_0, Sigma_0, model, cache_long)

# ControlSystems.jl steady-state gain
# dkalman returns L in predictor form: x̂(k+1|k) = A*x̂(k|k-1) + L*(y(k) - G*x̂(k|k-1))
# Our filter form: K = A \ L
L_predictor = dkalman(A, G, R1, R2)
K_expected = A \ L_predictor

K_ours = cache_long.gain[end]
println("Our steady-state K:\n", round.(K_ours; digits = 6))
println("ControlSystems K:\n", round.(K_expected; digits = 6))
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "ControlSystems.jl steady-state comparison" begin
    @test K_ours ≈ K_expected rtol = 1e-6
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

We perturb the first element of $\mu_0$ by a unit tangent and observe how the filtered means respond over time.

```{code-cell} julia
# Set up fresh arrays for AD
mu_fwd = [alloc_like(mu_0) for _ in 1:(T_kf + 1)]
Sigma_fwd = [alloc_like(Sigma_0) for _ in 1:(T_kf + 1)]
cache_fwd = alloc_kalman_cache(mu_0, Sigma_0, model, T_kf)
dmu_fwd = Enzyme.make_zero(mu_fwd)
dSigma_fwd = Enzyme.make_zero(Sigma_fwd)
dcache_fwd = Enzyme.make_zero(cache_fwd)

dmu_0_fwd = zeros(N)
dmu_0_fwd[1] = 1.0  # perturb first state
dy_fwd = Enzyme.make_zero(y_obs)

@inline function scalar_kalman!(mu, Sigma, y, mu_0, Sigma_0, model, cache)
    return kalman!(mu, Sigma, y, mu_0, Sigma_0, model, cache)
end

result_fwd = autodiff(Forward, scalar_kalman!,
                      Duplicated(mu_fwd, dmu_fwd),
                      Duplicated(Sigma_fwd, dSigma_fwd),
                      Duplicated(y_obs, dy_fwd),
                      Duplicated(copy(mu_0), dmu_0_fwd),
                      Const(Sigma_0),
                      Const(model),
                      Duplicated(cache_fwd, dcache_fwd))

println("Tangent of loglik w.r.t. μ₀[1]: ", result_fwd[1])

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
    @test result_fwd[1] != 0.0
    @test !isnan(result_fwd[1])
    @test any(!iszero, dmu_fwd[2])
end
```

### Reverse Mode: Gradient of Log-Likelihood

Reverse mode gives gradients of the log-likelihood with respect to **all** model parameters and initial conditions in a single sweep, regardless of parameter dimension. This is what makes gradient-based MLE practical.

As with the simulation, we can wrap the shadow allocation boilerplate into a reusable gradient function:

```{code-cell} julia
function gradient_kalman(y, mu_0, Sigma_0, model)
    T_g = length(y)
    mu = [alloc_like(mu_0) for _ in 1:(T_g + 1)]
    Sigma = [alloc_like(Sigma_0) for _ in 1:(T_g + 1)]
    cache = alloc_kalman_cache(mu_0, Sigma_0, model, T_g)
    dmu = Enzyme.make_zero(mu)
    dSigma = Enzyme.make_zero(Sigma)
    dcache = Enzyme.make_zero(cache)
    dmodel = Enzyme.make_zero(model)
    dmu_0 = Enzyme.make_zero(mu_0)
    dy = Enzyme.make_zero(y)

    autodiff(Reverse, scalar_kalman!,
             Duplicated(mu, dmu), Duplicated(Sigma, dSigma),
             Duplicated(y, dy), Duplicated(copy(mu_0), dmu_0),
             Const(Sigma_0), Duplicated(model, dmodel),
             Duplicated(cache, dcache))

    return (; grad_mu_0 = dmu_0, grad_model = dmodel)
end
```

We use the larger $N=5$ model for this demonstration.

```{code-cell} julia
Random.seed!(456)
y_big_obs = [randn(M_big) for _ in 1:T_big]

grads_kf = gradient_kalman(y_big_obs, mu_0_big, Sigma_0_big, model_big)

println("Gradient of loglik w.r.t. μ₀:")
println(round.(grads_kf.grad_mu_0; digits = 4))
println("\nGradient of loglik w.r.t. A (first row):")
println(round.(grads_kf.grad_model.A[1, :]; digits = 4))
```

```{code-cell} julia
---
tags: [remove-cell]
---
@testset "Reverse AD Kalman" begin
    @test sum(abs, grads_kf.grad_mu_0) > 0
    @test !any(isnan, grads_kf.grad_mu_0)
    @test sum(abs, grads_kf.grad_model.A) > 0
    @test !any(isnan, grads_kf.grad_model.A)
end
```

### AD Correctness: EnzymeTestUtils

We validate forward and reverse modes against finite differences using `test_forward` and `test_reverse` on a small ($N=2$, $T=2$) model.

Note: `Sigma_0` must be `Const` due to aliasing with `Sigma[1]` through `copyto!!`.

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

test_forward(scalar_kalman!, Const,
             (mu_et, Duplicated),
             (Sigma_et, Duplicated),
             (y_test, Duplicated),
             (copy(mu_0_test), Const),
             (copy(Sigma_0_test), Const),
             (model_test, Duplicated),
             (cache_et, Duplicated))
```

```{code-cell} julia
test_reverse(scalar_kalman!, Const,
             (mu_et, Duplicated),
             (Sigma_et, Duplicated),
             (y_test, Duplicated),
             (copy(mu_0_test), Const),
             (copy(Sigma_0_test), Const),
             (model_test, Duplicated),
             (cache_et, Duplicated))
```
