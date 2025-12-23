using LinearAlgebra, Random, Test, Enzyme, Statistics, EnzymeTestUtils, BenchmarkTools

# Note on inline:
# Sometimes a little speed, but a lot of it is to help the AD system reason, especially
# when using composite types like the model in `p`.  The challenge is that
# otherwise it has trouble with having parts of `p` be Const and parts Duplicated.

# =============================================================================
# Generic state-space model simulator (in-place)
# =============================================================================

@inline function simulate_ssm!(x, y, f!, g!, x_0, w, v, p)
    T = size(w, 2)
    @assert length(x_0) == size(x, 1)
    @assert size(x, 2) == T + 1
    @assert size(y, 2) == T + 1
    @assert size(v, 2) == T + 1

    # Works for both Duplicated and Const x_0
    @inbounds for i in eachindex(x_0)
        x[i, 1] = x_0[i]
    end

    @inbounds for t in 1:T
        @views f!(x[:, t + 1], x[:, t], w[:, t], p, t - 1)
        @views g!(y[:, t], x[:, t], v[:, t], p, t - 1)
    end

    # Apply observation equation at final time
    @views g!(y[:, T + 1], x[:, T + 1], v[:, T + 1], p, T)

    return nothing
end

# =============================================================================
# LSS-specific hooks for generic SSM simulator (in-place)
# =============================================================================

@inline function f_lss!(x_p, x, w, p, t)
    mul!(x_p, p.A, x) # x_{t+1} = A x_t
    mul!(x_p, p.C, w, 1.0, 1.0) # + C w_{t+1}
    return nothing
end

@inline function g_lss!(y, x, v, p, t)
    mul!(y, p.G, x) # y_t = G x_t
    mul!(y, p.H, v, 1.0, 1.0) # + H v_t
    return nothing
end

# =============================================================================
# Out-of-place versions for comparison
# =============================================================================

@inline function f_lss_oop(x, w, p, t)
    return p.A * x + p.C * w
end

@inline function g_lss_oop(x, v, p, t)
    return p.G * x + p.H * v
end

@inline function simulate_ssm_oop!(x, y, f, g, x_0, w, v, p)
    T = size(w, 2)
    @assert length(x_0) == size(x, 1)
    @assert size(x, 2) == T + 1
    @assert size(y, 2) == T + 1
    @assert size(v, 2) == T + 1

    @inbounds for i in eachindex(x_0)
        x[i, 1] = x_0[i]
    end

    @inbounds for t in 1:T
        @views x[:, t + 1] = f(x[:, t], w[:, t], p, t - 1)
        @views y[:, t] = g(x[:, t], v[:, t], p, t - 1)
    end

    # Apply observation equation at final time
    @views y[:, T + 1] = g(x[:, T + 1], v[:, T + 1], p, T)

    return nothing
end

# =============================================================================
# Hand-built LSS simulator for comparison (in-place)
# =============================================================================

function simulate_lss!(x, y, model, x_0, w, v)
    (; A, C, G, H) = model
    N, T1 = size(x)
    M, T1y = size(y)
    T = size(w, 2)

    # copy will promote activity
    @inbounds for i in eachindex(x_0)
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

# =============================================================================
# Model parameters and initial conditions
# =============================================================================

N = 5
M = 2
K = 2
L = 2
T = 100

A = rand(N, N) .- 0.5
println("Spectral radius of A: ", maximum(abs, eigvals(A)))
C = 0.1 .* rand(N, K)
G = rand(M, N)
H = 0.05 .* rand(M, L)
model = (; A, C, G, H)

x_0 = rand(N)
w = rand(K, T)
v = rand(L, T + 1)

# =============================================================================
# Consistency tests
# =============================================================================

# Consistency test: simulate_ssm! should match hand-built simulate_lss!
x_lss = zeros(N, T + 1)
y_lss = zeros(M, T + 1)
simulate_lss!(x_lss, y_lss, model, x_0, w, v)

x_ssm = zeros(N, T + 1)
y_ssm = zeros(M, T + 1)
simulate_ssm!(x_ssm, y_ssm, f_lss!, g_lss!, x_0, w, v, model)

@test isapprox(x_ssm, x_lss; rtol=1e-12, atol=0.0)
@test isapprox(y_ssm, y_lss; rtol=1e-12, atol=0.0)

# =============================================================================
# AD setup: preallocate shadow arrays
# =============================================================================

# Preallocate shadow arrays for AD
x = zeros(N, T + 1)
y = zeros(M, T + 1)
dx = Enzyme.make_zero(x)
dx_0 = Enzyme.make_zero(x_0)
dy = Enzyme.make_zero(y)
dw = Enzyme.make_zero(w)
dw[1] = 1.0  # Unit perturbation to first shock for forward-mode

# =============================================================================
# Forward-mode AD test functions
# =============================================================================

function test_AD_lss(x, y, model, x_0, w, v, dx, dy, dw)
    Enzyme.remake_zero!(dx)
    Enzyme.remake_zero!(dy)
    Enzyme.remake_zero!(dw)
    autodiff(Forward,
            simulate_lss!,
            Duplicated(x, dx),
            Duplicated(y, dy),
            Const(model),
            Const(x_0),
            Duplicated(w, dw),
            Const(v))
    return dx[:, 1:3]  # Early-state sensitivities
end

function test_AD_ssm(x, y, model, x_0, w, v, dx, dy, dw)
    Enzyme.remake_zero!(dx)
    Enzyme.remake_zero!(dy)
    Enzyme.remake_zero!(dw)
    autodiff(Forward,
            simulate_ssm!,
            Duplicated(x, dx),
            Duplicated(y, dy),
            Const(f_lss!),
            Const(g_lss!),
            Const(x_0),
            Duplicated(w, dw),
            Const(v),
            Const(model))

    return dx[:, 1:3]
end

test_AD_lss(x, y, model, x_0, w, v, dx, dy, dw)
test_AD_ssm(x, y, model, x_0, w, v, dx, dy, dw)

# =============================================================================
# Reverse-mode AD scalar wrapper functions and test functions
# =============================================================================

function scalar_lss(x, y, model, x_0, w, v)
    simulate_lss!(x, y, model, x_0, w, v)
    return mean(x) + mean(y)
end

function scalar_ssm(x, y, f!, g!, x_0, w, v, p)
    simulate_ssm!(x, y, f!, g!, x_0, w, v, p)
    return mean(x) + mean(y)
end

scalar_lss(x, y, model, x_0, w, v)
scalar_ssm(x, y, f_lss!, g_lss!, x_0, w, v, model)

# Reverse-mode AD test functions
function test_ADrev_lss(x, y, model, x_0, w, v, dx, dy, dx_0, dw)
    Enzyme.remake_zero!(dx)
    Enzyme.remake_zero!(dy)
    Enzyme.remake_zero!(dx_0)
    Enzyme.remake_zero!(dw)
    autodiff(Reverse,
            scalar_lss,
            Duplicated(x, dx),
            Duplicated(y, dy),
            Const(model),
            Duplicated(x_0, dx_0),
            Duplicated(w, dw),
            Const(v))
    return sum(abs, dx) + sum(abs, dy) + sum(abs, dx_0) + sum(abs, dw)
end

function test_ADrev_ssm(x, y, model, x_0, w, v, dx, dy, dx_0, dw)
    Enzyme.remake_zero!(dx)
    Enzyme.remake_zero!(dy)
    Enzyme.remake_zero!(dx_0)
    Enzyme.remake_zero!(dw)
    autodiff(Reverse,
            scalar_ssm,
            Duplicated(x, dx),
            Duplicated(y, dy),
            Const(f_lss!),
            Const(g_lss!),
            Duplicated(x_0, dx_0),
            Duplicated(w, dw),
            Const(v),
            Const(model))
    return sum(abs, dx) + sum(abs, dy) + sum(abs, dx_0) + sum(abs, dw)
end

dx_0_rev = Enzyme.make_zero(x_0)
test_ADrev_lss(x, y, model, x_0, w, v, dx, dy, dx_0_rev, dw)
test_ADrev_ssm(x, y, model, x_0, w, v, dx, dy, dx_0_rev, dw)

# =============================================================================
# Out-of-place versions: preallocate and warmup
# =============================================================================

x_oop = zeros(N, T + 1)
y_oop = zeros(M, T + 1)
dx_oop = Enzyme.make_zero(x_oop)
dy_oop = Enzyme.make_zero(y_oop)
dw_oop = Enzyme.make_zero(w)
dx0_oop = Enzyme.make_zero(x_0)

# Out-of-place scalar wrapper for reverse-mode
function scalar_ssm_oop(x, y, f, g, x_0, w, v, p)
    simulate_ssm_oop!(x, y, f, g, x_0, w, v, p)
    return mean(x) + mean(y)
end

function test_AD_oop(x, y, model, x_0, w, v, dx, dy, dw)
    Enzyme.remake_zero!(dx)
    Enzyme.remake_zero!(dy)
    Enzyme.remake_zero!(dw)
    autodiff(Forward,
            simulate_ssm_oop!,
            Duplicated(x, dx),
            Duplicated(y, dy),
            Const(f_lss_oop),
            Const(g_lss_oop),
            Const(x_0),
            Duplicated(w, dw),
            Const(v),
            Const(model))
    return dx[:, 1:3]
end

function test_ADrev_oop(x, y, model, x_0, w, v, dx, dy, dx_0, dw)
    Enzyme.remake_zero!(dx)
    Enzyme.remake_zero!(dy)
    Enzyme.remake_zero!(dx_0)
    Enzyme.remake_zero!(dw)
    autodiff(Reverse,
            scalar_ssm_oop,
            Duplicated(x, dx),
            Duplicated(y, dy),
            Const(f_lss_oop),
            Const(g_lss_oop),
            Duplicated(x_0, dx_0),
            Duplicated(w, dw),
            Const(v),
            Const(model))
    return sum(abs, dx) + sum(abs, dy) + sum(abs, dx_0) + sum(abs, dw)
end

simulate_ssm_oop!(x_oop, y_oop, f_lss_oop, g_lss_oop, x_0, w, v, model)
test_AD_oop(x_oop, y_oop, model, x_0, w, v, dx_oop, dy_oop, dw_oop)
test_ADrev_oop(x_oop, y_oop, model, x_0, w, v, dx_oop, dy_oop, dx0_oop, dw_oop)

# =============================================================================
# Benchmark comparisons grouped by task
# =============================================================================

println("\nSimulation: lss")
@btime simulate_lss!($x_lss, $y_lss, $model, $x_0, $w, $v);
println("Simulation: ssm")
@btime simulate_ssm!($x_ssm, $y_ssm, f_lss!, g_lss!, $x_0, $w, $v, $model);
println("Simulation: ssm_oop")
@btime simulate_ssm_oop!($x_oop, $y_oop, f_lss_oop, g_lss_oop, $x_0, $w, $v, $model);

println("")
println("Forward AD: lss")
@btime test_AD_lss($x, $y, $model, $x_0, $w, $v, $dx, $dy, $dw);
println("Forward AD: ssm")
@btime test_AD_ssm($x, $y, $model, $x_0, $w, $v, $dx, $dy, $dw);
println("Forward AD: ssm_oop")
@btime test_AD_oop($x_oop, $y_oop, $model, $x_0, $w, $v, $dx_oop, $dy_oop, $dw_oop);

println("")
println("Reverse AD: lss")
@btime test_ADrev_lss($x, $y, $model, $x_0, $w, $v, $dx, $dy, $dx_0_rev, $dw);
println("Reverse AD: ssm")
@btime test_ADrev_ssm($x, $y, $model, $x_0, $w, $v, $dx, $dy, $dx_0_rev, $dw);
println("Reverse AD: ssm_oop")
@btime test_ADrev_oop($x_oop, $y_oop, $model, $x_0, $w, $v, $dx_oop, $dy_oop, $dx0_oop, $dw_oop);