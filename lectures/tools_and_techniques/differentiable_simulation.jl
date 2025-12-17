using LinearAlgebra, Random, Plots, Test, Enzyme, Statistics

struct LSS{TA, TC, TG, TH}
    A::TA
    C::TC
    G::TG
    H::TH
end

# state and observation paths are stored column-wise for cache-friendly access
function simulate_lss!(x, y, model::LSS, x_0, w, v)
    (; A, C, G, H) = model
    N, x_cols = size(x)
    M, y_cols = size(y)
    T = size(w, 2)

    @assert size(A) == (N, N)
    @assert size(C, 1) == N
    @assert size(G) == (M, N)
    @assert size(H, 1) == M
    @assert size(w, 1) == size(C, 2)
    @assert size(v, 1) == size(H, 2)
    @assert size(v, 2) == T + 1
    @assert x_cols == T + 1
    @assert y_cols == T + 1
    @assert length(x_0) == N
    
    # Enzyme has challenges with activity analysis on broadcasting assignments
    @inbounds for i in 1:N
        x[i, 1] = x_0[i]
    end

    @inbounds for t in 1:T
        @views mul!(x[:, t + 1], A, x[:, t])
        # x_{t+1} = A x_t + C w_{t+1}; (1.0, 1.0) adds C w_{t+1} into the existing A x_t
        @views mul!(x[:, t + 1], C, w[:, t], 1.0, 1.0)

        @views mul!(y[:, t], G, x[:, t])
        # y_t = G x_t + H v_t; (1.0, 1.0) accumulates H v_t into the G x_t buffer
        @views mul!(y[:, t], H, v[:, t], 1.0, 1.0)
    end

    @views mul!(y[:, T + 1], G, x[:, T + 1])
    # y_{T+1} = G x_{T+1} + H v_{T+1}
    @views mul!(y[:, T + 1], H, v[:, T + 1], 1.0, 1.0)

    return nothing
end

function mean_first_observation(x, y, model, x_0, w, v)
    simulate_lss!(x, y, model, x_0, w, v)
    out = mean(@view y[1, :])
    return out
end

Random.seed!(1234)

N = 3
M = 2
K = 2
L = 2
T = 10

A = [0.8 0.1 0.0
     0.0 0.7 0.1
     0.0 0.0 0.6]
C = 0.1 .* randn(N, K)
G = [1.0 0.0 0.0
     0.0 1.0 0.3]
H = 0.05 .* randn(M, L)
model = LSS(A, C, G, H)

x_0 = randn(N)
w = randn(K, T)
v = randn(L, T + 1)

x = zeros(N, T + 1)
y = zeros(M, T + 1)

simulate_lss!(x, y, model, x_0, w, v)

# Forward-mode sensitivity of full state/observation paths to w[1]
x_w = zeros(N, T + 1)
y_w = zeros(M, T + 1)
dx_w = Enzyme.make_zero(x_w)
dy_w = Enzyme.make_zero(y_w)
dw = Enzyme.make_zero(w)
dw[1] = one(eltype(w))
autodiff(
    Forward,
    simulate_lss!,
    Duplicated(x_w, dx_w),
    Duplicated(y_w, dy_w),
    Const(model),
    Const(x_0),
    Duplicated(w, dw),
    Const(v),
)

dw_rev = zero(w)
x_rev = zeros(N, T + 1)
y_rev = zeros(M, T + 1)
dx_rev = Enzyme.make_zero(x_rev)
dy_rev = Enzyme.make_zero(y_rev)

autodiff(
    Reverse,
    mean_first_observation,
    Duplicated(x_rev, dx_rev),
    Duplicated(y_rev, dy_rev),
    Const(model),
    Const(x_0),
    Duplicated(w, dw_rev),
    Const(v),
)

@test all(isfinite, dw_rev)

@testset "batch tangents for A" begin
    x_batch = zeros(N, T + 1)
    y_batch = zeros(M, T + 1)
    dxs = (Enzyme.make_zero(x_batch), Enzyme.make_zero(x_batch))
    dys = (Enzyme.make_zero(y_batch), Enzyme.make_zero(y_batch))
    dmodels = (Enzyme.make_zero(model), Enzyme.make_zero(model))
    dmodels[1].A[1] = one(eltype(A))
    dmodels[2].A[2] = one(eltype(A))

    autodiff(
        Forward,
        simulate_lss!,
        BatchDuplicated(x_batch, dxs),
        BatchDuplicated(y_batch, dys),
        BatchDuplicated(model, dmodels),
        Const(x_0),
        Const(w),
        Const(v),
    )

    @test maximum(abs, dxs[1]) > 0
    @test maximum(abs, dys[1]) > 0
end

time = 0:T
fig_states = plot(
    time,
    x',
    lw = 2,
    xlabel = "t",
    ylabel = "state",
    label = ["x1", "x2", "x3"],
    title = "State Paths",
)

fig_observations = plot(
    time,
    y',
    lw = 2,
    xlabel = "t",
    ylabel = "observation",
    label = ["y1", "y2"],
    title = "Observation Paths",
)

display(fig_states)
display(fig_observations)
