using LinearAlgebra, Random, Plots, Test, Enzyme, Statistics

struct KFCache{T}
    mu_pred::Vector{T}
    innovation::Vector{T}
    innovation_solved::Vector{T}
    S::Matrix{T}
    K::Matrix{T}
    k_work::Matrix{T}
    Sigma_pred::Matrix{T}
    tmpKG::Matrix{T}
    tmpNM::Matrix{T}
    mu_update::Vector{T}
end

function alloc_cache(N, M)
    T = Float64
    return KFCache(
        zeros(T, N),
        zeros(T, M),
        zeros(T, M),
        zeros(T, M, M),
        zeros(T, N, M),
        zeros(T, M, N),
        zeros(T, N, N),
        zeros(T, N, N),
        zeros(T, N, M),
        zeros(T, N),
    )
end

# state and observation paths are stored column-wise for cache-friendly access
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

    @inbounds for i in 1:N
        x[i, 1] = x_0[i]
    end

    @inbounds for t in 1:T
        @views mul!(x[:, t + 1], A, x[:, t])
        @views mul!(x[:, t + 1], C, w[:, t], 1.0, 1.0)

        @views mul!(y[:, t], G, x[:, t])
        @views mul!(y[:, t], H, v[:, t], 1.0, 1.0)
    end

    @views mul!(y[:, T + 1], G, x[:, T + 1])
    @views mul!(y[:, T + 1], H, v[:, T + 1], 1.0, 1.0)

    return nothing
end

function kalman!(mu, Sigma, y, mu_0, Sigma_0, A, C, G, H, cache::KFCache)
    N, T1 = size(mu)
    M, T_obs = size(y)
    T = T_obs

    @inbounds for i in 1:N
        mu[i, 1] = mu_0[i]
    end
    @inbounds for i in 1:N, j in 1:N
        Sigma[i, j, 1] = Sigma_0[i, j]
    end

    loglik = zero(eltype(mu))

    @inbounds for t in 1:T
        # predict
        mul!(cache.mu_pred, A, @view mu[:, t])
        mul!(cache.tmpKG, A, @view Sigma[:, :, t])                        # A Σ
        mul!(cache.Sigma_pred, cache.tmpKG, transpose(A), 1.0, 0.0)       # A Σ A'
        mul!(cache.Sigma_pred, C, transpose(C), 1.0, 1.0)                 # + C C'

        # innovation: y_t - G mu_pred
        @inbounds for i in 1:M
            cache.innovation[i] = y[i, t]
        end
        mul!(cache.innovation, G, cache.mu_pred, -1.0, 1.0)

        # S = G Σ_pred G' + H H'
        mul!(cache.tmpNM, cache.Sigma_pred, transpose(G))                 # Σ_pred G'
        mul!(cache.S, G, cache.tmpNM)                                     # G Σ_pred G'
        mul!(cache.S, H, transpose(H), 1.0, 1.0)                          # + H H'

        # gain: K = Σ_pred G' S^{-1}
        # first solve S \ (Σ_pred G')' -> k_work = S \ tmpNM'
        @views for i in 1:M, j in 1:N
            cache.k_work[i, j] = cache.tmpNM[j, i]
        end
        @inbounds for i in 1:M
            cache.S[i, i] += 1e-8
        end
        F = cholesky!(Symmetric(cache.S, :U); check = false)
        ldiv!(F, cache.k_work)                                            # S \ tmpNM'
        @views for i in 1:N, j in 1:M
            cache.K[i, j] = cache.k_work[j, i]
        end

        # update mean: mu = mu_pred + K * innovation
        mul!(cache.mu_update, cache.K, cache.innovation, 1.0, 0.0)
        @inbounds for i in 1:N
            mu[i, t + 1] = cache.mu_pred[i] + cache.mu_update[i]
        end

        # update covariance: (I - K G) Σ_pred
        mul!(cache.tmpKG, cache.K, G)                                     # K G
        @views Sigma[:, :, t + 1] .= cache.Sigma_pred
        @views mul!(Sigma[:, :, t + 1], cache.tmpKG, cache.Sigma_pred, -1.0, 1.0)

        # log likelihood contribution: -0.5 (M log(2π) + logdet(S) + ν' S^{-1} ν)
        copyto!(cache.innovation_solved, cache.innovation)
        ldiv!(F, cache.innovation_solved)
        logdetS = 2sum(log, diag(F.U))
        quad = dot(cache.innovation_solved, cache.innovation_solved)
        loglik -= 0.5 * (M * log(2π) + logdetS + quad)
    end

    return loglik
end

# Example data
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

mu_0 = randn(N)
Sigma_0 = Matrix{Float64}(I, N, N)
w = randn(K, T)
v = randn(L, T + 1)
y = zeros(M, T + 1)

# simulate data to filter against
simulate_lss!(zeros(N, T + 1), y, A, C, G, H, mu_0, w, v)
y_obs = copy(@view y[:, 1:T])

mu = zeros(N, T + 1)
Sigma = zeros(N, N, T + 1)
cache = alloc_cache(N, M)

loglik = kalman!(mu, Sigma, y_obs, mu_0, Sigma_0, A, C, G, H, cache)

mu[:, end], loglik

# Forward-mode sensitivity of final mean to A[1,1]
Tobs = size(y_obs, 2)
mu_buf = zeros(N, Tobs + 1)
Sigma_buf = zeros(N, N, Tobs + 1)
cache_buf = alloc_cache(N, M)

function final_mean_sum(A_param, y_obs, mu0, Sigma0, C, G, H, mu_buf, Sigma_buf, cache_buf)
    @views mu_buf .= 0.0
    @views Sigma_buf .= 0.0
    kalman!(mu_buf, Sigma_buf, y_obs, mu0, Sigma0, A_param, C, G, H, cache_buf)
    return sum(@view mu_buf[:, end])
end

dA = Enzyme.make_zero(A)
dA[1, 1] = 1.0
mu_buf_shadow = Enzyme.make_zero(mu_buf)
Sigma_buf_shadow = Enzyme.make_zero(Sigma_buf)
cache_shadow = Enzyme.make_zero(cache_buf)
d_final_A = autodiff(
    Forward,
    final_mean_sum,
    Duplicated(A, dA),
    Const(y_obs),
    Const(mu_0),
    Const(Sigma_0),
    Const(C),
    Const(G),
    Const(H),
    Duplicated(mu_buf, mu_buf_shadow),
    Duplicated(Sigma_buf, Sigma_buf_shadow),
    Duplicated(cache_buf, cache_shadow),
)

# Reverse-mode sensitivity of final mean to mu_0
mu_rev = Enzyme.make_zero(mu_0)
autodiff(
    Reverse,
    (mu0_param, y_obs, Sigma0, A, C, G, H, mu_buf, Sigma_buf, cache_buf) ->
        final_mean_sum(A, y_obs, mu0_param, Sigma0, C, G, H, mu_buf, Sigma_buf, cache_buf),
    Duplicated(mu_0, mu_rev),
    Const(y_obs),
    Const(Sigma_0),
    Const(A),
    Const(C),
    Const(G),
    Const(H),
    Duplicated(mu_buf, mu_buf_shadow),
    Duplicated(Sigma_buf, Sigma_buf_shadow),
    Duplicated(cache_buf, cache_shadow),
)
mu_rev
