using LinearAlgebra, Random, Test, Enzyme, Statistics, EnzymeTestUtils, BenchmarkTools

# Reverse-mode gradient computed over N + K*T = $(N) + $(K)*$(T) = $(N + K*T) parameters
# (x_0 has N elements, w has K*T elements)

@inline function chol_test!(L, A, C)
    mul!(A, C, C')  # C = A * A'
    copyto!(L, A)
    cholesky!(L, NoPivot(); check = false)
    return L, A, C
end

function chol_forward_test(L, A, C, dL, dA, dC)
    Enzyme.remake_zero!(dL)
    Enzyme.remake_zero!(dA)
    Enzyme.remake_zero!(dC)
    dC[1, 1] = 1.0  # Seed perturbation
    autodiff(Forward,
            chol_test!,
            Duplicated(L, dL),
            Duplicated(A, dA),
            Duplicated(C, dC))
    return nothing
end

C = rand(10, 10)
A = zeros(size(C))
L = zeros(size(C))
dL = Enzyme.make_zero(L)
dA = Enzyme.make_zero(A)
dC = Enzyme.make_zero(C)
@btime chol_forward_test($L, $A, $C, $dL, $dA, $dC)
test_forward(chol_test!,
    Const, 
    (L, Duplicated),
    (A, Duplicated), 
    (C, Duplicated)
)

@inline function scalar_chol!(L, A, C)
    mul!(A, C, C')  # C = A * A'
    copyto!(L, A)
    cholesky!(L, NoPivot(); check = false)
    return sum(diag(L))
end

L = zeros(size(C))
A = zeros(size(C))
C = rand(10, 10)
dL = Enzyme.make_zero(L)
dA = Enzyme.make_zero(A)
dC = Enzyme.make_zero(C)
chol_test!(L, A, C)
chol_forward_test(L, A, C, dL, dA, dC)
@btime chol_forward_test($L, $A, $C, $dL, $dA, $dC)

# Reverse
function chol_reverse_test(L, A, C, dL, dA, dC)
    Enzyme.remake_zero!(dL)
    Enzyme.remake_zero!(dA)
    Enzyme.remake_zero!(dC)
    autodiff(Reverse,
            scalar_chol!,
            Duplicated(L, dL),
            Duplicated(A, dA),
            Duplicated(C, dC))
    return nothing
end
chol_reverse_test(L, A, C, dL, dA, dC)
@btime chol_reverse_test($L, $A, $C, $dL, $dA, $dC)
test_reverse(scalar_chol!, 
    Const,
    (L, Duplicated),
    (A, Duplicated), 
    (C, Duplicated)
)

#########
# Stack in views tests for Enzyme

@inline function chols_test!(Ls, As, Cs)
    for t in axes(Cs, 3)
        @views mul!(As[:, :, t], Cs[:, :, t], Cs[:, :, t]')
        @views copyto!(Ls[:, :, t], As[:, :, t])
        @views cholesky!(Ls[:, :, t], NoPivot(); check = false)
    end
    return Ls, As, Cs
end

function chols_forward_test(Ls, As, Cs, dLs, dAs, dCs)
    Enzyme.remake_zero!(dLs)
    Enzyme.remake_zero!(dAs)
    Enzyme.remake_zero!(dCs)
    dCs[1, 1, 1] = 1.0
    autodiff(Forward,
            chols_test!,
            Duplicated(Ls, dLs),
            Duplicated(As, dAs),
            Duplicated(Cs, dCs))
    return nothing
end

T_chol = 4
Cs = rand(10, 10, T_chol)
As = zeros(size(Cs))
Ls = zeros(size(Cs))
dLs = Enzyme.make_zero(Ls)
dAs = Enzyme.make_zero(As)
dCs = Enzyme.make_zero(Cs)
chols_test!(Ls, As, Cs)
chols_forward_test(Ls, As, Cs, dLs, dAs, dCs)
@btime chols_forward_test($Ls, $As, $Cs, $dLs, $dAs, $dCs)
test_forward(chols_test!,
    Const,
    (Ls, Duplicated),
    (As, Duplicated),
    (Cs, Duplicated)
)


@inline function scalar_chols!(Ls, As, Cs)
    total = zero(eltype(Ls))
    @inbounds for t in axes(Cs, 3)
        @views Ct = Cs[:, :, t]
        @views At = As[:, :, t]
        @views Lt = Ls[:, :, t]
        mul!(At, Ct, Ct')
        copyto!(Lt, At)
        cholesky!(Lt, NoPivot(); check = false)
        total += sum(diag(Lt))
    end
    return total
end

function chols_reverse_test(Ls, As, Cs, dLs, dAs, dCs)
    Enzyme.remake_zero!(dLs)
    Enzyme.remake_zero!(dAs)
    Enzyme.remake_zero!(dCs)
    autodiff(Reverse,
            scalar_chols!,
            Duplicated(Ls, dLs),
            Duplicated(As, dAs),
            Duplicated(Cs, dCs))
    return nothing
end
chols_reverse_test(Ls, As, Cs, dLs, dAs, dCs)
@btime chols_reverse_test($Ls, $As, $Cs, $dLs, $dAs, $dCs)
test_reverse(scalar_chols!,
    Const,
    (Ls, Duplicated),
    (As, Duplicated),
    (Cs, Duplicated)
)

