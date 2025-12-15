# Instructions for Differentiable Code
In general write clean code without worrying about small performance changes.  However, in some cases we need to maintain strict control over memory allocations, especially if I mention you should use Enzyme.

In that case, you are an expert Julia developer specializing in high-performance computing and zero-allocation patterns. When writing code, prioritize strict memory management and in-place operations. Adhere to the following guidelines`:

Our main target will be Enzyme.jl and occasionally ForwardDiff.jl.

## Enzyme.jl Guidelines
- See https://enzymead.github.io/Enzyme.jl/dev/ for enzyme documentation
- In cases when you need to write custom rules, EnzymeTestUtils provides a way to test against FiniteDiff automatically.
  - See https://enzymead.github.io/Enzyme.jl/dev/api/#EnzymeTestUtils.test_forward-Tuple%7BAny,%20Any,%20Vararg%7BAny%7D%7D
  - https://enzymead.github.io/Enzyme.jl/dev/api/#EnzymeTestUtils.test_reverse-Tuple{Any,%20Any,%20Vararg{Any}}
- You need to write inplace code without allocations.  See `Inplace Coding Guidelines` below
- Functions should be written to be type-stable
- You will need to preallocate any buffers and pass as arguments.

## Enzyme & Buffer Conventions
When writing the Enzyme Differentiable Julia code, adhere to these SciML/Enzyme patterns:

1.  **Function Signature:** `function name!(out, inputs..., params, cache)`. Place the `cache` (workspace) argument last.
2.  **Cache Structs:** Do not pass individual arrays for buffers. Create a `struct NameCache` to hold all pre-allocated temporary arrays.
3.  **Enzyme Compatibility:** * Ensure the code is compatible with `Enzyme.autodiff`. 
    * Assume the user will pass `Duplicated(cache, d_cache)` for the workspace.
    * The `cache` struct must contain mutable fields (Arrays) that match the precision of the inputs.
4.  **No Internal Allocation:** Strictly use the fields in `cache` for all intermediate vector/matrix operations.
5.  **Use make_zero**: Use `Enzyme.make_zero(x)` to create shadow copies for gradients of inputs/outputs/cache rather than manual allocation.
6.  **Enzyme Assignment Loops** Sometimes Enzyme hits an Activity Analysis issues with assignment, for example instad of `x[:,1] .= x_0` or equivalent things with view, it needs.  In that case you can use manual loop for assignment, but put a comment to explain why
    ```julia
    # Enzyme has challenges with activity analysis on broadcasting assignments
    @inbounds for i in 1:N
            x[i, 1] = x_0[i]
    end
    ```


Follow the coding pattern in this example, for example.
```
using Enzyme, LinearAlgebra

# 1. Define the Cache Struct
struct SimCache{T}
    tmp_vec::Vector{T}
    tmp_mat::Matrix{T}
end

# Helper to build it (good for users)
alloc_cache(n) = SimCache(zeros(n), zeros(n, n))

# 2. The In-Place Function
# Order: out, x (input), cache (buffers)
function simulate!(out, x, cache::SimCache)
    # unpack for clarity
    tmp = cache.tmp_vec
    
    # Example: Intermediate computation that needs a buffer
    # If we didn't have cache, this line would allocate
    @. tmp = x * x + 2
    
    # Write to out
    @. out = tmp^2
    return nothing
end

# 3. Setup for Enzyme
n = 100
x = rand(n)
out = zeros(n)
cache = alloc_cache(n)

# Create "Shadows" (gradients)
# Enzyme.make_zero creates a structural copy with all zeros
dx = Enzyme.make_zero(x)
dout = Enzyme.make_zero(out)     # "Seed" for the gradient (e.g., set to 1s if scalar)
dcache = Enzyme.make_zero(cache) # The shadow workspace!

# Initialize the seed (e.g., we want gradient of sum(out))
dout .= 1.0 

# 4. The Call
# Note: We pass Duplicated for the cache because it is MUTATED.
autodiff(Reverse, simulate!, 
    Duplicated(out, dout), 
    Duplicated(x, dx), 
    Duplicated(cache, dcache)
)
```

### Testing to ensure Type Safety and No Allocations
- For any Enzyme-differentiable function, you should add asserts during development to ensure type stability and no allocations.
- With the `Test` package, you can use `@inferred` to check type stability and `@allocated` to check allocations.

For example:

```julia
using Test

function stable_func(x)
    return x * 2.0
end

function unstable_func(x)
    # Returns Int if x > 0, Float64 otherwise (Type Unstable!)
    x > 0 ? 1 : 1.0 
end

@testset "Type Stability" begin
    x = 1.0
    
    # PASSES: Compiler infers Float64, actual is Float64
    @inferred stable_func(x)
    
    # FAILS: Compiler infers Union{Float64, Int64}, actual is one of them
    # Throws: "return type Union{Float64, Int64} does not match inferred type..."
    @inferred unstable_func(x)
end
function compute!(out, x)
    # Example: Allocation-free operation
    out .= x .* 2
    return nothing
end

@testset "Allocations" begin
    out = zeros(10)
    x = rand(10)

    # 1. WARMUP: Run once to force compilation
    compute!(out, x)

    # 2. ASSERT: Check allocations are exactly 0
    # Note: We interpolate arguments ($) inside benchmark tools, 
    # but strictly speaking @allocated just runs the expression.
    allocs = @allocated compute!(out, x)
    
    @test allocs == 0
end
```


### Testing with Finite Differences
In cases where a custom rule is required, or there is a concern that gradients may be incorrect, use EnzymeTestUtils to compare against finite differences.  Follow this example, which extends the above code.

Even if you only care about the gradient of `x`, you **must** mark the `out` and `cache` arguments as `Duplicated` because they are mutated.

**Example: Testing Reverse Mode**
```julia
using Enzyme, EnzymeTestUtils, Test

# 1. Setup Data
x = randn(10)
out = zeros(10)
cache = alloc_cache(10) # User-defined helper to build cache struct

# 2. Test Gradient of `x`
# We use `test_reverse` to compare AD against Finite Differences.
# Note: `out` and `cache` are mutated, so they MUST be Duplicated.
test_reverse(simulate!, Const, 
    (out, Duplicated),   # Mutated Output
    (x, Duplicated),     # Input (Active for differentiation)
    (cache, Duplicated)  # Mutated Workspace
)
# Test Tangent of `x`
test_forward(simulate!, Const, 
    (out, Duplicated), 
    (x, Duplicated), 
    (cache, Duplicated)
)
```

# Inplace Coding Guidelines

In that case, you are an expert Julia developer specializing in high-performance computing and zero-allocation patterns. When writing code, prioritize strict memory management and in-place operations. Adhere to the following guidelines:

### 1. Matrix Multiplication (`mul!`)
Avoid `*` for matrix operations when a destination buffer exists. Use `LinearAlgebra.mul!`.
- **Matrix-Matrix:** `mul!(C, A, B)` computes $C = AB$ overwriting $C$.
- **Matrix-Vector:** `mul!(y, A, x)` computes $y = Ax$ overwriting $y$.
- **5-Argument Interface:** Use `mul!(Y, A, B, \alpha, \beta)` to compute $Y = \alpha AB + \beta Y$. This is a "muladd" operation that avoids allocating a temporary buffer for $AB$ before adding it to $Y$.

### 2. In-Place Linear Solves (`ldiv!`)
Avoid `\` (backslash) inside hot loops.
- Pre-factorize matrices (e.g., LU, Cholesky) outside the loop: `F = lu(A)` or `F = cholesky(A)`.
- Use `ldiv!(F, b)` to solve $Ax=b$ in-place, overwriting vector `b` with the solution.
- Use `ldiv!(y, F, x)` if you need to keep `x` intact and write the solution to `y`.

### 3. Quadratic Forms (`dot`)
Avoid `x' * A * x` or `(x' * A) * y`. These create temporary vector allocations.
- Use the 3-argument dot product: `dot(x, A, y)` computes $x^T A y$ without allocating the intermediate vector $Ay$.
- For quadratic forms $x^T A x$, use `dot(x, A, x)`.

### 4. Symmetrization (Kalman Filters/Covariances)
In algorithms like Kalman filtering, numerical errors destroy symmetry.
- **Preferred:** Wrap the matrix in `Symmetric(A)` or `Hermitian(A)` rather than forcing data movement.
- **In-Place Enforcement:** If you must materialize the symmetric matrix (e.g., before a LAPACK call that accesses the full matrix), calculate one triangle and copy it to the other using `LinearAlgebra.copytri!(A, 'U')` (copies Upper to Lower) or `'L'` (Lower to Upper).
- **Averaging:** If you specifically need $A \leftarrow (A + A^T)/2$ to average errors, use an explicit loop or broadcast into a pre-allocated buffer to avoid creating a new matrix.

### 5. Array Views
Slicing arrays (`A[1:5, :]`) creates a copy.
- Use macros: `@views A[1:5, :]` creates a `SubArray` view automatically.
- Use function calls: `view(A, :, 1)` for specific programmatic control.
- **Caveat:** Be careful with views in dense linear algebra (BLAS); if the memory stride is non-unit, performance may degrade. However, strictly for avoiding allocation, views are required.
- Organize algorithms around column vs. row access patterns to maximize contiguous memory access when using views.

### 6. Transposition
Avoid `B = A'` unless it is a wrapper type without materialization.
- Use `transpose!(B, A)` or `adjoint!(B, A)` to move data into a pre-allocated buffer `B`.

### 7. Broadcasting and `map!`
- Use `@.` when possible for clarity
- Use `broadcast!(f, dest, args...)` or the syntactic sugar `dest .= f.(args...)` to fuse operations and write directly to `dest`.
- Use `map!(f, dest, src)` for element-wise mapping.

### 8. General "Buffer" Management
- When using Enzyme, use in-place variant `f!(out, x)` if there is a vector to be returned. Put the last argument as mutable buffers where required.
- Pass pre-allocated "workspace" vectors/matrices into functions as arguments rather than creating them inside the function body.
