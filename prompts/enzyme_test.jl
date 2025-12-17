using Enzyme, LinearAlgebra, EnzymeTestUtils

# 1. Define the Cache Struct
struct SimCache{T}
    tmp_vec::Vector{T}
    tmp_mat::Matrix{T}
end

# Helper to build it (good for users)
function alloc_cache(n)
    SimCache(zeros(n), zeros(n, n))
end

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

test_forward(simulate!, Const, 
    (out, Duplicated), 
    (x, Duplicated), 
    (cache, Duplicated)
)