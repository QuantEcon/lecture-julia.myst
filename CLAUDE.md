# Claude Code Instructions

This repository builds QuantEcon Julia lectures using JupyterBook + MyST Markdown + MyST-NB execution.

## Quick Reference

**Build lectures:**
```bash
source .venv/bin/activate
jb build lectures
```

**Run Julia directly:**
```bash
julia --project=lectures
```

**Error logs location:**
```
lectures/_build/html/reports/<subpath>/<notebook>.err.log
```

## Auto-Approved Commands

The following commands are safe to run without confirmation:
- `source .venv/bin/activate`
- `jb build lectures`
- `julia --project=lectures`
- Reading/viewing files in `lectures/_build/html/reports/`

## Workflow: After Editing Lectures

After editing any lecture `.md` file, always verify the build:
1. Activate environment: `source .venv/bin/activate`
2. Build: `jb build lectures`
3. Check for warnings/errors in output
4. If failures occur, inspect logs in `lectures/_build/html/reports/`

---

## Julia Code Style

Follow the [SciML Style Guide](https://github.com/SciML/SciMLStyle) and local [style.md](style.md). We use `sciml` style in JuliaFormatter.

### Naming and Functions
- Non-mutating vs. mutating: `f` vs. `f!` vs. `f!!` (in-place without out-of-place fallback)
- Use short-form `f(x) = x^2` for one-liners
- **Never** use `begin` blocks for function definitions
- **Never** use `f = x -> x^2` unless the lambda is not bound to a name
- Explicit `using` statements
- Use fixed seeds for reproducibility

### Control Flow and Loops
- Prefer `if` conditionals over `&&`: `if phi > remaining; continue; end` not `phi > remaining && continue`
- Prefer `for i in 1:N` over `for i = 1:N`

### Performance
- Use `@inbounds` and `@views` where appropriate
- Use broadcasting, comprehensions, vectorization where appropriate

### Parameters and Return Values
Following SciML style, use a standardized `p` argument for parameters:
```julia
f(x, p) = p.alpha * x^2  # p is a named tuple
```

**Named tuples over structs for parameters:**
```julia
p = (; alpha = 0.5, n = 10)
```

**Unpack in functions:**
```julia
function f(x, p)
    (; alpha, n) = p
    # ...
end
```

**Use shorthand for return named tuples:**
```julia
# Good
function f(x)
    a = x^2
    b = x + 1
    return (; a, b)
end

# Avoid
function f(x)
    a = x^2
    b = x + 1
    return (; a = a, b = b)
end
```

### Variable Naming
Reuse variable names across scopes where possible:
```julia
# Good
f(x) = x^2
x = 5
y = f(x)

# Avoid
f(x) = x^2
x_val = 5
y = f(x_val)
```

When unsure, imitate existing notebooks in `lectures/`.

---

## Differentiable Code (Enzyme.jl)

In general, write clean code without worrying about small performance changes. However, when strict control over memory allocations is needed (especially for Enzyme), prioritize zero-allocation patterns.

**Primary AD targets:** Enzyme.jl (main), ForwardDiff.jl (occasional)

**Documentation:** https://enzymead.github.io/Enzyme.jl/dev/

### Core Requirements
- Write in-place code without allocations
- Functions must be type-stable
- Preallocate all buffers and pass as arguments

### Function Signature Convention
```julia
function name!(out, inputs..., params, cache)
```
Place the `cache` (workspace) argument **last**.

### Cache Structs
Do not pass individual arrays for buffers. Create a struct to hold all pre-allocated temporaries:
```julia
struct SimCache{T}
    tmp_vec::Vector{T}
    tmp_mat::Matrix{T}
end

alloc_cache(n) = SimCache(zeros(n), zeros(n, n))
```

### Enzyme Compatibility
- Ensure code is compatible with `Enzyme.autodiff`
- User will pass `Duplicated(cache, d_cache)` for workspace
- Cache struct must contain mutable fields (Arrays) matching input precision
- Use `Enzyme.make_zero(x)` to create shadow copies for gradients

### Activity Analysis Workaround
Enzyme sometimes has issues with broadcasting assignments. Use manual loops with a comment:
```julia
# Enzyme has challenges with activity analysis on broadcasting assignments
@inbounds for i in 1:N
    x[i, 1] = x_0[i]
end
```

### Runtime Activity Errors
**Never** fix "Detected potential need for runtime activity" errors by enabling runtime activity flags. Always find a code modification that avoids the error. Common fixes:
- Replace broadcasting assignments with explicit loops (see above)
- Ensure all array operations use pre-allocated buffers
- Avoid closures that capture mutable state
- Make type annotations explicit where inference fails

### Complete Enzyme Example
```julia
using Enzyme, LinearAlgebra

struct SimCache{T}
    tmp_vec::Vector{T}
    tmp_mat::Matrix{T}
end
alloc_cache(n) = SimCache(zeros(n), zeros(n, n))

function simulate!(out, x, cache::SimCache)
    tmp = cache.tmp_vec
    @. tmp = x * x + 2
    @. out = tmp^2
    return nothing
end

# Setup
n = 100
x = rand(n)
out = zeros(n)
cache = alloc_cache(n)

# Create shadows with make_zero
dx = Enzyme.make_zero(x)
dout = Enzyme.make_zero(out)
dcache = Enzyme.make_zero(cache)

dout .= 1.0  # Seed for gradient of sum(out)

# Autodiff call - cache MUST be Duplicated because it's mutated
autodiff(Reverse, simulate!,
    Duplicated(out, dout),
    Duplicated(x, dx),
    Duplicated(cache, dcache))
```

### Testing Type Stability and Allocations
```julia
using Test

@testset "Type Stability" begin
    @inferred stable_func(1.0)  # Passes if return type is inferred
end

@testset "Allocations" begin
    out = zeros(10)
    x = rand(10)
    compute!(out, x)  # Warmup
    @test @allocated(compute!(out, x)) == 0
end
```

### Testing with Finite Differences
Use EnzymeTestUtils to compare AD against finite differences. Mark `out` and `cache` as `Duplicated` even if you only care about `x` gradient:
```julia
using Enzyme, EnzymeTestUtils, Test

x = randn(10)
out = zeros(10)
cache = alloc_cache(10)

# Test reverse mode
test_reverse(simulate!, Const,
    (out, Duplicated),
    (x, Duplicated),
    (cache, Duplicated))

# Test forward mode
test_forward(simulate!, Const,
    (out, Duplicated),
    (x, Duplicated),
    (cache, Duplicated))
```

**EnzymeTestUtils docs:**
- [test_forward](https://enzymead.github.io/Enzyme.jl/dev/api/#EnzymeTestUtils.test_forward-Tuple%7BAny,%20Any,%20Vararg%7BAny%7D%7D)
- [test_reverse](https://enzymead.github.io/Enzyme.jl/dev/api/#EnzymeTestUtils.test_reverse-Tuple{Any,%20Any,%20Vararg{Any}})

---

## In-Place Coding Guidelines

For zero-allocation code (required for Enzyme), follow these patterns:

### Matrix Multiplication (`mul!`)
Avoid `*` when a destination buffer exists:
```julia
mul!(C, A, B)           # C = A*B
mul!(y, A, x)           # y = A*x
mul!(Y, A, B, α, β)     # Y = α*A*B + β*Y (no temp allocation)
```

### Linear Solves (`ldiv!`)
Avoid `\` inside hot loops. Pre-factorize outside:
```julia
F = lu(A)       # or cholesky(A)
ldiv!(F, b)     # Solve in-place, overwrites b
ldiv!(y, F, x)  # Solve, keep x intact, write to y
```

### Quadratic Forms (`dot`)
Avoid `x' * A * x` (allocates intermediate). Use 3-argument dot:
```julia
dot(x, A, y)    # x'*A*y without allocating A*y
dot(x, A, x)    # Quadratic form x'*A*x
```

### Symmetrization
For covariance matrices where numerical errors destroy symmetry:
- **Preferred:** Wrap with `Symmetric(A)` or `Hermitian(A)`
- **Materialize:** Use `LinearAlgebra.copytri!(A, 'U')` (Upper→Lower) or `'L'`
- **Averaging:** Use explicit loop for $(A + A^T)/2$ to avoid allocation

### Array Views
Slicing (`A[1:5, :]`) allocates. Use views:
```julia
@views A[1:5, :]        # SubArray, no copy
view(A, :, 1)           # Programmatic control
```
Organize algorithms for column-major access when using views.

### Transposition
Avoid `B = A'` if it materializes. Use:
```julia
transpose!(B, A)
adjoint!(B, A)
```

### Broadcasting
```julia
@. dest = f(args)                    # Fused, writes to dest
broadcast!(f, dest, args...)         # Explicit form
map!(f, dest, src)                   # Element-wise mapping
```

### Buffer Management
- Use in-place `f!(out, x)` for functions returning vectors
- Pass pre-allocated workspace as arguments (last position)
- Never allocate inside hot loops

---

## Repository Structure

```
lectures/
    topic_1/
        notebook_1.md
        notebook_2.md
    intro.md
    Manifest.toml
    Project.toml
    _config.yml
    _toc.yml
style.md
requirements.txt
```

Do not change folder structure or modify `_config.yml` unless explicitly asked.

---

## MyST Notebook Editing

- Preserve mystnb execution blocks
- Preserve input/output tags (e.g., `hide-output`, `remove-input`)
- Maintain anchors, index directives, labels, and roles
- Use fenced code blocks with correct language identifiers
- Do not mix Markdown and code execution in ambiguous ways

---

## Documentation Text

- Keep the pedagogical tone of existing lectures
- Use clean math notation:
  - Inline: `$x_t = \rho x_{t-1} + \sigma \varepsilon_t$`
  - Display: `$$ ... $$`
- Prefer concise explanations

---

## Environment Notes

Do **not** manually add package activation code like:
```julia
import Pkg; Pkg.activate(...)
```
The environment is handled automatically by JupyterBook or via `julia --project=lectures`.

---

## Allowed Tasks

- Fix failing notebook cells
- Improve explanations, math, and MyST structure
- Propose small utility Julia functions
- Improve clarity or reproducibility

## Tasks to Avoid

- Change folder structure
- Modify `_config.yml` unless asked
- Convert markdown notebooks to `.ipynb`
- Reorder large sections or rewrite whole lectures
- Add new dependencies without explicit instruction

---

## Reference

- [Style guide](style.md)
- [MyST-NB docs](https://myst-nb.readthedocs.io/)
