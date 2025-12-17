# Background and Guidelines
I am trying to add lecture notes which show differentiable simulation in Julia using Enzyme.jl.  We will add these to various lectures when they are complete.

You are an expert Julia developer specializing in high-performance computing, zero-allocation patterns, and auto-differentiation. When writing code, prioritize strict memory management and in-place operations. Adhere to the following guidelines`:
- See ./AGENTS.md for more details and look at programming patterns in the code examples as required.
- Read ./prompts/differentiation.md for specific Enzyme.jl coding patterns and inplace.
- You can create a .jl file and then run it with `julia --project=lectures ...` since that project and manifest file are already set up with Enzyme and all dependencies.
- Do not add new dependencies unless absolutely necessary.
- Our primary AD target is Enzyme.jl.
- However, it has some constraints on usage.  In particular, it does not support allocation inside of differentiable functions (e.g., avoids garbage collection). See ./prompts/differentiation.md for specific coding patterns to follow when writing Enzyme-differentiable Julia code.

# Notation for Linear State Space Models
If using a LSS, the math notation/formulation is as follows:
$$
x_{t+1} = A x_t + C w_{t+1}
y_t = G x_t + H v_t
$$
where
- w_{t+1} ~ N(0, I
- v_t ~ N(0, I).  
- x_t \in R^N
- y_t \in R^M
- w_{t+1} \in R^K
- v_t \in R^L.
- A \in R^{N x N}, C \in R^{N x K}, G \in R^{M x N}, H \in R^{M x L}.
- In cases with an initial prior x_0 ~ N(mu_0, Sigma_0) if normal

# Task
### STEP 1:
I want to show first code for an inplace simulation of a LSS model.  Write a function with signature
```julia
simulate_lss!(x, y, A, C, G, H, x_0, w, v)
```
- The simulation length is implicit in the `w` and `v`.  i.e., T = length(w) and I believe that v is then of size T+1
- The output `x` and `y` are preallocated arrays of size (N, T+1) and (M, T+1) respectively.  Organize the column vs. row ordering to ensure cache contiguity.
I do not believe it needs a cache in this case since it would modify the `x` and `y` in place.
- When you have that function write a simple example with $N = 3, M = 2, K = 2, L = 2$ and $T = 10$ where you either randomly choose the matrices or pick something reasonable (start with small `C` and `H` to avoid dominating noise).  Draw the random normal shocks with `randn`.
- Use plots to show the output of the `x` and `y` time series.

Let me know when you have written this as a nice, clean jl file in ./lectures/more_julia/differentiable_simulation.jl
