# Project Instructions

This repository builds QuantEcon Julia lectures using JupyterBook + MyST Markdown + MyST-NB execution.

See global rules (`~/.claude/rules/`) for Julia, JupyterBook, and notation conventions.

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

To run in julia use lectures project, i.e. `julia --project=lectures`.

Use the virtual environment with `uv` installed and activated to run `uv run jb build` etc.

## Guardrails

**Do:**
- Fix failing notebook cells
- Improve explanations, math, and MyST structure
- Propose small utility Julia functions
- Improve clarity or reproducibility
- Follow patterns already present in similar notebooks
- Prefer conservative, minimally invasive edits

**Don't:**
- Change folder structure
- Modify `_config.yml` unless asked
- Convert markdown notebooks to `.ipynb`
- Reorder large sections or rewrite whole lectures
- Add new dependencies without explicit instruction

## Reference

- [Style guide](style.md)
- [MyST-NB docs](https://myst-nb.readthedocs.io/)

## Regression Testing Framework

Lecture notebooks embed regression tests in `remove-cell` tagged code cells. These tests run during the JupyterBook build and catch silent breakage from dependency updates, RNG changes, or refactoring.

### Pattern

```markdown
\`\`\`{code-cell} julia
---
tags: [remove-cell]
---
@testset "Descriptive Name" begin
    @test results.v[4] ≈ 20.749453024528787
    @test sigma == [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 6, 6]
end
\`\`\`
```

The `using Test` import goes in its own `remove-cell` near the top of the notebook.

### Seed placement

`Random.seed!(42)` goes in **visible** cells (not `remove-cell`) immediately before stochastic code, so readers can reproduce output. Deterministic computations (PFI, VFI, pure arithmetic) do not need seeds.

### Tolerance conventions

| Comparison | When to use |
|------------|------------|
| `≈` (bare) | Default for deterministic floating-point (`rtol ≈ 1.5e-8`) |
| `atol=...` | Values near zero where relative tolerance is meaningless |
| `==` | Exact integer/boolean comparisons (policy vectors, flags) |
| `rtol=...` | Only if needed; flag `rtol > 1e-5` as suspicious |

### Re-enabling commented-out tests

Many notebooks have tests disabled with `#test` or `# @test` from a past RNG change. To re-enable:

1. Add `using Random` to imports if stochastic sections exist
2. Add `Random.seed!(42)` in visible cells before stochastic code
3. Write a temporary script (`/tmp/<notebook>_values.jl`) replicating the notebook's computations; run with `julia --project=lectures` to get fresh expected values
4. Change `#test` → `@test` and update hardcoded values
5. Add a few canary tests for important results (see below)
6. Verify with a test script that runs all assertions in cell order
7. **Bug check:** If a fresh value differs from the old commented-out value by more than the expected tolerance (e.g., `rtol > 1e-5` for deterministic code), flag it to the caller — this likely indicates a bug was introduced after the tests were commented out, not just an RNG/precision change

### Canary tests

When adding tests, prefer "canary" values that depend on many prior calculations (e.g., `v[end]`, mid-grid policy values, endpoint stationary distributions). These catch upstream breakage that element-specific tests might miss.

## Releasing

Releases use date-based tags on `main`. Before releasing, verify:

1. The local repo is on the `main` branch and up to date (`git checkout main && git pull`)
2. No "Build Cache" GitHub Actions workflow is currently running on `main` (check with `gh run list --workflow=cache.yml -b main`)

To create a release:

```bash
gh release create publish-<YYYY><mon><D> --target main \
  --title "PUBLISH: <D><ordinal> <Month> <YYYY>" --generate-notes
```

- **Tag**: `publish-YYYYmonD` — lowercase 3-letter month, day without leading zero (e.g., `publish-2026feb9`)
- **Title**: `PUBLISH: 9th February 2026`
- **Body**: Auto-generated from merged PRs (`--generate-notes`)
- If releasing again on the same day, append a letter suffix: `publish-2026feb9b`
