# Project Instructions for AI Coding Tools

This repository builds the QuantEcon Julia lectures using JupyterBook + MyST Markdown + MyST-NB execution.
AI coding tools working in this repo should follow these guidelines for consistent formatting, structure, and execution.

------------------------------------------------------------------------------

## 1. Code Style and Formatting Conventions

Follow the QuantEcon `style.md` guide strictly:

- Use the conventions described in:
  https://github.com/QuantEcon/lecture-julia.myst/blob/main/style.md
- Maintain consistent Julia formatting, including:
  - Non-mutating vs. mutating naming (`f` vs. `f!`)
  - Explicit `using` statements
  - Avoid global variables in notebooks
  - Reproducibility via fixed seeds
- When editing MyST Markdown:
  - Use fenced code blocks with correct language identifiers
  - Do not mix Markdown and code execution in ambiguous ways
  - Preserve section structure, callouts, and MyST directives

If unsure about style, imitate existing notebooks in `lectures/`.

------------------------------------------------------------------------------

## 2. Repository Structure Expectations

Maintain the existing layout:

lectures/
    topic_1/
        notebook_1.md
        notebook_2.md
    topic_2/
        notebook_3.md
    intro.md
    Manifest.toml
    Project.toml
    _config.yml
    _toc.yml
style.md
requirements.txt


The requirements.txt file is for the jupyterbook dependencies, not the lecture notes themselves.
Do not introduce new directory structures without explicit instruction.

------------------------------------------------------------------------------

## 3. Environment Setup for Notebook Execution

AI tools may compile or test notebooks using the following:

1. Activate the Python environment:

   source .venv/bin/activate

2. Build the JupyterBook:

   jb build lectures

3. Inspect failures:
   The JupyterBook warnings will include file paths like:

   /path/to/notebook.md: WARNING: Executing notebook failed
   Reports saved in:
   lectures/_build/html/reports/<subpath>/<notebook>.err.log

AI suggestions should reference these logs when debugging notebook execution.

To run Julia directly with the correct environment:

```
julia --project=lectures
```

Do not manually add in importing code for packages in a lecture (i.e., `import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))` etc.) as it will be correctly activated by either jupyterbook or your activation of the julia environment with `julia --project=lectures`.

------------------------------------------------------------------------------

## 4. Expectations for Notebook Editing

### Julia Code Cells
- Produce idiomatic, stable, predictable Julia.
- Avoid unnecessary refactoring.
- Use broadcasting, comprehensions, and vectorization when appropriate.
- Prioritize clarity and reproducibility.

### MyST Notebook Cells
- Preserve mystnb execution blocks.
- Preserve input/output tags (e.g., hide-output, remove-input).
- Maintain anchors, index directives, labels, and roles.
- Do not introduce unsupported Jupyter features.

------------------------------------------------------------------------------

## 5. Documentation and Text Edits

AI-generated text must:
- Keep the pedagogical tone of existing QuantEcon lectures.
- Use clean mathematical notation:
  - Inline math: $x_t = \\rho x_{t-1} + \\sigma \\varepsilon_t$
  - Display math: $$ ... $$
- Prefer concise explanations and avoid verbosity.

------------------------------------------------------------------------------

## 6. Commit Behavior

AI tools should:
- Keep commits small and targeted.
- Avoid mass rewrites.
- Maintain full compatibility with the JupyterBook build.

------------------------------------------------------------------------------

## 7. Good Default Behaviors for AI Tools

When uncertain:
- Follow patterns already present in similar notebooks.
- Test execution with `jb build lectures`.
- Refer to the style guide.
- Prefer conservative, minimally invasive edits.
- Prioritize correctness of executed notebooks.

------------------------------------------------------------------------------

## 8. Allowed Tasks

AI tools may:
- Fix failing notebook cells.
- Improve explanations, math, and MyST structure.
- Propose small utility Julia functions.
- Improve clarity or reproducibility.

------------------------------------------------------------------------------

## 9. Tasks to Avoid

AI tools should not:
- Change folder structure.
- Modify `_config.yml` unless asked.
- Convert markdown notebooks back to `.ipynb`.
- Reorder large sections or rewrite whole lectures.
- Add new dependencies without explicit instruction.

------------------------------------------------------------------------------

## 10. Reference Materials

- Style guide:
  https://github.com/QuantEcon/lecture-julia.myst/blob/main/style.md

- MyST-NB docs:
  https://myst-nb.readthedocs.io/

- QuantEcon Julia documentation.

------------------------------------------------------------------------------

These instructions apply to all AI coding assistants operating in this repository.