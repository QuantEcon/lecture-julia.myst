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
