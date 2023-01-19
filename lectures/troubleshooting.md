---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.8
---

(troubleshooting)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Troubleshooting

```{contents} Contents
:depth: 2
```

This troubleshooting page is to help ensure you software environment is setup correctly
to run this lecture set locally on your machine.

## Resetting your Lectures

Using VS Code, you can easily revert back individual lectures, all of the lectures, or get updated versions of the lecture notes.

See the lecture on [setting up your environment](reset_notebooks) for more.

If the `Project.toml` or `Manifest.toml` files are modified, then you may want to redo the [instantiation](install_packages) step to ensure you have the correct versions.


(reset_julia)=
## Fixing Your Local Environment

If packages are misbehaving, you may want to simply [reset the lectures](reset_notebooks) or at least the `Project.toml` and `Manifest.toml` files from the lecture notes, and then redo the [instantiation](install_packages) step  This will fix nearly every problem.

However, if you are still having issues you could delete the entire `.julia` directory for your users, and then redo the installation of packages as well as `] add IJulia`.  It is fast for recent versions of Julia.

The directory  is located in your user directory (e.g. `~/.julia` on MacOS and Linux, and `C:\Users\YOURUSERNAME\.julia` on Windows) or you can find this directory by running `DEPOT_PATH[1]` in a Julia REPL.

## Upgrading Julia

You should be able to upgrade Julia by simply
- Installing the latest release from the [Julia website](https://julialang.org/downloads/).
- Uninstalling your old release (so that the VS Code extension only uses the newest version)
- And, finally, you will need to redo the `] add IJulia` [installation step](intro_repl) to ensure that Jupyter knows how to find the new version.

```{warning}
While upgrading Julia will typically work with the notebooks, there may be upgrades where a particular lecture has problems.  Make sure that you have [updated your notebooks](reset_notebooks) in case bug fixes were made, and post an issue to the  [GitHub source for these lectures](https://github.com/QuantEcon/lecture-julia.myst/issues).
```

## Reporting an Issue

One way to give feedback is to raise an issue through our [issue tracker](https://github.com/QuantEcon/lecture-julia.myst/issues).

Please be as specific as possible.  Tell us where the problem is and as much
detail about your local set up as you can provide.

Another feedback option is to use our [discourse forum](https://discourse.quantecon.org/).

Finally, you can provide direct feedback to [contact@quantecon.org](mailto:contact@quantecon.org)

