---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(getting_started)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Setting up Your Julia Environment

```{contents} Contents
:depth: 2
```

## Overview

In this lecture we will cover how to get up and running with Julia.

There are a few different options for using Julia, including a {ref}`local desktop installation <jl_jupyterlocal>` and {ref}`Jupyter hosted on the web <jl_jupyterhub>`.

If you have access to a web-based Jupyter and Julia setup, it is typically the most straightforward way to get started.

## A Note on Jupyter

Like Python and R, and unlike products such as Matlab and Stata, there is a looser connection between Julia as a programming language and Julia as a specific development environment.

While you will eventually use other editors, there are some advantages to starting with the [Jupyter](http://jupyter.org/) environment while learning Julia.

* The ability to mix formatted text (including mathematical expressions) and code in a single document.
* Nicely formatted output including tables, figures, animation, video, etc.
* Conversion tools to generate PDF slides, static HTML, etc.
* {ref}`Online Jupyter <jl_jupyterhub>` may be available, and requires no installation.

We'll discuss the workflow on these features in the {doc}`next lecture <../getting_started_julia/julia_environment>`.

(jl_jupyterlocal)=
## Desktop Installation of Julia and Jupyter

If you want to install these tools locally on your machine

* Download and install Julia, from [download page](http://julialang.org/downloads/) , accepting all default options.
* Currently, these instructions and packages will work with Julia 1.4.X or 1.5.X

(intro_repl)=
* Open Julia, by either
    1. Navigating to Julia through your menus or desktop icons (Windows, Mac), or
    1. Opening a terminal and typing `julia` (Linux; to set this up on Mac, see end of section)

You should now be looking at something like this

```{figure} /_static/figures/julia_term_1.png
:width: 100%
```

This is called the JULIA *REPL* (Read-Evaluate-Print-Loop), which we discuss more {ref}`later <repl_main>`.

* In the Julia REPL, hit `]` to enter package mode and then enter.

```{code-block} julia
add IJulia InstantiateFromURL
```

This adds packages for

* The  `IJulia` kernel which links Julia to Jupyter (i.e., allows your browser to run Julia code, manage Julia packages, etc.).
* The `InstantiateFromURL` which is a tool written by the QE team to manage package dependencies for the lectures.

Note: To set up the Julia terminal command on Mac, see [here](https://julialang.org/downloads/platform/#macos).

**Note**: To obtain the full set of packages we use, at this stage you can run the following (see {ref}`the package setup section <package_setup>`.)

```{code-block} julia
using InstantiateFromURL
github_project("QuantEcon/quantecon-notebooks-julia", version = "0.8.0", instantiate = true, precompile = true)
```

(jupyter_installation)=
### Installing Jupyter

If you have previously installed Jupyter (e.g., installing Anaconda Python by downloading the binary <https://www.anaconda.com/download/>)
then the `add IJulia` installs everything you need into your existing environment.

Otherwise, you can let `IJulia` install its own version of Conda by following [these instructions](https://julialang.github.io/IJulia.jl/dev/manual/running/).

(clone_lectures)=
### Starting Jupyter

Next, let's install the QuantEcon lecture notes to our machine and run them (for more details on the tools we'll use, see our lecture on {doc}`version control <../more_julia/version_control>`).

1. Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git/).
1. (**Optional, but strongly recommended**) Install the [GitHub Desktop](https://desktop.github.com/).

#### GitHub Desktop Approach

After installing the Git Desktop application, click [this link](x-github-client://openRepo/https://github.com/QuantEcon/quantecon-notebooks-julia) on your desktop computer to automatically install the notebooks.

It should open a window in the GitHub desktop app like this

```{figure} /_static/figures/git-desktop-intro.png
:width: 100%
```

Choose a path you like and clone the repo.

**Note:** the workflow will be easiest if you clone the repo to the default location relative to the home folder for your user.

Then, you can run Jupyterlab using the Conda installation with

```{code-block} none
jupyter lab
```

Or following [these instructions](https://julialang.github.io/IJulia.jl/dev/manual/running/) instructions if you didn't install Anaconda separately.

Navigate to the location you stored the lecture notes, and open the {doc}`Interacting with Julia <../getting_started_julia/julia_environment>` notebook to explore this interface and start writing code.

#### Git Command Line Approach

If you do not wish to install the GitHub Desktop, you can get the notebooks using the Git command-line tool.

Open a new terminal session and run

```{code-block} none
git clone https://github.com/quantecon/quantecon-notebooks-julia
```

This will download the repository with the notebooks in the working directory.

Then, `cd` to that location in your Mac, Linux, or Windows PowerShell terminal

```{code-block} none
cd quantecon-notebooks-julia
```

Then, either using the `using IJulia; jupyterlab()` or execute `jupyter lab` within your shell.

And open the {doc}`Interacting With Julia <../getting_started_julia/julia_environment>` lecture (the file `julia_environment.ipynb` in the list of notebooks in JupyterLab) to continue.

## Using Julia on the Web

If you have access to an online Julia installation, it is the easiest way to get started.

Eventually, you will want to do a {ref}`local installation <jl_jupyterlocal>` in order to use other
{doc}`tools and editors <../more_julia/tools_editors>` such as [Atom/Juno](http://junolab.org/), but
don't let the environment get in the way of learning the language.

(jl_jupyterhub)=
### Using Julia with JupyterHub

If you have access to a web-based solution for Jupyter, then that is typically a straightforward option

* Students: ask your department if these resources are available.
* Universities and workgroups: email [contact@quantecon.org](mailto:contact@quantecon.org") for
  help on setting up a shared JupyterHub instance with precompiled packages ready for these lecture notes.

#### Obtaining Notebooks

Your first step is to get a copy of the notebooks in your JupyterHub environment.

While you can individually download the notebooks from the website, the easiest way to access the notebooks is usually to clone the repository with Git into your JupyterHub environment.

JupyterHub installations have different methods for cloning repositories, with which you can use the url for the notebooks repository: [https://github.com/QuantEcon/quantecon-notebooks-julia](https://github.com/QuantEcon/quantecon-notebooks-julia).

(package_setup)=
## Installing Packages

After you have some of the notebooks available, as in {ref}`above <clone_lectures>`, these lectures depend on functionality (like packages for plotting, benchmarking, and statistics) that are not installed with every Jupyter installation on the web.

If your online Jupyter does not come with QuantEcon packages pre-installed, you can install the `InstantiateFromURL` package, which is a tool written by the QE team to manage package dependencies for the lectures.

To add this package, in an online Jupyter notebook run (typically with `<Shift-Enter>`)

```{code-block} julia
---
tags: [hide-output]
---
] add InstantiateFromURL
```

Then, run

```{code-block} julia
using InstantiateFromURL
github_project("QuantEcon/quantecon-notebooks-julia", version = "0.8.0", instantiate = true, precompile = true)
```

If your online Jupyter environment does not have the packages pre-installed, it may take 15-20 minutes for your first QuantEcon notebook to run.

After this step, open the downloaded {doc}`Interacting with Julia <../getting_started_julia/julia_environment>` notebook to begin writing code.

If the QuantEcon notebooks do not work after this installation step, you may need to speak to the JupyterHub administrator.

