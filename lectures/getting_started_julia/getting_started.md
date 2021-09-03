---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.6
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

While there are alternative ways to access Julia (e.g. if you have a JupyterHub provided by your university), this section assumes you will install it to your {ref}`local desktop  <jl_jupyterlocal>`.

## A Note on Jupyter

Like Python and R, and unlike products such as Matlab and Stata, there is a looser connection between Julia as a programming language and Julia as a specific development environment.

While you will eventually use other editors, there are some advantages to starting with the [Jupyter](http://jupyter.org/) environment while learning Julia.

* The ability to mix formatted text (including mathematical expressions) and code in a single document.
* Nicely formatted output including tables, figures, animation, video, etc.
* Conversion tools to generate PDF slides, static HTML, etc.

We'll discuss the workflow on these features in the {doc}`next lecture <../getting_started_julia/julia_environment>`.

(jl_jupyterlocal)=
## Desktop Installation of Julia and Jupyter

If you want to install these tools locally on your machine

### Installing Jupyter
Conda provides an easy to install package of jupyter, python, and many data science tools.

If you have not previously installed conda or Jupyter, then 
* Download the binary <https://www.anaconda.com/download/>) and follow the [installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for your platform.
* If given the option for your operating system, let Conda add Python to your PATH environment variables.

```{note}
While Conda is the easiest way to install jupyter, it is not strictly required.  With any python installation you can use `pip install jupyter`.  Alternatively you can let `IJulia` install its own version of Conda by following [these instructions](https://julialang.github.io/IJulia.jl/dev/manual/running/), or use the experimental support for [Jupyter notebooks in VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) which does not require a python installation.
```

(intro_repl)=
### Install Julia
After Conda is installed, you can install Julia.

* Download and install Julia, from [download page](http://julialang.org/downloads/), accepting all default options.

* Open Julia, by either
    1. Navigating to Julia through your menus or desktop icons (Windows, Mac), or
    2. Opening a terminal and typing `julia` (Linux; to set this up on Mac, see end of section)

You should now be looking at something like this

```{figure} /_static/figures/julia_term_1.png
:width: 100%
```

This is called the JULIA *REPL* (Read-Evaluate-Print-Loop), which we discuss more {ref}`later <repl_main>`.

* In the Julia REPL, hit `]` to enter package mode and then enter.

```{code-block} julia
add IJulia
```

This adds packages for the `IJulia` kernel which links Julia to Jupyter (i.e., allows your browser to run Julia code, manage Julia packages, etc.).

You can exit the julia REPL by hitting backspace to exit the package mode, and then 

```{code-block} julia
exit()
```

```{note}
To set up the Julia terminal command on Mac, see [here](https://julialang.org/downloads/platform/#macos).
```

(clone_lectures)=
### Downloading the Notebooks

Next, let's install the QuantEcon lecture notes to our machine and run them (for more details on the tools we'll use, see our lecture on {doc}`version control <../software_engineering/version_control>`).

1. Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git/), which is the industry standard tool for managing versioned code.
2. Open a new terminal for your machine and navigate to the parent folder of where you wish to store the notebooks.
  - The workflow will be easiest if you clone the repo to the default location relative to the home folder for your user.  For example, on Windows a good choice might be `c:\users\YOURUSERNAME\GitHub`
3. Then run,

```{code-block} bash
git clone https://github.com/quantecon/lecture-julia.notebooks
```

This will download the repository with the notebooks in the working directory.

Then, `cd` to that location in your Mac, Linux, or Windows PowerShell terminal

```{code-block} bash
cd lecture-julia.notebooks
```

Alternatively, if you are already a user of Visual Studio Code, you can clone within VS Code by using the `> Git: Clone` command.  See the lectures on [tools](../software_engineering/tools_editors.md) and [source code control](../software_engineering/version_control.md) for more details.

### Installing Packages

After you have the notebooks available, as in {ref}`above <clone_lectures>`, these lectures depend on functionality (like packages for plotting, benchmarking, and statistics) that are not installed with every Jupyter installation on the web.

Given that you are in the `lecture-julia.notebooks` directory, you can download and install all of the required packages with
```{code-block} bash
julia --project --threads auto -e 'using Pkg; Pkg.instantiate();'
```
Or, alternatively, in a Julia REPL run `] activate; instantiate` or simply `] instantiate` if you start the julia terminal with `julia --project --threads auto` within this folder.

The package installation and compilation will take several minutes, but afterwards you will be able to use all of the notebooks without further installatoin.

### Running JupyterLab

Then, you can run Jupyterlab using the Conda installation with

```{code-block} bash
jupyter lab
```

Or following [these instructions](https://julialang.github.io/IJulia.jl/dev/manual/running/) instructions if you didn't install Anaconda separately and wished for Julia to manage it separately.

You should see a webpage such as 

```{figure} /_static/figures/jupyterlab_first.png
:width: 100%
```

Navigate to the location you stored the lecture notes, and open the {doc}`Interacting with Julia <../getting_started_julia/julia_environment>` notebook (the file `getting_started_julia/julia_environment.ipynb` in the list of notebooks in JupyterLab) to explore this interface and start writing code.
