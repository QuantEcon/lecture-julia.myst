---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia
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

It is not strictly required for running the lectures, but we will strongly encourage installing and using [Visual Studio Code (VS Code)](https://code.visualstudio.com/).

We will use it as our primary editor starting in the {doc}`tools lecture <../software_engineering/tools_editors>`.

## TL;DR
Julia and the lecture notebooks can be installed without Jupyter or Python:

1. Install [Git](https://git-scm.com/install/)
2. Install [VS Code](https://code.visualstudio.com/)
3. Install Julia following the [Juliaup instructions](https://github.com/JuliaLang/juliaup#installation)
   - Windows: `winget install julia -s msstore` in a terminal
   - Linux/Mac: `curl -fsSL https://install.julialang.org | sh` in a terminal
4. Install the [VS Code Julia extension](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia)
5. In VS Code, open the command palette with `<Ctrl+Shift+P>` and type `> Git: Clone` to clone the repository `https://github.com/quantecon/lecture-julia.notebooks` in a new window
6. Start a Julia REPL in the integrated terminal with `> Julia: Start REPL` from the command palette, then enter package mode with `]` and then type `instantiate`.
  - That process will take several minutes to download and compile all of the packages used in the lectures.
7. Open any of the `.ipynb` files in VS Code and select the `Julia` channel (i.e., not the Jupyter channels if you have them installed) when prompted to run the notebooks directly within VS Code

At that point, you can directly move to the {doc}`julia by example <../getting_started_julia/julia_by_example>` lecture.


## A Note on Jupyter

[Jupyter](http://jupyter.org/) notebooks are an alternative way to work with Julia, letting you mix code, formatted text, and output in a single document.  However, the recommended workflow for these lectures is VS Code (see {ref}`Setting up Git and VS Code <initial_vscode_setup>`).  If you prefer standalone Jupyter Lab, see the [installation instructions below](jl_jupyterlocal).

These lectures assume some prior programming experience (variables, loops, conditionals). The [Julia documentation](https://docs.julialang.org/) is a good starting point for newcomers.

(jl_jupyterlocal)=
## Desktop Installation of Julia and Jupyter

```{note}
This section is **only needed if you want the standalone Jupyter Lab workflow**.  If you plan to use VS Code (recommended) or Google Colab, you can skip to {ref}`Setting up Git and VS Code <initial_vscode_setup>`.
```

(install_jupyter)=
### Installing Jupyter
[Anaconda](https://www.anaconda.com/) provides an easy way to install Jupyter, Python, and many data science tools.

1. Download the binary (<https://www.anaconda.com/download/>) and follow the [installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for your platform.
2. If given the option, let Conda add Python to your PATH environment variables.

```{note}
There is direct support for [Jupyter notebooks in VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) with **no Python installation**. See {ref}`VS Code Julia Kernel <running_vscode_kernel>`.
```

(intro_repl)=
### Install Julia

```{note}
The "official" installation for Julia is now [Juliaup](https://github.com/JuliaLang/juliaup), which makes it easier to upgrade and manage concurrent Julia versions.  See [here](https://github.com/JuliaLang/juliaup#using-juliaup) for a list of commands, such as `juliaup update` to upgrade to the latest available Julia version.

**Troubleshooting:** On Mac/Linux, if you have permissions issues on the installation use `sudo curl -fsSL https://install.julialang.org | sh`.  If there are still permissions issues, [see here](https://github.com/JuliaLang/juliaup/wiki/Permission-problems-during-setup) for further steps.
```

1. Download and install Julia following the [Juliaup instructions](https://github.com/JuliaLang/juliaup#installation)
    - Windows: `winget install julia -s msstore` in a terminal
    - Linux/Mac: `curl -fsSL https://install.julialang.org | sh` in a terminal
    - If you have previously installed Julia manually, you will need to uninstall previous versions before switching to `juliaup`

2. Open Julia, by either
    - Navigating to Julia through your menus or desktop icons (Windows, Mac), or
    - Opening a terminal and typing `julia`

   You should now be looking at something like this

   ```{figure} /_static/figures/julia_term_1.png
   :width: 100%
   ```

   This is called the JULIA *REPL* (Read-Evaluate-Print-Loop), which we discuss more {ref}`later <repl_main>`.

3. In the Julia REPL, hit `]` to enter package mode and then enter:

   ```{code-block} julia
   add IJulia
   ```

   This adds the `IJulia` kernel which links Julia to Jupyter (i.e., allows your browser to run Julia code, manage Julia packages, etc.).

4. Exit the package mode with backspace and then quit with `exit()`.

```{note}
As entering package mode is common in these notes, we will denote this with `] add IJulia`, etc.
```

(initial_vscode_setup)=
## Setting up Git and VS Code

First, install [Git](more_on_git), the industry standard version-control tool, which lets you download files and their entire version history from a server (e.g. on GitHub) to your desktop.  We cover Git in detail in the lectures on {doc}`source code control <../software_engineering/version_control>` and {doc}`testing <../software_engineering/testing>`.


1. Install [Git](https://git-scm.com/install/) and accept the default arguments.
   - If you allow Git to add to your path, then you can run it with the `git` command, but we will frequently use the built-in VS Code features.
3. (Optional) Install [VS Code](https://code.visualstudio.com/) for your platform and open it
   - On Windows, during install under `Select Additional Tasks`, choose all options that begin with `Add "Open with Code" action`. This lets you open VS Code from inside File Explorer folders directly.
   - While optional, we find the experience with VS Code will be much easier and the transition to more advanced tools will be more seamless.
4. (Optional) Install the [VS Code Julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) extension
   - After installation of VS Code, you should be able to choose `Install` on the webpage of any extensions and it will open on your desktop.
   - Otherwise: run VS Code and open the extensions with `<Ctrl+Shift+X>` or selecting extensions in the left-hand side of the VS Code window.  Then search for `Julia` in the Marketplace.
   ```{figure} /_static/figures/vscode_intro_0.png
   :width: 60%
   ```
   - No further configuration should be required, but see [here](install_vscode) if you have issues.

The VS Code and the VS Code Julia extension will help us better manage environments during our initial setup, and will provide a seamless transition to the more {doc}`advanced tools <../software_engineering/tools_editors>`.

(command_palette)=
### Command Palette on VS Code

A key feature within VS Code is the [Command Palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette), which can be accessed with `<Ctrl+Shift+P>` or `View > Command Palette...` in the menus.

```{figure} https://code.visualstudio.com/assets/docs/getstarted/userinterface/commands.png
:width: 60%
```

This is so common that in these notes we
denote opening the command palette and searching for a command with things like `> Julia: Start REPL` , etc.

(clone_lectures)=
## Downloading the Notebooks

Next, let's install the QuantEcon lecture notes to our machine and run them (for more details on the tools we'll use, see our lecture on {doc}`version control <../software_engineering/version_control>`).

1. Open the command palette with `<Ctrl+Shift+P>` and type `> Git: Clone`
2. For the Repository URL, enter `https://github.com/quantecon/lecture-julia.notebooks`
3. Choose the location to clone when prompted
   - For example, on Windows a good choice is `c:\Users\YOURUSERNAME\Documents\GitHub`.  On Linux and macOS, `~` or `~/GitHub`.
4. Accept the option to open in a new window when prompted

```{admonition} Cloning without VS Code
:class: dropdown

Alternatively, you can clone from the command line:

1. Open a terminal and navigate to a convenient parent folder (see above for suggestions).
2. Run `git clone https://github.com/quantecon/lecture-julia.notebooks`
3. `cd lecture-julia.notebooks`
4. Open the directory in VS Code with `code .`, or open it manually.
```

If you have opened this in VS Code, it should look something like

```{figure} /_static/figures/vscode_intro_1.png
:width: 100%
```



(install_packages)=
## Installing Packages

After you have the notebooks available, as described in [the previous section](clone_lectures), we can install the required packages for plotting, benchmarking, and statistics.

For this, we will use the integrated terminal in VS Code.

Recall that you can start this directly from the [command palette](command_palette) with `<Ctrl+Shift+P>` then typing part of the `> Julia: Start REPL` command.
```{figure} /_static/figures/vscode_intro_2.png
:width: 75%
```

1. Start a REPL; it may do an initial compilation of packages in the background, but will then look something like
 
   ```{figure} /_static/figures/vscode_intro_3.png
   :width: 100%
   ```

3. Next type `]` to enter the package mode, which should indicate that the local project is activated by changing the cursor to `(quantecon-notebooks-julia) pkg>`.

4. Type `instantiate` to install all of the packages required for these notes.

   ```{figure} /_static/figures/vscode_intro_4.png
   :width: 100%
   ```

This process will take several minutes to download and compile all of the files used by the lectures.

```{attention}
If the package-mode cursor shows `(@v1.x) pkg>` instead of `(quantecon-notebooks-julia) pkg>`, type `activate .` to activate the local project.
```

```{admonition} Why use the integrated REPL?
:class: dropdown
The VS Code integrated REPL automatically sets thread count and activates the local project.  If you use an external REPL, launch it with `julia --project --threads auto`.  See [here](repl_main) for details.
```


(running_jupyterlab)=
## Running JupyterLab

```{note}
This section is only needed for the standalone Jupyter Lab workflow.  If you use VS Code, you can run notebooks directly --- see {ref}`VS Code Julia Kernel <running_vscode_kernel>`.
```

Open a terminal in the notebook directory and run:

```{code-block} bash
jupyter lab
```

This launches Jupyter with access to the current directory.  A browser tab should open automatically; if not, follow the link shown in the terminal output.

```{figure} /_static/figures/jupyterlab_first.png
:width: 100%
```

(reset_notebooks)=
## Refreshing the Notebooks after Modification

To revert notebooks to the latest version from the server:

1. In VS Code, open the "Source Control" pane (`<Ctrl+Shift+G>`), right-click "Changes", and select `Discard All Changes`.
2. To pull the latest updates, use the `> Git: Pull` command or click the sync arrow next to "main" in the status bar.

If the `Project.toml` or `Manifest.toml` files changed, re-enter package mode (`]`) and run `instantiate` to update packages.

We will explore these features in the {doc}`source code control <../software_engineering/version_control>` lecture.


(julia_environment)=
## Interacting with Julia

If you are new to Jupyter notebooks, see the [QuantEcon Python lecture](https://python-programming.quantecon.org/getting_started.html) for an introduction to the interface, including cells, execution, and keyboard shortcuts — the basics are identical regardless of language.

Below we cover Julia-specific features of the notebook environment.

### Plots

Run the following cell

```{code-cell} julia
using Plots
using LinearAlgebra
plot(sin, -2π, 2π, label = "sin(x)")
```

```{attention}
If `Plots` is not found, either [install the packages](install_packages) (or run `using Pkg; Pkg.instantiate()` in a new cell), or — if you downloaded this notebook rather than [cloning the repository](clone_lectures) — install manually with `] add Plots`.
```

### Inserting Unicode (e.g. Greek letters)

Julia supports the use of [unicode characters](https://docs.julialang.org/en/v1/manual/unicode-input/)
such as `α` and `β` in your code.

Unicode characters can be typed quickly in Jupyter using the `tab` key.

Try creating a new code cell and typing `\alpha`, then hitting the `tab` key on your keyboard.

There are other operators with a mathematical notation.  For example, the `LinearAlgebra` package has a `dot` function identical to the LaTeX `\cdot`.

```{code-cell} julia
using LinearAlgebra
x = [1, 2]
y = [3, 4]
@show dot(x, y)
@show x ⋅ y;
```

### Shell and Package Commands

You can execute shell commands in the REPL or a notebook cell by prepending `;` (e.g., `; ls`), and package operations by prepending `]` (e.g., `] st`).  Cells using `;` or `]` must be one-liners.

(running_vscode_kernel)=
## Running the VS Code Julia Kernel
The Jupyter Extension for VS Code supports Julia directly without the need for a Python installation.

With cloned notebooks, open a `.ipynb` file directly in VS Code.  Depending on your setup, you may see a request to choose the kernel for executing the notebook.

To do so, click on the `Choose Kernel` or `Select Another Kernel...` which may display an option such as

   ```{figure} /_static/figures/vscode_julia_kernel.png
   :width: 80%
   ```

Choose the `Julia`  kernel, rather than the `Jupyter Kernel...` to bypass the Python Jupyter setup.  If successful, you will see the kernel name as `Julia channel` or something similar.

With the kernel selected, you will be able to run cells in the VS Code UI with similar features to Jupyter Lab.  For example, below shows the results of play icon next to a code cell, which will `Execute Cell` and display the results inline.


   ```{figure} /_static/figures/vscode_jupyter_kernel_execute.png
   :width: 100%
   ```

(running_colab)=
## Running on Google Colab

[Google Colab](https://colab.research.google.com/) provides a hosted Julia runtime that can run the lecture notebooks directly in your browser with no local installation.

The easiest way to launch any lecture notebook in Colab is to click the {fas}`circle-play` icon at the top of the page and select **Colab** from the Notebook Launcher:

```{figure} /_static/figures/colab_launcher.png
:width: 60%
```

Alternatively, you can navigate to the [notebook repository](https://github.com/quantecon/lecture-julia.notebooks) and download or copy the URL.  Then log in to [colab.research.google.com](https://colab.research.google.com) and choose **File > Open notebook > GitHub** or the **Upload** tab to open the notebook.

Once the notebook is open in Colab:

1. Colab should automatically detect the Julia kernel
2. Before running the notebook, you will need to install the required packages.  Look at the first code cell for the list of packages and modify the cell to include a call to `Pkg.add`.  For example, if the first cell is

   ```{code-block} julia
   using LinearAlgebra, Statistics, Plots, LaTeXStrings
   ```

   then modify this cell, or add a cell above, with

   ```{code-block} julia
   using Pkg
   Pkg.add(["LinearAlgebra", "Statistics", "Plots", "LaTeXStrings"])
   ```

   This only needs to be done once per package — Colab will remember installed packages for the duration of the session.

   Alternatively, you can ask Gemini to install the packages and it will generate the installation code for you within the cell.

   ```{figure} /_static/figures/colab_install.png
   :width: 100%
   ```

3. After installation completes, run the first cell and continue through the notebook as usual
