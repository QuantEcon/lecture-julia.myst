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

It is not strictly required for running the lectures, but we will strongly encourage installing and using [Visual Studio Code (VS Code)](https://code.visualstudio.com/).

As the most popular and best-supported open-source code editor, it provides a large number of useful features and extensions - even if we do not edit the Julia code within it directly.  Later, in the {doc}`tools lecture <../software_engineering/tools_editors>` we will use VS Code directly.

## A Note on Jupyter

Like Python and R, and unlike products such as Matlab and Stata, there is a looser connection between Julia as a programming language and Julia as a specific development environment.

While you will eventually use other editors, there are some advantages to starting with the [Jupyter](http://jupyter.org/) environment while learning Julia.

* The ability to mix formatted text (including mathematical expressions) and code in a single document.
* Nicely formatted output including tables, figures, animation, video, etc.
* Conversion tools to generate PDF slides, static HTML, etc.

We'll discuss the workflow on these features in the {doc}`next lecture <../getting_started_julia/julia_environment>`.

```{admonition} Quick installation for experience users
If you have already installed Jupyter, Julia, and Git and have experience with these tools, you can 
  - Get the notebooks repositories with `git clone https://github.com/quantecon/lecture-julia.notebooks` 
  - Open a Jupyter notebook within the downloaded notebooks
  - Install the necessary packages `using Pkg; Pkg.instantiate()`

At that point, you could directly move on to the {doc}`julia by example <../getting_started_julia/julia_by_example>` lecture.

However, as we strongly recommend becoming familiar with VS Code as a transition towards using more advanced tools and to support better software engineering workflows, so consider walking through the rest of these instructions.
```

(jl_jupyterlocal)=
## Desktop Installation of Julia and Jupyter

In this section, we will describe the installation of Julia and Jupyter on your desktop.


```{tip}
On Windows, you probably want to install the new open-source [Windows Terminal](https://github.com/microsoft/terminal).  See [here](https://aka.ms/terminal) for installation instructions, and select the option to add the explorer context menu if provided.

It provides a much more modern terminal with better font support for Julia, and with better operating system integration.  For example, you can right-click on a folder in the File Explorer and choose `Open in Microsoft Terminal` to start a terminal in that location.
```

### Installing Jupyter
Conda provides an easy to install package of jupyter, python, and many data science tools.

If you have not previously installed conda or Jupyter, then 
1. Download the binary <https://www.anaconda.com/download/>) and follow the [installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for your platform.
2. If given the option for your operating system, let Conda add Python to your PATH environment variables.

```{note}
While Conda is the easiest way to install jupyter, it is not strictly required.  With any python installation you can use `pip install jupyter`.  Alternatively you can let `IJulia` install its own version of Conda by following [these instructions](https://julialang.github.io/IJulia.jl/dev/manual/running/), or use the experimental support for [Jupyter notebooks in VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) which does not require a python installation.
```

(intro_repl)=
### Install Julia
After Conda is installed, you can install Julia.

1. Download and install Julia, from [download page](http://julialang.org/downloads/), accepting all default options.

2. Open Julia, by either
    - Navigating to Julia through your menus or desktop icons (Windows, Mac), or
    - Opening a terminal and type `julia` (Linux; to set this up on Mac, see end of section)

   You should now be looking at something like this
   
   ```{figure} /_static/figures/julia_term_1.png
   :width: 100%
   ```
   
   This is called the JULIA *REPL* (Read-Evaluate-Print-Loop), which we discuss more {ref}`later <repl_main>`.

3. In the Julia REPL, hit `]` to enter package mode and then enter.

   ```{code-block} julia
   add IJulia
   ```

   This adds packages for the `IJulia` kernel which links Julia to Jupyter you previously installed with Anaconda (i.e., allows your browser to run Julia code, manage Julia packages, etc.).

4. You can exit the julia REPL by hitting backspace to exit the package mode, and then 

   ```{code-block} julia
   exit()
   ```

```{tip}
To set up the Julia terminal command on Mac, see [here](https://julialang.org/downloads/platform/#macos).
```

(initial_vscode_setup)=
## Setting up Git and (Strongly Recommended) VS Code

A primary benefit of using [open-source](https://en.wikipedia.org/wiki/Open-source_software) languages such as Julia, Python, and R is that they can enable far better workflows for both collaboration and [reproducible research](https://en.wikipedia.org/wiki/Reproducibility#Reproducible_research).

Reproducibility will ensure that you, you future self, your collaborators, and eventually the public will be able to run the exact code with the identical environment with which you provided the results - or even roll back to a snapshot in the past where the results may have been different in order to compare.

We will explore these topics in detail in the lectures on {doc}`source code control <../software_engineering/version_control>` and {doc}`continuous integration and test-driven development <../software_engineering/testing>`, but it is worth installing and beginning to use these tools immediately.


First, we will install [Git](more_on_git), which has become the industry standard open-source version-control tool.  This lets you download both the files and the entire version history from a server (e.g. on GitHub) to your desktop.


1. Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git/) and accept the default arguments.
   - If this is on your PATH, then you can run it with the `git` command, but we will frequently use the built-in VS Code features.
3. (Optional) Install [VS Code](https://code.visualstudio.com/) for your platform and open it
   - On Windows, during install under `Select Additional Tasks`, choose all options that begin with `Add "Open with Code" action`. This lets you open VS Code from inside File Explorer folders directly.
   - While optional, we find the experience with VS Code will be much easier and the transition to more advanced tools will be more seamless.
4. (Optional) Install the [VS Code Julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) extension
   - No further configuration should be required, but see [here](install_vscode) if you have issues.
   - After installation of VS Code, you should be able to choose `Install` on the webpage of any extensions and it will open on your desktop.
   - Otherwise, open the extensions with `<Ctrl+Shift+X>` or selecting extensions in the left-hand side of the VS Code window.  Then search for `Julia` in the Marketplace.

```{figure} /_static/figures/vscode_intro_0.png
:width: 50%
```

The VS Code and the VS Code Julia extension will help us better manage environments during our initial setup, and will provide a seamless transition to the more {doc}`advanced tools <../software_engineering/tools_editors>`.

(command_palette)=
### Command Palette on VS Code

A key feature within VS Code is the [Command Palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette), which can be accessed with `<Ctrl+Shift+P>` or `View > Command Palette...` in the menus.

```{figure} https://code.visualstudio.com/assets/docs/getstarted/userinterface/commands.png
:width: 75%
```

With this, you can type partial strings for different commands and it helps you to find features of vscode and its extensions.  This is so common that in these notes we
denote opening the command palette and searching for a command with things like `> Julia: Start REPL` , etc.  You will only need to type part of the string, and the command palette remembers
your most recent and common commands.

[Integrated Terminals](https://code.visualstudio.com/docs/editor/integrated-terminal) within VS Code are a convenient because they are local to that project, detect hypertext links, and provide better fonts.

To launch a terminal, use ``<Ctrl+`>``, `>View: Toggle Integrated Terminal` with the command palette, or `View > Terminal` in the menus.


(clone_lectures)=
## Downloading the Notebooks

Next, let's install the QuantEcon lecture notes to our machine and run them (for more details on the tools we'll use, see our lecture on {doc}`version control <../software_engineering/version_control>`).

While the lecture notes can be cloned within VS Code directly, we will use the command-line to begin the introduction to source control tools.

1. Choose and create if necessary a convenient parent folder where you would like the notebooks directory
   - The workflow will be easiest if you clone the repo to the default location relative to the home folder for your user.
   - For example, on Windows a good choice might be `c:\Users\YOURUSERNAME\Documents\GitHub` or simply `c:\Users\YOURUSERNAME\Documents`.  On linux and MacOS, your home directory `~` or `~/GitHub`.
2. Open a new terminal for your machine and navigate to the parent folder of where you wish to store the notebooks.
   - On Windows: if using the [Windows Terminal](See [here](https://aka.ms/terminal)) you can simply right-click on the directory in the File Explorer and choose to "Open in Microsoft Terminal" or, alternatively "Git Bash Here" to use the terminal provided by Git.
3. Execute the following code in the terminal to download the entire suite of notebooks associated with these lectures.
   ```{code-block} bash
   git clone https://github.com/quantecon/lecture-julia.notebooks
   ```
   This will download the repository with the notebooks into the directory `lecture-julia.notebooks` within your working directory.
4. Then, `cd` to that location in your terminal

   ```{code-block} bash
   cd lecture-julia.notebooks
   ```
5. Finally, you can open this directory from your terminal with the following

   ```{code-block} bash
   code .
   ```

Which should provide a screen such as 

```{figure} /_static/figures/vscode_intro_1.png
:width: 100%
```


```{admonition} Cloning Directly from VS Code
Alternatively, if you are already a user of Visual Studio Code, you can clone within VS Code by using the `> Git: Clone` command from the [command palette](command_palette).  See the lectures on [tools](../software_engineering/tools_editors.md) and [source code control](../software_engineering/version_control.md) for more details.
```

(install_packages)=
## Installing Packages

After you have the notebooks available, as in {ref}`above <clone_lectures>`, these lectures depend on functionality (like packages for plotting, benchmarking, and statistics) that are not installed with every Jupyter installation on the web.

As discussed above, you can start this directly from the [command palette](command_palette) with `<Ctrl+Shift+P>` then typing part of the `> Julia: Start REPL` command.
```{figure} /_static/figures/vscode_intro_2.png
:width: 75%
```

1. Start a REPL  it may do an initial compilation of some background packages, but will then look something like
 
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
If the cursor is instead `(@v1.6) pkg>` then you may not have started the integrated terminal in the correct location, or you used an external REPL.  Assuming that you are in the correct location, if you type `activate .` in the package mode, the cursor should change to `(quantecon-notebooks-julia) pkg>` as it activates this project file.

One benefit of using the integrated REPL is that it will set important options for launching Julia (e.g. the number of threads) and activate the local project files (i.e. the `Project.toml` file in the notebooks directory) automatically.  If you use an external REPL, you will need to set these manually.  Here you would want to run the REPL with `julia --project --threads auto`  to tell Julia to set the number of threads equal to your local machine's number of cores, and to activate the existing project.  See [here](starting-a-repl) for more details.
```


(running_jupyterlab)=
## Running JupyterLab

You can start Jupyter within any directory by executing the following in a terminal

```{code-block} bash
jupyter lab
```

This runs a process giving Jupyter permission to access the this directory, but not its parents.  This is especially convenient to do in VS Code since we have already navigated to this directory.



1. If the  Julia REPL still open, create a new terminal by clicking on the `+` button on the terminal pane and create a new terminal appropriate for your operating system.  Close the Julia REPL if you wish
   ```{figure} /_static/figures/vscode_intro_5.png
   :width: 75%
   ```

   - [As before](command_palette), if the terminal pane is not available, use ``<Ctrl+`>`` or `>View: Toggle Integrated Terminal` to see the pane.
   - You can close the Julia REPL if you wish, or create multiple temrinals in this interface

2. Within the new terminal, execute `jupyter lab`.  This should run in the background in this terminal, with output such as
   ```{figure} /_static/figures/vscode_intro_6.png
   :width: 100%
   ```



The process should launch a webpage on your desktop, which may look like

```{figure} /_static/figures/jupyterlab_first.png
:width: 100%
```

If it does not start automatically, use the link at the bottom of the output in the terminal (which should show with `Follow Link`).


Navigate to the {doc}`Interacting with Julia <../getting_started_julia/julia_environment>` notebook (the file `getting_started_julia/julia_environment.ipynb` in the list of notebooks in JupyterLab) to explore this interface and start writing code.

(reset_notebooks)=
## Refreshing the Notebooks after Modification

As you work through the notebooks, you may wish to reset these to the most recent version on the server.

1. To see this, modify one of the notebooks in Jupyter, and then go back to the VS Code, which should now highlight on the left hand side that one or more modified files have been modified.

2. Choose the highlighted "Source Control" pane, or use `<Ctrl+Shift+G>` then it will summarize all of the modified files.

3. To revert back to the versions you previously downloaded, right click on the "Changes" and then choose to 

```{figure} /_static/figures/vscode_intro_7.png
:width: 100%
```

Additionally, if the notebooks themselves are modified as the lecture notes evolve, you can first discard any changes, and then either use `> Git: Pull` command or click on the arrow next to "main" on the bottom of the screen to download the latest versions.

If the `Project.toml` or `Manifest.toml` files are modified, then you may want to redo the [instantiation](install_packages) step to ensure you have the correct versions.



We will explore these sorts of features, and how to use them for your own projects, in the {doc}`source code control <../software_engineering/version_control>` lecture.
