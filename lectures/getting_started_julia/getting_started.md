---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.12
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

As the most popular and best-supported open-source code editor, it provides a large number of useful features and extensions.  We will begin to use it as a primary editor in the {doc}`tools lecture <../software_engineering/tools_editors>`.

## TL;DR
For those with more experience, Julia and the lectures can be installed without any installation Jupyter or Python:

1. Install [Git](https://git-scm.com/install/)
2. Install [VS Code](https://code.visualstudio.com/)
3. Install Julia following the [Juliaup instructions](https://github.com/JuliaLang/juliaup#installation)
   - Windows: `winget install julia -s msstore` in a terminal
   - Linux/Mac: `curl -fsSL https://install.julialang.org | sh` in a terminal
4. Install the [VS Code Julia extension](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia)
5. In VS Code, open the command palette with `<Ctrl+Shift+P>` and type `> Git: Clone` to clone the repository `https://github.com/quantecon/lecture-julia.notebooks` in a new window
6. Start a Julia REPL in the integrated terminal with `> Julia: Start REPL` from the command palette, then enter package mode with `]` and then type `instantiate`.
  - That process will take several minutes to download and compile all of the packages used in the lectures.
7. Open any of the `.ipynb` files in VS Code and select the `Julia` in the `Julia 1.12 channel` challenge (i.e., not the Jupyter channels if you have them installed) when prompted to run the notebooks directly within VS Code

At that point, you can directly move on to the {doc}`julia by example <../getting_started_julia/julia_by_example>` lecture.


## A Note on Jupyter

Like Python and R, and unlike products such as Matlab and Stata, there is a looser connection between Julia as a programming language and Julia as a specific development environment.

While you will eventually use other editors, there are some advantages to starting with the [Jupyter](http://jupyter.org/) environment while learning Julia.

* The ability to mix formatted text (including mathematical expressions) and code in a single document.
* Nicely formatted output including tables, figures, animation, video, etc.
* Conversion tools to generate PDF slides, static HTML, etc.

We'll discuss the workflow on these features in the [next section](julia_environment)

For those with little to no programming experience (e.g. you have never used a loop or "if" statement) see the list of [introductory resources](intro_resources).

(jl_jupyterlocal)=
## Desktop Installation of Julia and Jupyter

In this section, we will describe the installation of Julia and Jupyter on your desktop.


(install_jupyter)=
### Installing Jupyter
[Anaconda](https://www.anaconda.com/) provides an easy to install package of jupyter, python, and many data science tools.

If you have not previously installed Conda or Jupyter, then 
1. Download the binary (<https://www.anaconda.com/download/>) and follow the [installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for your platform.
2. If given the option for your operating system, let Conda add Python to your PATH environment variables.

```{note}
While Conda is the easiest way to install jupyter, it is not strictly required.  With any python you can install with `pip install jupyter`.  More advanced users should consider switching to the [uv](https://github.com/astral-sh/uv) package manager.

In addition, there is direct support for [Jupyter notebooks in VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) with **no Python installation**. See {ref}`VS Code Julia Kernel  <running_vscode_kernel>`.

```

(intro_repl)=
### Install Julia
After Jupyter is installed, you can install Julia.

```{note}
The "official" installation for Julia is now [Juliaup](https://github.com/JuliaLang/juliaup), which makes it easier to upgrade and manage concurrent Julia versions.  See [here](https://github.com/JuliaLang/juliaup#using-juliaup) for a list of commands, such as `juliaup update` to upgrade to the latest available Julia version after installation, or ways to switch to newer Julia versions after they are released.

**Troubleshooting:** On Mac/Linux, if you have permissions issues on the installation use `sudo curl -fsSL https://install.julialang.org | sh`.  If there are still permissions issus, [see here](https://github.com/JuliaLang/juliaup/wiki/Permission-problems-during-setup) suggests executing `sudo chown $(id -u):$(id -g) ~/.bashrc`, `sudo chown $(id -u):$(id -g) ~/.zshrc`, and `sudo chown $(id -u):$(id -g) ~/.bash_profile` then retry the installation.
```

1. Download and install Julia following the [Juliaup instructions](https://github.com/JuliaLang/juliaup#installation)
    - Windows: easiest method is `winget install julia -s msstore` in a terminal
    - Linux/Mac: in a terminal use `curl -fsSL https://install.julialang.org | sh`.  To open a terminal on macOS press `Cmd + Space` to open Spotlight, then type `Terminal`, or use the Launchpad
    - If you have previously installed Julia manually, you will need to uninstall previous versions before switching to `juliaup`.
    - Alternatively, can manually install from [download page](http://julialang.org/downloads/), accepting all default options

2. Open Julia, by either
    - Navigating to Julia through your menus or desktop icons (Windows, Mac), or
    - Opening a terminal and type `julia` (which should work for all OS if you used `juliaup`.  Otherwise see [here](https://julialang.org/downloads/platform/))

   You should now be looking at something like this
   
   ```{figure} /_static/figures/julia_term_1.png
   :width: 100%
   ```
   
   This is called the JULIA *REPL* (Read-Evaluate-Print-Loop), which we discuss more {ref}`later <repl_main>`.

3. In the Julia REPL, hit `]` to enter package mode and then enter:

   ```{code-block} julia
   add IJulia
   ```

   This adds packages for the `IJulia` kernel which links Julia to Jupyter you previously installed with Anaconda (i.e., allows your browser to run Julia code, manage Julia packages, etc.).

4. You can exit the julia REPL by hitting backspace to exit the package mode, and then 

   ```{code-block} julia
   exit()
   ```

```{note}
As entering of the package mode is so common in these notes, we will denote this with a `] IJulia`, etc.  On Windows and in Jupyter, you can directly copy this into your terminal, whereas on Linux and macOS you may need to manually enter the package mode.
```

(initial_vscode_setup)=
## Setting up Git and VS Code

A primary benefit of using [open-source](https://en.wikipedia.org/wiki/Open-source_software) languages such as Julia, Python, and R is that they can enable far better workflows for both collaboration and [reproducible research](https://en.wikipedia.org/wiki/Reproducibility#Reproducible_research).

Reproducibility will ensure that you, your future self, your collaborators, and eventually the public will be able to run the exact code with the identical environment with which you provided the results - or even roll back to a snapshot in the past where the results may have been different in order to compare.

We will explore these topics in detail in the lectures on {doc}`source code control <../software_engineering/version_control>` and {doc}`continuous integration and test-driven development <../software_engineering/testing>`, but it is worth installing and beginning to use these tools immediately.


First, we will install [Git](more_on_git), which has become the industry standard open-source version-control tool.  This lets you download both the files and the entire version history from a server (e.g. on GitHub) to your desktop.


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

```{tip}
You can type partial strings for different commands and it helps you to find features of vscode and its extensions.  Furthermore, the command palette remembers your most recent and common commands.
```


[Integrated Terminals](https://code.visualstudio.com/docs/editor/integrated-terminal) within VS Code are a convenient because they are local to that project, detect hypertext links, and provide better fonts.

To launch a terminal, use either (1) ``<Ctrl+`>``, (2) `>View: Toggle Terminal` with the command palette, or (3) `View > Terminal` in the menus.


```{note}
Becoming comfortable with VS Code and tools for source code control/software engineering is an essential step towards ensuring reproducibility.  An easy way to begin that process is to start using VS Code to edit LaTeX, and practice managing your `.tex` files in GitHub rather than dropbox or similar alternatives.  While not directly connected to Julia, this familiarity will make everything easier - even for proprietary languages such as Stata and Matlab.  See [here](vscode_latex) for instructions on this setup process.
```


(clone_lectures)=
## Downloading the Notebooks

Next, let's install the QuantEcon lecture notes to our machine and run them (for more details on the tools we'll use, see our lecture on {doc}`version control <../software_engineering/version_control>`).

With VS Code installed, you can easily clone the lecture notes repository

1. Open the command palette with `<Ctrl+Shift+P>` and type `> Git: Clone`
2. For the Repository URL, provide enter `https://github.com/quantecon/lecture-julia.notebooks`
Alternatively, if you are already a user of Visual Studio Code, you can clone within VS Code by using the `> Git: Clone` command from the [command palette](command_palette).  See the lectures on [tools](../software_engineering/tools_editors.md) and [source code control](../software_engineering/version_control.md) for more details.
3. Choose the location to clone when prompted
   - The workflow will be easiest if you clone the repo to the default location relative to the home folder for your user.
   - For example, on Windows a good choice might be `c:\Users\YOURUSERNAME\Documents\GitHub` or simply `c:\Users\YOURUSERNAME\Documents`.  On linux and macOS, your home directory `~` or `~/GitHub`.
4. Accept the option to open in a new window when prompted

```{admonition} Cloning without VS Code

To use the command-line to begin the introduction to source control tools.

1. Choose and create if necessary a convenient parent folder where you would like the notebooks directory, see above for suggestions.
2. Open a new terminal for your machine and navigate to the parent folder of where you wish to store the notebooks.
   - On Windows: if using the [Windows Terminal](https://aka.ms/terminal) you can simply right-click on the directory in the File Explorer and choose to "Open in Microsoft Terminal" or, alternatively "Git Bash Here" to use the terminal provided by Git.  On macOS, see [here](https://apple.stackexchange.com/questions/11323/how-can-i-open-a-terminal-window-directly-from-my-current-finder-location) for a discussion of different approaches.
3. Execute the following code in the terminal to download the entire suite of notebooks associated with these lectures with `git clone https://github.com/quantecon/lecture-julia.notebooks`
   This will download the repository with the notebooks into the directory `lecture-julia.notebooks` within your working directory.
4. Then, `cd` to that location in your terminal with `cd lecture-julia.notebooks`
5. Finally, you can open this directory from your terminal in VS Code with `code .` or manually opening
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
If the cursor is instead `(@v1.12) pkg>` then you may not have started the integrated terminal in the correct location, or you used an external REPL.  Assuming that you are in the correct location, if you type `activate .` in the package mode, the cursor should change to `(quantecon-notebooks-julia) pkg>` as it activates this project file.

One benefit of using the integrated REPL is that it will set important options for launching Julia (e.g. the number of threads) and activate the local project files (i.e. the `Project.toml` file in the notebooks directory) automatically.  If you use an external REPL, you will need to set these manually.  Here you would want to run the REPL with `julia --project --threads auto`  to tell Julia to set the number of threads equal to your local machine's number of cores, and to activate the existing project.  See [here](repl_main) for more details.
```


(running_jupyterlab)=
## Running JupyterLab

You can start Jupyter within any directory by executing the following in a terminal

```{code-block} bash
jupyter lab
```

This runs a process giving Jupyter permission to access this directory, but not its parents.  This is especially convenient to do in VS Code since we have already navigated to this directory:

1. If the Julia REPL is still open, create a new terminal by clicking on the `+` button on the terminal pane and create a new terminal appropriate for your operating system.  Close the Julia REPL if you wish.
   ```{figure} /_static/figures/vscode_intro_5.png
   :width: 75%
   ```

   - [As before](command_palette), if the terminal pane is not available, use ``<Ctrl+`>`` or `>View: Toggle Terminal` to see the pane.
   - You can close the Julia REPL if you wish, or create multiple terminals in this interface

2. Within the new terminal, execute `jupyter lab`.  This should run in the background in this terminal, with output such as
   ```{figure} /_static/figures/vscode_intro_6.png
   :width: 100%
   ```



The process should launch a webpage on your desktop, which may look like

```{figure} /_static/figures/jupyterlab_first.png
:width: 100%
```

If it does not start automatically, use the link at the bottom of the output in the terminal (which should show with `Follow link (ctrl + click`).


Proceed to the next section on [Jupyter](julia_environment) to explore this interface and start writing code.

(reset_notebooks)=
## Refreshing the Notebooks after Modification

As you work through the notebooks, you may wish to reset these to the most recent version on the server.

1. To see this, modify one of the notebooks in Jupyter, and then go back to VS Code, which should now highlight on the left hand side that one or more modified files have been modified.

2. Choose the highlighted "Source Control" pane, or use `<Ctrl+Shift+G>`; then it will summarize all of the modified files.

3. To revert back to the versions you previously downloaded, right click on "Changes" and then choose `Discard All Changes`:

```{figure} /_static/figures/vscode_intro_7.png
:width: 100%
```

Additionally, if the notebooks themselves are modified as the lecture notes evolve, you can first discard any changes, and then either use `> Git: Pull` command or click on the arrow next to "main" on the bottom left of the screen to download the latest versions. Here "main" refers to the main branch of the repo where the latest versions are hosted.

If the `Project.toml` or `Manifest.toml` files are modified, then after reverting you will want to redo the step to ensure you have the correct versions.  To do this, run the project, enter package management mode with `]`, ensure that `(quantecon-notebooks-julia) pkg>` is displayed, and then type `instantiate` to install the correct versions.

We will explore these sorts of features, and how to use them for your own projects, in the {doc}`source code control <../software_engineering/version_control>` lecture.


(julia_environment)=
## Interacting with Julia

Next, we'll start examining different features of the Julia and Jupyter environments.

While we emphasize a [local installation of Jupyter](jl_jupyterlocal), other alternatives exist.

For example,
- Some universities may have JupyterHub installations available - which provide a hosted Jupyter environment.  However, it would require the hub to have explicit Julia support.
- VS Code has rapidly progressing [support for Jupyter](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) using an existing Jupyter installation.
- The combination of the new [VS Code Jupyter](optional_extensions) and [VS Code Julia](install_vscode) extensions supports Jupyter notebooks without even a fragile Conda/python installation
- Online services such as [JuliaHub](https://juliahub.com/lp/) provide a tailored experience for Julia.  Be warned, however, that [Colab](https://colab.research.google.com/) and others are only designed for Python, and adding Julia requires a great deal of effort.

## Using Jupyter

(ipython_notebook)=
### Getting Started

```{note}
The easiest way to get started with these notebooks is to follow the {ref}`cloning instructions <clone_lectures>` earlier.  
```

Launch `jupyter lab` and navigate to this notebook(i.e. `getting_started_julia/getting_started.ipynb` )

See [here](running_jupyterlab) for the previous instructions on launching Jupyter Lab.  

Your web browser should open to a page that looks something like this

```{figure} /_static/figures/starting_nb_julia.png
:width: 100%
```

The page you are looking at is called the "dashboard".

If you click on "Julia 1.x.x" under "Notebook" you should have the option to start a Julia notebook.

Here's what your Julia notebook should look like

```{figure} /_static/figures/nb2_julia.png
:width: 100%
```

The notebook displays an *active cell*, into which you can type Julia commands.

### Notebook Basics

Notice that in the previous figure the cell is surrounded by a blue border.

This means that the cell is selected, and double-clicking will place it in edit mode.

As a result, you can type in Julia code and it will appear in the cell.

When you're ready to execute these commands, hit `Shift-Enter`

```{figure} /_static/figures/nb3_julia.png
:width: 100%
```

#### Modal Editing

The next thing to understand about the Jupyter notebook is that it uses a *modal* editing system.

This means that the effect of typing at the keyboard **depends on which mode you are in**.

The two modes are

1. Edit mode
    * Indicated by a blue border around one cell, as in the pictures above.
    * Whatever you type appears as is in that cell.
1. Command mode
    * The blue border disappears and turns into a plain grey border.
    * Key strokes are interpreted as commands --- for example, typing b adds a new cell below  the current one.

(To learn about other commands available in command mode, go to "Keyboard Shortcuts" in the "Help" menu)

#### Switching modes

* To switch to command mode from edit mode, hit the `Esc` key.
* To switch to edit mode from command mode, hit `Enter` or click in a cell.

The modal behavior of the Jupyter notebook is a little tricky at first but very efficient when you get used to it.

#### Plots

Run the following cell

```{code-cell} julia
using Plots
plot(sin, -2π, 2π, label = "sin(x)")
```

You'll see something like this (although the style of plot depends on your
installation)

```{figure} /_static/figures/nb4_julia.png
:width: 100%
```

```{attention}
If this code fails to work because the `Plots` package is missing, then either you

  1. did not [install the packages](install_packages) in the previous lecture
     - You should go back and follow the [install the packages](install_packages) instructions, or just call `using Pkg; Pkg.instantiate()` in a new cell.
  2. downloaded or moved this notebook rather than [cloning the notebook repository](clone_lectures).  In that case, it does not have the associated `Project.toml` file local to it.
     - Consider [cloning the notebook repository](clone_lectures) instead.
     - If you would prefer not, then you can manually install packages as you need them.  For example, in this case you could type `] add Plots` into a code cell in the notebook or into your Julia REPL.
```

### Working with the Notebook

Let's go over some more Jupyter notebook features --- enough so that we can press ahead with programming.

#### Tab Completion

Tab completion in Jupyter makes it easy to find Julia commands and functions available.

For example if you type `rep` and hit the tab key you'll get a list of all
commands that start with `rep`

```{figure} /_static/figures/nb5_julia.png
:width: 100%
```

(gs_help)=
#### Getting Help

To get help on the Julia function such as `repeat`, enter `? repeat`.

Documentation should now appear in the browser

```{figure} /_static/figures/repeatexample.png
:width: 100%
```

#### Other Content

In addition to executing code, the Jupyter notebook allows you to embed text, equations, figures and even videos in the page.

For example, here we enter a mixture of plain text and LaTeX instead of code

```{figure} /_static/figures/nb6_julia.png
:width: 100%
```

Next we `Esc` to enter command mode and then type `m` to indicate that we
are writing [Markdown](http://daringfireball.net/projects/markdown/), a mark-up language similar to (but simpler than) LaTeX.

(You can also use your mouse to select `Markdown` from the `Code` drop-down box just below the list of menu items)

Now we `Shift + Enter` to produce this

```{figure} /_static/figures/nb7_julia.png
:width: 100%
```

#### Inserting Unicode (e.g. Greek letters)

Julia supports the use of [unicode characters](https://docs.julialang.org/en/v1/manual/unicode-input/)
such as `α` and `β` in your code.

Unicode characters can be typed quickly in Jupyter using the `tab` key.

Try creating a new code cell and typing `\alpha`, then hitting the `tab` key on your keyboard.

There are other operators with a mathematical notation.  For example, the `LinearAlgebra` package has a `dot` function as identical to the latex `\cdot`.

```{code-cell} julia
using LinearAlgebra
x = [1, 2]
y = [3, 4]
@show dot(x, y)
@show x ⋅ y;
```

#### Shell Commands

You can execute shell commands (system commands) in Jupyter by prepending a semicolon.

For example, `; ls` will execute the UNIX style shell command `ls`,
which --- at least for UNIX style operating systems --- lists the
contents of the current working directory.

These shell commands are handled by your default system shell and hence are platform specific.

#### Package Operations

You can execute package operations in the notebook by prepending a `]`.

For example, `] st` will give the status of installed packages in the current environment.

**Note**: Cells where you use `;` and `]` must not have any other instructions in them (i.e., they should be one-liners).

### Sharing Notebooks

Notebook files are just text files structured in [JSON](https://en.wikipedia.org/wiki/JSON) and typically end with `.ipynb`.

A notebook can easily be saved and shared between users --- you just need to
pass around the `ipynb` file.

To open an existing `ipynb` file, import it from the dashboard (the first
browser page that opens when you start Jupyter notebook) and run the cells or edit as discussed above.

The Jupyter organization has a site for sharing notebooks called [nbviewer](http://nbviewer.jupyter.org/)
which provides a static HTML representations of notebooks.

QuantEcon also hosts the [QuantEcon Notes](http://notes.quantecon.org/) website, where you can upload and share your notebooks with other economists and the QuantEcon community.

(running_vscode_kernel)=
## Running the VS Code Julia Kernel
The Jupyter Extension for VS Code supports Julia directly without the need for a Python installation.

With cloned notebooks, open a `.ipynb` file directly in VS Code.  Depending on your setup, you may see a request to choose the kernel for executing the notebook.

To do so, click on the `Choose Kernel` or `Select Another Kernel...` which may display an option such as

   ```{figure} /_static/figures/vscode_julia_kernel.png
   :width: 80%
   ```

Choose the `Julia`  kernel, rather than the `Jupyter Kernel...` to bypass the Python Jupyter setup.  If successful, you will see the kernel name as `Julia 1.12 channel` or something similar.

With the kernel selected, you will be able to run cells in the VS Code UI with similar features to Jupyter Lab.  For example, below shows the results of play icon next to a code cell, which will `Execute Cell` and display the results inline.


   ```{figure} /_static/figures/vscode_jupyter_kernel_execute.png
   :width: 100%
   ```
