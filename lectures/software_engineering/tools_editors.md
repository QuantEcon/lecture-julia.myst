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

(tools_editors)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Visual Studio Code and Other Tools

```{contents} Contents
:depth: 2
```

While Jupyter notebooks are a great way to get started with the language, eventually you will want to use more powerful tools.  Visual Studio Code (VS Code) in particular, is the most popular open source editor for programming - with a huge set of extensions and strong industry support.

While you can use source code control, run terminals and the REPL ("Read-Evaluate-Print Loop") without VS Code, we will concentrate on using it as a full IDE for all of these features.

(install_vscode)=
## Installing VS Code

To install VS Code and the Julia Extension,

1. First, ensure you followed the instructions for setting up Julia {ref}`on your local computer <jl_jupyterlocal>`.
2. In particular, ensure you did the initial [VS Code setup](initial_vscode_setup)
3. Install [VS Code](https://code.visualstudio.com/) for your platform and open it

See the [Julia VS Code Documentation](https://www.julia-vscode.org/docs/dev/gettingstarted/#Installation-and-Configuration-1) for more details.

If you have done a typical Julia installation, then this may be all that is needed and no configuration may be necessary.  However, if you have installed Julia in a non-standard location you may need to manually set the executable path.  See [here](https://www.julia-vscode.org/docs/dev/gettingstarted/#Configuring-the-Julia-extension-1) for instructions if it errors when starting Julia terminals.

```{tip} "Open in Code" on MacOS and Linux
VS Code supports the "Open with Code" action in File Explorer if chosen during the installation.  This is convenient, but not necessary.  To support it on MacOS you need to install a [Automator script or separate package](https://stackoverflow.com/questions/64040393/open-a-folder-in-vscode-through-finder-in-macos).  Similarly, see [here](https://github.com/vvanloc/Nautilus-OpenInVSCode) for support on linux.
```

While the general [VS Code documentation](https://code.visualstudio.com/docs/getstarted/userinterface) is excellent, we will review a few of the key concepts directly.  In addition, see [here](optional_extensions) for other useful extensions when using VS Code not directly connected to these lectures.

### Optional Extensions and Settings 

Open the settings with `> Preferences: Open User Settings` (see above for opening the command palette with `<Ctrl-Shift-P>`).

As a few optional suggestions for working with the settings,

- In the settings, search for `Tab Size` and you should find `Editor: Tab Size` which you can modify to 4.
- Search for `quick open` and change `Workbench > Editor: Enable Preview from Quick Open` and consider setting it to false, though this is a matter of personal taste.
- Finally, if you are on Windows, search for `eol` and change `Files: Eol` to be `\n`.

A key feature of VS Code is that it can synchronize your extensions and settings across all of your computers, and even when used in-browser (e.g. with [GitHub CodeSpaces](https://github.com/features/codespaces)).  To turn on,
- Ensure you have a [GitHub account](https://github.com/), which will be useful for {doc}`further lectures <../software_engineering/version_control>`
- Choose `Turn On Settings Sync...` in the gear icon at the bottom of the screen
```{figure} https://code.visualstudio.com/assets/docs/editor/settings-sync/turn-on-sync.png
:width: 50%
```
- You can stick with all of the defaults, and read more in the [instructions](https://code.visualstudio.com/docs/editor/settings-sync)

(terminals)=
### Integrated Terminals in VS Code

A key benefit of VS Code is that you can use terminals and command-line interfaces directly within the editor, and starting in the convenient location relative to an opened project.

Furthermore, in the case of julia (and other languages such as python) this will activate the project automatically.  See the [documentation](https://code.visualstudio.com/docs/editor/integrated-terminal) for more details.

You can open the terminal panel with  `> View: Toggle Terminal` , typing ``Ctrl+` ``, or by clicking on the list of warnings and errors bottom bar in VS Code.  If no existing terminal exists, it will create a new one.


(vscode)=
## Using VS Code with Julia

The documentation for the VS Code extension provides many examples:
- [Creating a Hello World](https://www.julia-vscode.org/docs/dev/gettingstarted/#Creating-Your-First-Julia-Hello-World-program-1)
- [Debugging](https://www.julia-vscode.org/docs/dev/userguide/debugging/)
- [Integrated Plots](https://www.julia-vscode.org/docs/dev/userguide/plotgallery/)
- [Code Completion](https://www.julia-vscode.org/docs/dev/userguide/editingcode/)
- [Code Navigation](https://www.julia-vscode.org/docs/dev/userguide/codenavigation/)

In addition, there are excellent youtube videos about provide more background.  For example, [Package Development in VSCode](https://www.youtube.com/watch?v=F1R3ETaRQXY) shows advanced features.

### Hello World
To walk through a simple, but complete example.

Create a new directory on your computer, for example `hello_world` and then open VS Code in that folder  This can be done several ways,
   - Within a terminal on your operating system, navigate that directory and type `code .`
   - On Windows, right click on the folder and choose `Open with Code` - trusting the authors as required on opening the folder.
   - In the VS Code Menu, choose `File/Open Folder...`

Next, in the left hand panel, under `HELLO_WORLD`, right click and choose `New File` and create as `hello.jl`.  The screen should look something like

```{figure} /_static/figures/vscode_file_created.png
:width: 100%
```

Type some code as such `f(x) = x + 1` into the file, into the `.jl` file, save it, and while your mouse curser is on the line of code do `<Shift-Enter>`.  If a julia REPL wasn't already started, it will be started and the code will be run within its kernel

```{figure} /_static/figures/vscode_jl_function.png
:width: 100%
```

At this point, the function is available for use within either the code or the REPL.  You can get inline results by adding more code to the file and executing each line with `<Shift-Enter>`.

```{figure} /_static/figures/vscode_jl_function_2.png
:width: 100%
```

That code is also accessible within the REPL.  Executing the function there,

```{figure} /_static/figures/vscode_repl_1.png
:width: 100%
```

Because the REPL and the files are synchronized, you can modify functions and simple choose `<shift-Enter>` to update their definitions before analyzing the results in the REPL or in the file itself.


(adding_packages)=
### Adding Packages

Next we will go through simple use of the plotting and package management.

```{note}
VS Code typically activates the current project correctly.  However, when choosing to enter the package mode, if the prompt changes to `(@v1.6) pkg>` rather than `(hello_world) pkg >` then you will need to manually activate the project.  In that case, ensure that you are in the correct location and choose `] activate .`.

You can always see the current package location and details with `] st`.  See [Julia Environments](jl_packages) for more details.
```

The REPL.  First, type `]` to enter the package management mode, then `add Plots`.  Depending on whether you have done similar operations before, this may download a lot of dependencies.  See below for an example

```{figure} /_static/figures/vscode_package_added.png
:width: 100%
```

Crucially, you will notice that two new files are created. `Project.toml` and `Manifest.toml`.  These provide a snapshot of all of the packages used in this particular project.

Add code in the `.jl` file for a simple plot, and it will be shown on a separate pane

```{figure} /_static/figures/vscode_plots.png
:width: 100%
```

To exit package management mode and return to the REPL, type `Ctrl+C`. To then go from the REPL back to the VS Code terminal, type `Ctrl+D`.

### Executing Files

First we will reorganize our file so that it is a set of functions with a call at the end rather than a script.  Replace the code with
```{code-block} none
using Plots, Random

f(x) = x + 1
function plot_results()
    x = 1:5
    y = f.(x)
    plot(x, y)
    print(y)
end

# execute main function
plot_results()
```

While it is fine to use scripts with code not organized in functions for exploration, you will want to organize any serious computations inside of a function.  While this may not matter for every problem, this will let the compiler avoid global variables and highly optimize the results.

```{note}
The behavior of global variables accessed in loops in the `REPL`, Debugger, inline code evaluations, and Jupyter notebooks is different from executed files.  See the documentation on [soft scoping](https://docs.julialang.org/en/v1/manual/variables-and-scoping/#On-Soft-Scope) for more background.  A good heuristic which will avoid these issues is to (1) never use loops outside of a function when you write `.jl` files; and (2) if you ever use the `global` keyword, assume something is wrong and you should put something in a function.
```

You can execute a `.jl` file in several ways.

Within a terminal, you can provide the path to the file.  For example,
```{code-block} none
julia --threads auto --project hello.jl
```

See the [REPL](repl_main) section for more details on the commandline options.

Alternatively, within VS Code itself you can use the `<Ctrl-F5>` to run the new file.  


(debugger)=
### Using the Debugger

To debug your function, first click to the left of a line of code to create a breakpoint (seen as a red dot).

Next, use `<Ctrl-Shift-D>` or select the run and debug tab on the left in Julia to see
```{figure} /_static/figures/debugger_1.png
:width: 100%
```

Then choose the `Run and Debug` option and it will execute `plot_results()` at the bottom of the file, and then stop inside at the breakpoint. 

```{note}
Some users may see other options like `Run active Julia file` instead of `Run and Debug` in the run and debug tab. 
```

```{figure} /_static/figures/debugger_2.png
:width: 100%
```

You can use the toolbar at the top to step through the code, as described in the [documentation](https://www.julia-vscode.org/docs/dev/userguide/debugging/).


As an alternative way to start the debugger, you can instead debug a call within the REPL with commands like `@run plot_results()`.

```{note}
The Julia Debugger runs some of the code in an interpreted mode that might be far slower than typical  compiled and optimized code.  For this reason, it may not be possible to use it in all the same ways you might use something like the Matlab debugger.  However, if you organize your code appropriately, you may find that the [compile mode](https://www.julia-vscode.org/docs/stable/userguide/debugging/#Compile-mode-1) enables you to concentrate on debugging only some of the functions, while letting slow functions remain compiled.
```


(repl_main)=
## The REPL
Even if you are working primarily in `.jl` and/or Jupyter Notebooks in Julia, you will need to become comfortable with the REPL.  We saw some initial use of this when [adding packages](adding_packages) and exploring the code above, but the REPL has many [more features](https://docs.julialang.org/en/v1/stdlib/REPL/#The-Julia-REPL).

### Starting a REPL
There are several ways to start the REPL.
- Within VS Code, executing code inline will start it as required.
- In the command palette of VS Code, choose  `> Julia: Start REPL`
- Outside of VS Code, if julia was installed on your system path, you can simply type `julia` or with other options

The command line options for starting Julia are set to decent defaults for terminals running within VS Code, but you will want to set them yourself if starting the REPL otherwise.

As an example, the argument `--threads` determines the number of threads that Julia starts with.  If starting `julia` on the command line without specifying any arguments, it will default to 1 (or check an environment variable).  To have Julia automatically choose the number of threads based on the number of processors for your machine, pass the `--threads auto` argument.

```{note}
VS Code sets the number of threads automatically based on the number of cores on your machine, but the value can be modified in its `> Preferences: Open User Settings` and then search for `Julia: Num Threads`.
```

The most important choice is the `--project` toggle which determines whether you want to activate an existing project (or create a new one) when starting the interpreter.  Since a key feature of Julia is to have fully reproducible environments, you will want to do this whenever possible.

```{note}
A key difference between Julia and some other package managers is that it is capable of having different versions of each package for different projects - which ensures all projects are fully reproducible by you, your future self, and any other collaborators.  While there is a global set of packages available (e.g. `IJulia.jl` to ensure Jupyter support) you should try to keep the packages in different projects separated.  See the documentation on [environments](https://docs.julialang.org/en/v1/manual/code-loading/#Environments-1) and the [package manager](https://pkgdocs.julialang.org/v1/getting-started/) for more.
```

If you start the terminal without activating a project, you can activate it afterwards with `] activate .` or `using Pkg; Pkg.activate()`.

To see this in action, within an external terminal we will open it using both methods, and then use `] st` to see which project is activated and which packages are available.

First, with `julia --threads auto` we see that the globally installed packages are available at first, but that the existing `Project.toml` and `Manifest.toml` in that folder are chosen after we choose `] activate .`
```{figure} /_static/figures/repl_1.png
:width: 100%
```

Next, with `julia --threads auto --project` the project is automatically activated
```{figure} /_static/figures/repl_2.png
:width: 100%
```


Finally, if you choose the `--project` option in a folder which doesn't have an existing project file, it will create them as required.

A few other features of the REPL include,

### More Features and Modes

Hitting `;` brings you into shell mode, which lets you run bash commands (PowerShell on Windows)

```{code-block} none
; pwd
```

In addition, `?` will bring you into help mode.

The key use case is to find docstrings for functions and macros, e.g.

```{code-block} julia
? print
```


(jl_packages)=
## Package Environments

As discussed, the Julia package manager allowed you to create fully reproducible environments (in the same spirit as Python's and Conda virtual environments).

As we saw before, `]` brings you into package mode.  Some of the key choices are

* `] instantiate` (or `using Pkg; Pkg.instantiate()` in the normal julia mode) will check if you have all of the packages and versions mentioned in the `Project.toml` and `Manifest.toml` files, and install as required.
  - This feature will let you reproduce the entire environment and, if a `Manifest.toml` is available, the exact package versions used for a project.  For example, these lecture notes use [Project.toml](https://github.com/QuantEcon/lecture-julia.notebooks/blob/main/Project.toml) and [Manifest.toml](https://github.com/QuantEcon/lecture-julia.notebooks/blob/main/Manifest.toml) - which you likely instantiated during installation after downloading these notebooks.
* `] add Expectations` will add a package (here, `Expectations.jl`) to the activated project file (or the global environment if none is activated).
* Likewise, `] rm Expectations` will remove that package.
* `] st` will show you a snapshot of what you have installed.
* `] up` will upgrade versions of your packages to the latest versions possible given the graph of compatibility used in each.

```{note}
On some operating systems (such as OSX) REPL pasting may not work for package mode, and you will need to access it in the standard way (i.e., hit `]` first and then run your commands).
```

## More Options and Configuration Choices

VS Code and the related ecosystem have an enormous number of features and additional options.

The following are some optional choices, not all directly connected to Julia.

(optional_extensions)=
### Optional Extensions

While not required for these lectures, consider installing the following extensions.  As before, you can search for them on the Marketplace or choose `Install` from the webpages themselves.

1. [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter): VS Code increasingly supports Jupyter notebooks directly, and this extension provides the ability to open and edit `.ipynb` notebook files without installing Conda and running `jupyter lab`.
1. [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens): An extension that provides an enormous amount of detail on exact code changes within github repositories (e.g., seamless information on the time and individual who [last modified](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens#current-line-blame-) each line of code)
2. [GitHub Pull Requests and Issues](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github): while VS Code supports the git {doc}`version control <../software_engineering/version_control>` natively, these extension provides additional features for working with repositories on GitHub itself.
3. [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one): For editing the markdown format, such as `README.md` and similar files.
4. Finally, VS Code has an excellent [latex editor](https://github.com/James-Yu/LaTeX-Workshop#manual)
   - Install a recent version of miktex or texlive for your platform
   - Install [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) and (optionally) the [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) for spell checking
   - To get started, just add magic comments at the top of the latex file (removing the `!BIB` line if there is no bibtex reference) and then `F5` or the equivalent command to compile:

```{code-block} none
% !TEX program = pdflatex
% !BIB program = bibtex
% !TEX enableSynctex = true
```
### Font Choices

Beyond their general use, the integrated terminals will use fonts installed within VS Code.  Given that Julia code supports mathematical notation, the extra support in good fonts can be helpful.
- [JuliaMono](https://juliamono.netlify.app/download/) and  [Cascadia Code](https://github.com/microsoft/cascadia-code) and [Fira Code](https://github.com/tonsky/FiraCode) are all good options.
- You can adapt [these](https://juliamono.netlify.app/faq/#vs-code) or [these](https://techstacker.com/change-vscode-code-font/) instructions depending on the font choice.

If on Windows, for your external terminal consider installing [Windows Terminal](https://aka.ms/terminal), which is built to support [Cascadia](https://docs.microsoft.com/en-us/windows/terminal/cascadia-code) and [Powerline](https://docs.microsoft.com/en-us/windows/terminal/tutorials/powerline-setup) fonts.

### Remote and Collaborative Extensions

If you ever need to use clusters or work with reproducible [containers](https://code.visualstudio.com/docs/remote/containers), VS Code has strong support for those features.  Key extensions for these are:
- [VS LiveShare](https://marketplace.visualstudio.com/items?itemName=MS-vsliveshare.vsliveshare-pack): Collaborative coding within VS Code.
- [Remote Extensions Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack): Tools to access remote servers, local containers.  Install [OpenSSH](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse) if required.
- [SFTP](https://marketplace.visualstudio.com/iems?itemName=liximomo.sftp): Secure copying of files to supported cloud services.

Windows users will find good support to access a local linux installation with the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and the associated [VS Code Extension](https://code.visualstudio.com/docs/remote/wsl-tutorial).
