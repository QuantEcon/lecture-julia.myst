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

# Visual Studio Code and other Julia Tools

```{contents} Contents
:depth: 2
```

While Jupyter notebooks are a great way to get started with the language, eventually you will want to use more powerful tools.  Visual Studio Code (VS Code) in particular, is the most popular open source editor for programming, and has a huge set of extensions and strong industry support.

While you can use source code control, run terminals/REPLs, without VS Code, we will concentrate on using it as a full IDE for all of these features.

## Installing VS Code

To install VS Code and the Julia Extension,

1. Follow the instructions for setting up Julia {ref}`on your local computer <jl_jupyterlocal>`.
2. Install [VS Code](https://code.visualstudio.com/) for your platform and open it
3. Install the [VS Code Julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) extension
   - After installation of VS Code, you should be able to choose `Install` on the webpage of any extensions and it will open on your desktop.
   - Otherwise, open the extensions with `<Ctrl-Shift-X>` or selecting extensions in the left-hand side of the VS Code window.  Then search for `Julia` in the Marketplace

See the [Julia VS Code Documentation](https://www.julia-vscode.org/docs/dev/gettingstarted/#Installation-and-Configuration-1) for more details.

If you have done a typical Julia installation, then this may be all that is needed and no configuration may be necessary.  However, if you have installed Julia in a non-standard location you may need to manually set the executable path.  See [here](https://www.julia-vscode.org/docs/dev/gettingstarted/#Configuring-the-Julia-extension-1) for instructions if it errors when starting Julia terminals.

### Optional Extensions

While not required for these lectures, consider installing the following extensions.  As before, you can search for them on the Marketplace or choose `Install` from the webpages themselves.

1. [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter): VS Code increasingly supports Jupyter notebooks directly, and this extension provides the ability to open and edit `.ipynb` notebook files wihtout running `jupyter lab`, etc.
1. [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens): An extension that provides an enormous amount of detail on exact code changes within github repositories (e.g., seamless information on the time and individual who [last modified](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens#current-line-blame-) each line of code)
2. [GitHub Pull Requests and Issues](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github): while VS Code supports the git {doc}`version control <../software_engineering/version_control>` natively, these extension provides additional features for working with repositories on GitHub itself.
3. [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one): For editing the markdown format, such as `README.md` and similar files.
4. Finally, VS Code is an excellent latex editor.  To install support,
   - Install a recent version of miktex or texlive for your platform
   - Install [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
   - Optionally install [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) for spell checking
   - See [documentation](https://github.com/James-Yu/LaTeX-Workshop#manual). The easiest use is to put the magic comment `!TEX program = pdflatex` at the top of a `.tex` file.
   - 


See the [VS Code documentation](https://code.visualstudio.com/docs/getstarted/userinterface) for an introduction.  

A key feature is the [Command Palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette), which can be accessed with `<Ctrl+Shift+P>`.

```{figure} [/_static/figures/juno-standard-layout.png](https://code.visualstudio.com/assets/docs/getstarted/userinterface/commands.png)
:width: 50%
```

With this, you can type partial strings for different commands and it helps you to find features of vscode and its extensions.  This is so common that in these notes we
denote opening the command palette and searching for a command with things like `> Julia: Start REPL` , etc.  You will only need to type part of the string, and the command palette remembers
your most commonly used 


### Optional Extensions and Settings 


Open the settings, either with `> Preferences: User Settings` (recall this can be found with the palette and `<Ctrl-Shift-P>`) or with the menu.

As a few optional suggestions for working with the settings,

- In the settings, search for `Tab Size` and you should find `Editor: Tab Size` which you can modify to 4.
- Search for `quick open` and change `Workbench > Editor: Enable Preview from Quick Open` and consider setting it to false, though this is a matter of personal taste.
- Finally, if you are on Windows, search for `eol` and change `Files: Eol` to be `\n`.




(repl_main)=
## The REPL

Previously, we discussed basic use of the Julia REPL ("Read-Evaluate-Print Loop").

Here, we'll consider some more advanced features.

### Shell Mode

Hitting `;` brings you into shell mode, which lets you run bash commands (PowerShell on Windows)

```{code-cell} julia
; pwd
```

You can also use Julia variables from shell mode

```{code-cell} julia
x = 2
```

```{code-cell} julia
; echo $x
```

### Package Mode

Hitting `]` brings you into package mode.

* `] add Expectations` will add a package (here, `Expectations.jl`).
* Likewise, `] rm Expectations` will remove that package.
* `] st` will show you a snapshot of what you have installed.
* `] up` will (intelligently) upgrade versions of your packages.
* `] precompile` will precompile everything possible.
* `] build` will execute build scripts for all packages.
* Running `] preview` before a command (i.e., `] preview up`) will display the changes without executing.

You can get a full list of package mode commands by running

```{code-cell} julia
] ?
```

On some operating systems (such as OSX) REPL pasting may not work for package mode, and you will need to access it in the standard way (i.e., hit `]` first and then run your commands).

### Help Mode

Hitting `?` will bring you into help mode.

The key use case is to find docstrings for functions and macros, e.g.

```{code-block} julia
? print
```

Note that objects must be loaded for Julia to return their documentation, e.g.

```{code-block} julia
? @test
```

will fail, but

```{code-block} julia
using Test
```

```{code-block} julia
? @test
```

will succeed.

## Atom

As discussed {doc}`previously <../getting_started_julia/getting_started>`, eventually you will want to use a fully fledged text editor.

The most feature-rich one for Julia development is [Atom](https://atom.io/), with the [Juno](http://junolab.org/) package.

There are several reasons to use a text editor like Atom, including

(upgrading_julia)=
### Upgrading Julia

To get a new release working with Jupyter, run (in the new version's REPL)

```{code-block} julia
] add IJulia
] build IJulia
```

This will install (and build) the `IJulia` kernel.

To get it working with Atom, open the command palette and type "Julia Client: Settings."

Then, in the box labelled "Julia Path," enter the path to yor Julia executabe.

You can find the folder by running `Sys.BINDIR` in a new REPL, and then add the `/julia` at the end to give the exact path.

For example:

```{figure} /_static/figures/julia-path.png
:width: 100%
```

### Standard Layout

If you follow the instructions, you should see something like this when you open a new file.

If you don't, simply go to the command palette and type "Julia standard layout"

```{figure} /_static/figures/juno-standard-layout.png
:width: 100%
```

The bottom pane is a standard REPL, which supports the different modes above.

The "workspace" pane is a snapshot of currently-defined objects.

For example, if we define an object in the REPL

```{code-cell} julia
x = 2
```

Our workspace should read

```{figure} /_static/figures/juno-workspace-1.png
:width: 100%
```

The `ans` variable simply captures the result of the last computation.

The `Documentation` pane simply lets us query Julia documentation

```{figure} /_static/figures/juno-docs.png
:width: 100%
```

The `Plots` pane captures Julia plots output (the code is as follows)

```{code-block} julia
using Plots
gr(fmt = :png);
data = rand(10, 10)
h = heatmap(data)
```

```{figure} /_static/figures/juno-plots.png
:width: 100%
```

**Note:** The plots feature is not perfectly reliable across all plotting backends, see [the Basic Usage](http://docs.junolab.org/latest/man/basic_usage.html) page.

### Other Features

* `` Shift + Enter `` will evaluate a highlighted selection or line (as above).
* The run symbol in the left sidebar (or `Ctrl+Shift+Enter`) will run the whole file.

See [basic usage](http://docs.junolab.org/latest/man/basic_usage.html) for an exploration of features, and  the [FAQ](http://docs.junolab.org/latest/man/faq.html) for more advanced steps.

(jl_packages)=
## Package Environments

Julia's package manager lets you set up Python-style "virtualenvs," or subsets of packages that draw from an underlying pool of assets on the machine.

This way, you can work with (and specify) the dependencies (i.e., required packages) for one project without worrying about impacts on other projects.

* An `environment` is a set of packages specified by a `Project.toml` (and optionally, a `Manifest.toml`).
* A `registry` is a git repository corresponding to a list of (typically) registered packages, from which Julia can pull (for more on git repositories, see {doc}`version control <../software_engineering/version_control>`).
* A `depot` is a directory, like `~/.julia`, which contains assets (compile caches, registries, package source directories, etc.).

Essentially, an environment is a dependency tree for a project, or a "frame of mind" for Julia's package manager.

* We can see the default (`v1.1`) environment as such

```{code-cell} julia
] st
```

* We can also create and activate a new environment

```{code-cell} julia
] generate ExampleEnvironment
```

* And go to it

```{code-cell} julia
; cd ExampleEnvironment
```

* To activate the directory, simply

```{code-cell} julia
] activate .
```

where "." stands in for the "present working directory".

* Let's make some changes to this

```{code-cell} julia
] add Expectations Parameters
```

Note the lack of commas

* To see the changes, simply open the `ExampleEnvironment` directory in an editor like Atom.

The Project TOML should look something like this

```{code-block} none
name = "ExampleEnvironment"
uuid = "14d3e79e-e2e5-11e8-28b9-19823016c34c"
authors = ["QuantEcon User <quanteconuser@gmail.com>"]
version = "0.1.0"

[deps]
Expectations = "2fe49d83-0758-5602-8f54-1f90ad0d522b"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
```

We can also

```{code-cell} julia
] precompile
```

**Note** The TOML files are independent of the actual assets (which live in `~/.julia/packages`, `~/.julia/dev`, and `~/.julia/compiled`).

You can think of the TOML as specifying demands for resources, which are supplied by the `~/.julia` user depot.

* To return to the default Julia environment, simply

```{code-cell} julia
] activate
```

without any arguments.

* Lastly, let's clean up

```{code-cell} julia
; cd ..
```

```{code-cell} julia
; rm -rf ExampleEnvironment
```

### InstantiateFromURL

With this knowledge, we can explain the operation of the setup block


What this `github_project` function does is activate (and if necessary, download, instantiate and precompile) a particular Julia environment.

(docker_main)=
