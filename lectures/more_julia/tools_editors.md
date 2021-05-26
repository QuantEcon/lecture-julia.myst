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

(tools_editors)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Julia Tools and Editors

```{contents} Contents
:depth: 2
```

Co-authored with Arnav Sood

While Jupyter notebooks are a great way to get started with the language, eventually you will want to use more powerful tools.

We'll discuss a few of them here, such as

* Text editors like Atom, which come with rich Julia support for debugging, documentation, git integration, plotting and inspecting data, and code execution.
* The Julia REPL, which has specialized modes for package management, shell commands, and help.

Note that we assume you've already completed the {doc}`getting started <../getting_started_julia/getting_started>` and {doc}`interacting with Julia <../getting_started_julia/julia_environment>` lectures.

## Preliminary Setup

Follow the instructions for setting up Julia {ref}`on your local computer <jl_jupyterlocal>`.

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

* Git integration (more on this in the {doc}`next lecture <../more_julia/version_control>`).
* Painless inspection of variables and data.
* Easily run code blocks, and drop in custom snippets of code.
* Integration with Julia documentation and plots.

### Installation and Configuration

#### Installing Atom

1. Download and Install Atom from the [Atom website](https://atom.io/).
1. (Optional, but recommended): Change default Atom settings
    * Use `Ctrl-,` to get the `Settings` pane
    * Choose the `Packages` tab
    * Type `line-ending-selector` into the Filter and then click "Settings" for that package
        * Change the default line ending to `LF` (only necessary on Windows)
    * Choose the Editor tab
        * Turn on `Soft Wrap`
        * Set the `Tab Length` default to `4`

#### Installing Juno

1. Use `Ctrl-,` to get the Settings pane.
1. Go to the `Install` tab.
1. Type `uber-juno` into the search box and then click Install on the package that appears.
1. Wait while Juno installs dependencies.
1. When it asks you whether or not to use the standard layout, click `yes`.

At that point, you should see a built-in REPL at the bottom of the screen and be able to start using Julia and Atom.

(atom_troubleshooting)=
#### Troubleshooting

Sometimes, Juno will fail to find the Julia executable (say, if it's installed somewhere nonstandard, or you have multiple).

To do this
1. `Ctrl-,` to get Settings pane, and select the Packages tab.
2. Type in `julia-client` and choose Settings.
3. Find the Julia Path, and fill it in with the location of the Julia binary.

* To find the binary, you could run `Sys.BINDIR` in the REPL, then add in an additional `/julia` to the end of the screen.
* e.g. `C:\Users\YOURUSERNAME\AppData\Local\Julia-1.0.1\bin\julia.exe` on Windows as `/Applications/Julia-1.0.app/Contents/Resources/julia/bin/julia` on OSX.

> 

See the [setup instructions for Juno](http://docs.junolab.org/latest/man/installation.html)  if you have further issues.

If you upgrade Atom and it breaks Juno, run the following in a terminal.

```{code-block} none
apm uninstall ink julia-client
apm install ink julia-client
```

If you aren't able to install `apm` in your PATH, you can do the above by running the following in PowerShell:

```{code-block} none
cd $ENV:LOCALAPPDATA/atom/bin
```

Then navigating to a folder like `C:\Users\USERNAME\AppData\Local\atom\bin` (which will contain the `apm` tool), and running:

```{code-block} none
./apm uninstall ink julia-client
./apm install ink julia-client
```

(upgrading_julia)=
#### Upgrading Julia

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
* A `registry` is a git repository corresponding to a list of (typically) registered packages, from which Julia can pull (for more on git repositories, see {doc}`version control <../more_julia/version_control>`).
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

```{literalinclude} _static/includes/deps_generic.jl
---
tags: [hide-output]
---
```

What this `github_project` function does is activate (and if necessary, download, instantiate and precompile) a particular Julia environment.

(docker_main)=
