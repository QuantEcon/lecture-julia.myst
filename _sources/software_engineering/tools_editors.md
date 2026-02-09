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

See [Modern Julia Workflows](https://modernjuliaworkflows.org/) for alternative approaches.

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
- If you are on Windows, search for `eol` and change `Files: Eol` to be `\n`.

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
VS Code typically activates the current project correctly.  However, when choosing to enter the package mode, if the prompt changes to `(@v1.12) pkg>` rather than `(hello_world) pkg >` then you will need to manually activate the project.  In that case, ensure that you are in the correct location and choose `] activate .`.

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
```{code-block} julia
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
```{code-block} bash
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

To emphasize this point, this is an example of the `]st ` showing the global environment has only the bare minimum of packages installed.  With this workflow, all other packages are installed only when a given project is activated.
```{code-block} none
(@v1.12) pkg> st
Status `~/.julia/environments/v1.12/Project.toml`
  [7073ff75] IJulia v1.30.6
  [14b8a8f1] PkgTemplates v0.7.56
  [295af30f] Revise v3.10.0
```

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

```{code-block} julia
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
* `] add Distributions` will add a package (here, `Distributions.jl`) to the activated project file (or the global environment if none is activated).
* Likewise, `] rm Distributions` will remove that package.
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
2. [GitHub Pull Requests and Issues](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github): while VS Code supports the git {doc}`version control <../software_engineering/version_control>` natively, these extension provides additional features for working with repositories on GitHub itself.
3. [GitHub Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot): AI-assisted code completion.  See [GitHub Education Pack](https://education.github.com/pack) for free access if you are a student or educator.
4. [OpenAI Codex](https://marketplace.visualstudio.com/items?itemName=openai.chatgpt): official ChatGPT-powered coding agent for VS Code.  Works best if you already have ChatGPT Plus/Pro (or higher) and provides Copilot-like chat, edits, and inline help inside the editor.
5. [Claude Code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code): Anthropic's agentic coding tool that can edit files, run commands, and search your codebase.  Available as a VS Code extension.  See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) for setup and usage.

(llm_instructions)=
## Using LLMs with VS Code

GitHub Copilot, Google Gemini Code Assist, OpenAI Codex, and Claude Code can provide completions, refactors, and documentation suggestions directly in VS Code.

For consistent answers across a project you can add short "instructions" files that the assistants read to understand your style and constraints:

- **GitHub Copilot**: Place `.github/copilot-instructions.md` in the repo root to define project-wide guidance (file name and location are fixed; see the [Copilot docs](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)). You can also create a personal `global-copilot-instructions.md` in your user settings directory if you want defaults across repos.
- **Google Gemini**:
  - **VS Code**: Install the [Gemini Code Assist](https://cloud.google.com/code/docs/vscode/install) extension (part of Google Cloud Code). While Gemini does not yet auto-ingest a hidden config file, the convention is to place a `GEMINI.md` in your repo root and explicitly reference it (e.g., keep it open or `@mention` it) at the start of a session.
  - **CLI**: You can use the [Google Cloud CLI](https://cloud.google.com/sdk/gcloud/reference/gemini) (`gcloud gemini ...`) for pipe-based operations, or use the API directly. For repo-chat on the command line, many use community tools (like `llm` or `aider`) configured with a Gemini API key.
- **OpenAI Codex**: Codex looks for `AGENTS.md` (or `AGENTS.override.md`) starting from your repo root and up to the current directory, plus a global copy at `~/.codex/AGENTS.md` (Codex home defaults to `~/.codex`, but you can point `CODEX_HOME` elsewhere if you want it under `~/.openai`). You can add other fallback names—such as `instructions.md` at the repo root—via `project_doc_fallback_filenames` in `~/.codex/config.toml`. See the [AGENTS.md guide](https://developers.openai.com/codex/guides/agents-md) for details.
- **Claude Code**: Place a `CLAUDE.md` file in your repo root to provide project-specific instructions.  Claude Code automatically reads this file at the start of each session.  You can also create a personal `~/.claude/CLAUDE.md` for global defaults across all projects, and folder-level `CLAUDE.md` files for directory-specific guidance.  Beyond instructions, Claude Code supports "skills"—custom slash commands (e.g., `/commit`, `/review-pr`) that you define in `~/.claude/commands/` or `.claude/commands/` within a project. See the [Claude Code memory documentation](https://docs.anthropic.com/en/docs/claude-code/memory) for details on `CLAUDE.md` and the [slash commands guide](https://docs.anthropic.com/en/docs/claude-code/slash-commands) for creating custom skills.

A minimal example you can adapt (drop this same content into `.github/copilot-instructions.md`, `GEMINI.md`, `AGENTS.md`, and/or `CLAUDE.md`):

```markdown
## Project context
- Julia lectures and demos targeting Julia 1.12.

## Style
- Follow the SciML style guide where applicable: [https://docs.sciml.ai/SciMLStyle/stable/](https://docs.sciml.ai/SciMLStyle/stable/).
- Use snake_case for files, functions, and variables; 4-space indent; concise comments.
- Prefer pure functions and keep logic inside modules.

## Tests
- Always add or update unit tests with code changes.
- Recommend `julia --project -e 'using Pkg; Pkg.test()'` before claiming completion; if tests are slow, state what would be run.

## Structure
- Keep the existing layout:
  - /src for package code
  - /test for tests
  - Project.toml and Manifest.toml tracked and updated only as needed
- Avoid adding new top-level folders unless requested.

## Safety
- Avoid destructive git commands; prefer minimal diffs (apply_patch) and cite file paths/lines.
```

### Sharing Instructions Across Tools with Symlinks

To avoid maintaining duplicate content across multiple instruction files, you can create one canonical file and use symbolic links for the others.  For example, using `AGENTS.md` as the primary file:

```{code-block} bash
# Create symlinks (macOS/Linux)
ln -s AGENTS.md CLAUDE.md
ln -s AGENTS.md GEMINI.md
mkdir -p .github && ln -s ../AGENTS.md .github/copilot-instructions.md
```

With this setup, you only edit `AGENTS.md` and all other tools read the same content through their symlinks.

```{note}
**Windows users**: Symlinks require extra setup.  Enable Developer Mode (Settings → Privacy & security → For developers → toggle "Developer Mode" on), then configure Git with `git config --global core.symlinks true`.  You must clone the repository fresh after enabling these settings for symlinks to work correctly.
```

(claude_github_actions)=
### Using Claude for Code Review with GitHub Actions

Beyond using LLMs inside your editor, you can integrate Claude directly into your GitHub workflow so that it automatically reviews pull requests, responds to questions in issues, and even implements changes — all triggered by mentioning `@claude` in a comment.

This is powered by [Claude Code GitHub Actions](https://github.com/anthropics/claude-code-action), an official GitHub Action maintained by Anthropic.  Once installed, any collaborator can tag `@claude` in a PR comment or issue and receive an AI-powered response that follows your project's `CLAUDE.md` guidelines.

#### Quick Setup

The fastest way to set up the action is from the Claude Code CLI:

```{code-block} bash
claude            # start Claude Code
/install-github-app   # guided installer
```

This command walks you through installing the [Claude GitHub app](https://github.com/apps/claude), adding your `ANTHROPIC_API_KEY` as a repository secret, and creating the workflow file.

```{note}
You must be a **repository admin** to install the GitHub app and add secrets.  The app requests read & write permissions for Contents, Issues, and Pull requests.
```

#### Manual Setup

If you prefer to configure things yourself:

1. Install the Claude GitHub app at [github.com/apps/claude](https://github.com/apps/claude) and grant it access to your repository.
2. Add your `ANTHROPIC_API_KEY` to the repository secrets (Settings → Secrets and variables → Actions).
3. Create a workflow file at `.github/workflows/claude.yml`:

```{code-block} yaml
name: Claude Code
on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
jobs:
  claude:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
```

With this workflow in place, mentioning `@claude` in any PR or issue comment triggers the action.  For example:

```{code-block} none
@claude review this PR for correctness and style
@claude how should I handle the edge case in this function?
@claude fix the type instability in the inner loop
```

#### Automated Review on Every PR

You can also configure Claude to review every pull request automatically, without needing an `@claude` mention:

```{code-block} yaml
name: Code Review
on:
  pull_request:
    types: [opened, synchronize]
jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: "/review"
          claude_args: "--max-turns 5"
```

See the [Claude Code GitHub Actions documentation](https://docs.anthropic.com/en/docs/claude-code/github-actions) and the [action repository](https://github.com/anthropics/claude-code-action) for the full set of configuration options, including use with AWS Bedrock and Google Vertex AI.

(vscode_latex)=
## VS Code as a LaTeX Editor

VS Code has an outstanding LaTeX editing extension, which provides a good way to become comfortable with the tool and managing source code online.
1. Install a recent copy of tex
   - Typically Windows users would want [MiKTeX](https://miktex.org/download)
   - macOS and Linux users can also use use either [MiKTeX](https://miktex.org/download) or [TeX Live](https://www.tug.org/texlive/)
   - If you install MiKTeX, ensure you choose the "Always install missing packages on-the-fly" option
2. With a new VS Code session, install the following extensions
   - [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
   - (Optional) [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)
3. (Optional) If you wish to have the editor automatically compile and display the document when you save your `.tex` file: Then open the VS Code settings, and search for `autobuild` to set the option `Latex-workshop > Latex > Auto Build: Run` to `onSave`.

No further configuration should be required, but see [the manual](https://github.com/James-Yu/LaTeX-Workshop/wiki/Install) if you have problems.


While there many ways to execute a compilation workflow, one method is to use "magic comments" at the top of a latex file.  This is not specific to LaTeX Workshop, and can be used by other tools.
1. In VS Code, create a new file, such as `rough_notes.tex` and copy in the following
   ```{code-block} latex
   % !TEX program = pdflatex
   % !TEX enableSynctex = true

   \documentclass{article}
   \title{Rough Notes}
   \begin{document}
      \maketitle
      Some rough notes
   \end{document}
   ```
2. Save the document.  If you enabled the automatic build option in your settings, this should compile it.  Otherwise, use `F5` or the command palette `> Latex Workshop: Build Latex Project`.

   If you hover over the magnifying glass icon near the top right hand corner, It should look something like, 
   ```{figure} /_static/figures/vscode_latex_1.png
   :width: 100%
   ```

3. Click on that link, or use `<Ctrl+Alt+V>` to get display the PDF Preview.
   - The first time you do this, it will ask you to choose the PDF display. Choose `VSCode tab`
   - If you modify the document and save (or manually rebuild) the view will update
   - If you double-click on the PDF it can take you to the synced section of the latex.  Conversely, if you use the `<Ctrl+Alt+J>` or the palette `> TeX Workshop: SyncTex from Cursor` it will find the appropriate section of the PDF based on your current cursor position in the `.tex` file.

4. Select the Problems pane at the bottom of the screen (which likely shows 1 warning and no error) or open it with `<Ctrl+Shift+M>`

```{figure} /_static/figures/vscode_latex_2.png
:width: 100%
```

In that screenshot, we have also selected the `TeX` pane on the left hand side, which provides additional options for you to explore.  All of them have command-palette equivalents.

If you wanted to have a bibliography, you would add it into the tex file as normal, and just add in the additional magic comment `% !BIB program = bibtex`

Finally, when using source code control, you will want to make sure you add the intermediate files to your `.gitignore` (see [here](discarding_changes) for more).  Typically, you would want to ignore
```{code-block} none
*.aux
*.log
*.synctex.gz
*.pdf
```
Ignoring the `*.pdf` is optional but strongly encouraged as it will ensure you don't clog the repository with the binary pdf files, and make collaboration easier by preventing clashes on this file.
