# lecture-julia.myst

Source for julia.quantecon.org
## Local Development 

### Setup

1. Download and install [Julia 1.6](https://julialang.org/downloads).

2. Install [`conda`](https://www.anaconda.com/products/individual)
    - See [Conda Installation](https://datascience.quantecon.org/introduction/local_install.html#installation) for examples
    - Add conda to path  

3. Install [vscode](https://code.visualstudio.com/) and accept defaults if possible:
   - Some highly recommended packages.  After installation of vscode, you should be able to click `Install` link on the webpage of any extensions
      - [MyST-Markdown](https://github.com/executablebooks/myst-vs-code)
      - [Julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia)
   - Other optional, but recommended extensions
      - [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server)
      - [Github Support](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github)
      - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
      - [Editing Markdown](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
      - [Extra Git Tools](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)
      - [Spell Checking](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)
      - [YAML support](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)

   - Go to settings with `<Ctrl-Shift-P>` and search for the following settings to change:
      - `files.eol` to `\n`
      - `enablePreviewFromQuickOpen` to turn it off
      - `Tab Size` to `4`

4. If on Windows, install [git](https://git-scm.com/downloads) and run the following in a terminal

    ```bash
    git config --global core.eol lf
    git config --global core.autocrlf false
    ```

5. Clone this repository (in vscode, you can use `<Ctrl-Shift-P>` then `Clone` then `Clone from GitHub` then choose the repo as `https://github.com/QuantEcon/lecture-julia.myst`).  Or with github desktop, choose the `<> Code` dropdown on this website

6. Open this repository in vscode, either from Github Desktop  with `<Ctrl-Shift-A>` or with `code .` in the right folder in a terminal
    - After opening this repo, any terminals start at its root.

7. Start a vscode terminal with ``<Ctrl+`>`` or through any other method.  Create a conda environment.

    ```bash
    conda create -n lecture-julia.myst python=3.8
    conda activate lecture-julia.myst
    pip install -r requirements.txt
    ```

    This will install all the jupyterbook packages required to edit and build the lectures.

8.  Set the default interpreter for vscode's python to be the conda env
    - Go `<Ctrl-Shift-P>` then `Python: Select Interpreter`
    - Then choose the interpeter with `lecture-julia.myst` which should now be automatically activated in the terminal.
    - If the interpreter does not show up in the drop-down, close and reopen vscode, then try again. Alternatively, you can run this step at the end of the setup process.
        - Whenever reopening vscode,  re-run `conda activate lecture-julia.myst` to ensure the environment remains active.

9.  Install general julia packages if not already installed.

    ```bash
    julia -e 'using Pkg; Pkg.add("IJulia");'
    ```
    
    On Windows, you should run the following instead to avoid a quoting issue:
    
    ```bash
    julia -e "using Pkg; Pkg.add(\"IJulia\");"
    ```
    
    If the terminal responds with `'Julia' is not recognized`, close and reopen vscode, then try again. Make sure to re-activate the environment.

10.  Install Julia packages required for lecture notes.

     ```bash
     julia --project=lectures --threads auto -e 'using Pkg; Pkg.instantiate();'
     ```
     
     On Windows, run the following instead:
     
     ```bash
     julia --project=lectures --threads auto -e "using Pkg; Pkg.instantiate();"
     ```

**(Optional) REPL Integration**
With [MyST-Markdown](https://github.com/executablebooks/myst-vs-code) and [Julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) installed, you can ensure that `<Ctrl-Enter>` on lines of code are sent to a Julia REPL.
1.  Open Key Bindings with `<Ctrl-K Ctrl-S>`
2.  Search for the `Julia: Send Current Line or Selection to REPL` binding
3.  Right Click on the key binding with `juliamarkdown` on it, and choose `Change When Expression`, and change `juliamarkdown` to just `markdown`

## Executing Code in Markdown Files
If you installed the REPL Integration above, then in a `.md` file,

1. Start a Julia REPL with `> Julia: Start REPL`
2. Activate the project file in the REPL with `] activate lectures`
3. Then, assuming that you setup the keybindings above, you can send a line of code in the markdown to the REPL with `<Ctrl-Enter>`

Code can be executed line by line, or you can select a chunk of code and 
## Example Operations
### Building the lectures
To do a full build of the lectures:

```bash
jupyter-book build lectures
```

or

```bash
jb build lectures
```

This will take a while. But it will populate your cache, so future iteration is faster. 

On Windows, if you get the following error:

```
ImportError: DLL load failed while importing win32api: The specified procedure could not be found.
```

then run `conda install pywin32` and build the lectures again.

If you have [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) installed, then go to `_build/html/index.html` in the explorer, and right-click to choose `Live Preview: Show Preview`

### Cleaning Lectures
To clean up (i.e., delete the build)

```bash
jupyter-book clean lectures
```

or 

```bash
jb clean lectures
```

to clean the `execution` cache you can use

```bash
jb clean lectures --all
```
### Debugging Generated Content

After execution, you can find the generated `.ipynb` and `.jl` files in `_build/jupyter_execute` for each lecture.
- To see errors, you can open these in jupyterlab, the Jupyter support within VSCode, etc.
- If using the Julia REPL in VS Code, make sure to do `] activate lectures` prior to testing to ensure the packages are activated.  This is not necessary when opening in Jupyter.
- Finally, the code is written using interactive scoping, so `include(_build/jupyter_execute/dynamic_programming/mccall_model.jl)` etc. may not work.  However, `shift-enter` within VS Code to the REPL will work, and you can execute these with [SoftGlobalScope.jl](https://github.com/stevengj/SoftGlobalScope.jl) if strictly required.
