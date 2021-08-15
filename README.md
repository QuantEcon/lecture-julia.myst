# lecture-julia.myst

Source for julia.quantecon.org

## Local Development 

### Setup

1. Download and install [Julia 1.6](https://julialang.org/downloads).

2. Install [`conda`](https://www.anaconda.com/products/individual)
    - See [Conda Installation](https://datascience.quantecon.org/introduction/local_install.html#installation) for examples
    - Add conda to path  

3. Install with [vscode](https://code.visualstudio.com/) and accept defaults if possible:
   - (Optional) some highly recommended packages .  After installation of vscode, you should be able to click `Install` link on the webpage of any extensions
      - [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server)
      - [Github Support](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github)
      - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
      - [Julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia)
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

9.  Install general julia packages if not already installed

    ```bash
    julia -e 'using Pkg; Pkg.add("IJulia");'
    ```

10.  Install Julia packages required for lecture notes

    ```bash
    julia --project=lectures --threads auto -e 'using Pkg; Pkg.instantiate();'
    ```

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

If you have [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) installed, then go to `_build/html/index.html` in the explorer, and right-click to choose `Live Preview: Show Preview`

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