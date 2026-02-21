# WSL Editor Setup
Editing on Windows is not entirely supported due to python/sphinx issues.  Instead, on Windows use  WSL with VS Code.

## Setup WSL and VS Code

1. To get "Ubuntu on Windows" and other linux kernels see [instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

- Fresh install quick path (PowerShell as Administrator):
    ```
    wsl --install -d Ubuntu
    ```
    - **Note:** When typing your UNIX password during setup, nothing will appear on screen. This is normal — just type and press Enter.

    - If installation fails, ensure required features are enabled (Powershell as administrator):
        ```
        dism /online /get-features /format:table | findstr /i "Microsoft-Windows-Subsystem-Linux"
        dism /online /get-features /format:table | findstr /i "VirtualMachinePlatform"
        ```
        - If either is Disabled, enable and reboot:

        ```
         dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
         dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

        ```

2. Install [VSCode](https://code.visualstudio.com/) with remote WSL support on windows
 - See [VS Code Remote Editing](https://code.visualstudio.com/docs/remote/remote-overview)
 - [VS Code Remote WSL](https://code.visualstudio.com/docs/remote/wsl#_opening-a-terminal-in-wsl)
3. Clone this repository using VS Code on Windows
4. After you open it, choose `> WSL: Reopen in WSL Window` from the command palette (Ctrl+Shift+P) if you have setup WSL/etc. properly
    - If you are successful, the bottom left corner of VS Code will show `>< WSL:Ubuntu>`
    - Now any terminals you run should be the linux ubuntu terminals

5. Assuming you have `git` installed and setup properly on windows, to get git credentials integrated, in a windows terminal (i.e. not in WSL) run
    ```
    git config --global credential.helper wincred
    ```
    - Then in a WSL terminal (within VS Code or otherwise),
    ```
    git config --global user.email "you@example.com"
    git config --global user.name "Your Name"
    git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-wincred.exe"
    ```
    - (See more details in [Sharing Credentials](https://code.visualstudio.com/docs/remote/troubleshooting#_sharing-git-credentials-between-windows-and-wsl) )


## Setup Linux Packages/etc.

1. Start within your home directory in linux or in a WSL terminal

2. In a WSL terminal, Go to your home directory and make sure key dependencies are installed
    ```bash
    cd ~
    sudo apt update
    sudo apt-get upgrade
    sudo apt install make gcc unzip
    sudo apt-get update
    sudo apt install -y libxt6 libxrender1 libgl1 libgl1-mesa-dri libqt5widgets5
    ```
3. Install conda with the updated version of
```
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh
```
   - accept default paths
   - accept licensing terms
   - *IMPORTANT* Manually choose `yes` to have it do the `conda init`
4. Install julia with [juliaup](https://github.com/JuliaLang/juliaup?tab=readme-ov-file#mac-linux-and-freebsd)
    ```bash
    curl -fsSL https://install.julialang.org | sh
    ```
5. Install key VSCode extensions (which are a separate set of installations for WSL vs. Windows)
    - Search for `python, julia, MyST-Markdown, Jupyter` and if you have support `GitHub Copilot, GitHub Copilot Chat`
    - Usually requires restart VS Code
6. In a terminal, create a virtual environment for conda and install packages
    ```bash
    conda create -n lecture-julia.myst python=3.11
    conda activate lecture-julia.myst
    pip install -r requirements.txt
    ```
7. Can try `> Python: Select Interpreter` and choose the `lecture-julia.myst` environment to make it automatic, but it does not always work.
8. Start a WSL terminal, make sure that the `lecture-julia.myst` is activated (and use `conda activate lecture-julia.myst` if not) then install the version-free Julia Jupyter kernel:
    ```bash
    julia -e 'using Pkg; Pkg.add("IJulia"); using IJulia; installkernel("Julia", "--project=@.", specname="julia")'
    ```
    This registers a kernel named `julia` (without a version suffix) that the notebooks expect, and includes `--project=@.` so it automatically activates the nearest `Project.toml`.
9. Install the Julia packages required for the lecture notes, running in a terminal (assuming still at the root)
    ```bash
    julia --project=lectures --threads auto -e 'using Pkg; Pkg.instantiate();'
    ```
    - This should take 10-15 minutes the first time.

At this point, you should be ready to follow the instructions in the main [README.md](./README.md)
