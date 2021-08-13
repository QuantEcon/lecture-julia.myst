# lecture-julia.myst
Source for julia.quantecon.org

## Content Development Installation Instructions (Not for End-users)
### WSL and VSCode if on Windows
- To get "Ubuntu on Windows" and other linux kernels see [instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
  - For the installation, run it in `Powershell` as an administrator
  - Hint on copy-paste:  One way to paste into a a windows (of any sort) is the `<ctrl-c>` text somewhere else and then, while selected in the terminal at the cursor, to `<right click>` the mouse (which pastes).
  - Use WSL2 and see https://docs.microsoft.com/en-us/windows/wsl/wsl2-kernel for manual installation instructions if required.
- Install [VSCode](https://code.visualstudio.com/) with remote WSL support on windows
 - See [VS Code Remote Editing](https://code.visualstudio.com/docs/remote/remote-overview)
 - [VS Code Remote WSL](https://code.visualstudio.com/docs/remote/wsl#_opening-a-terminal-in-wsl)


To open the WSL in VS Code
- Click on the "><" icon on the bottom left hand corner, and open the remote folder in your WSL image (e.g. `~/lecture-source-jl`)
- Choose "TERMINAL" to open a [WSL terminal](https://code.visualstudio.com/docs/remote/wsl#_opening-a-terminal-in-wsl), and run any of the above jupinx or make commands.


To get git credentials integrated, in a windows terminal (i.e. not in WSL) run
```
git config --global credential.helper wincred
```
Then in a WSL terminal (within VS Code or otherwise),
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-wincred.exe"
```
(See more details in [Sharing Credentials](https://code.visualstudio.com/docs/remote/troubleshooting#_sharing-git-credentials-between-windows-and-wsl) )

### Installation
1. Start within your home directory in linux or in a WSL terminal
2. Install Conda

   -  In the Ubuntu terminal, first install python/etc. tools
   ```bash
   cd ~
   wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
   bash Anaconda3-2021.05-Linux-x86_64.sh
   ```
   -  Create a directory `.conda` by running `mkdir ~/.conda` if the warning "Unable to register the environment" shows up
3. The installation will take time. You should:
- accept default paths
- accept licensing terms
- *IMPORTANT* Manually choose `yes` to have it do the `conda init`
- Delete the installation file
    ```bash
    pip install --upgrade --force-reinstall pyzmq
    rm Anaconda3-2021.05-Linux-x86_64.sh
    cd -
    ```
4. Install and activate the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate lecture-datascience    
    ```
4. Either manually run the steps in [setup.sh](deps/setup.sh) or execute it in a terminal within the cloned repo
   ```bash
   bash deps/setup.sh
   ```

## Building the Lectures
To build the lectures and cache all results:
```bash
jupyter-book build lectures
```