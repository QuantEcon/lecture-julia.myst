# lecture-julia.myst
Source for julia.quantecon.org

## Development 

### Setup

1. Download and install [Julia 1.6](https://julialang.org/downloads).

2. Install [`conda`](https://www.anaconda.com/products/individual).

3. Download this repo in VSCode, and create a conda environment for it: 

```
conda env create -f environment.yml
```
This will install all packages required to edit and build the lectures.

**Note**: Make sure you activate this environment whenever working on the lectures, by running `conda activate lecture-datascience`

4. In VSCode, open Julia REPL, install `IJulia`, then activate the project in `lectures` and install all of its packages:

```
] add IJulia
] activate lectures
] instantiate
```

5. Try building the lectures:

```bash
jupyter-book build lectures
```
This will take a while. But it will populate your cache, so future iteration is faster. 

4. To clean up (i.e., delete the build.)

```
jupyter-book clean lectures
```
