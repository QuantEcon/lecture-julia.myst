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

(testing)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Packages, Testing, and Continuous Integration

```{contents} Contents
:depth: 2
```

```{epigraph}
A complex system that works is invariably found to have evolved from a simple system that worked. The inverse proposition also appears to be true: A complex system designed from scratch never works and cannot be made to work. You have to start over, beginning with a working simple system -- [Gall's Law](https://en.wikipedia.org/wiki/John_Gall_%28author%29#Gall%27s_law).
```

Co-authored with Arnav Sood

This lecture discusses structuring a project as a Julia module, and testing it with tools from GitHub.

Benefits include

* Specifying dependencies (and their versions) so that your project works across Julia setups and over time.
* Being able to load your project's functions from outside without copy/pasting.
* Writing tests that run locally, *and automatically on the GitHub server*.
* Having GitHub test your project across operating systems, Julia versions, etc.

**Note:** Throughout this lecture, important points and sequential workflow steps are listed as bullets.

The goal of this style of coding is to ensure that [test driven development](test_driven) is possible, which will help ensure that your research is reproducible by yourself, your future self, and others - while avoiding accidental mistakes that can silently break old code and change results.

It furthermore, it makes collaboration on code much more feasible - allowing individuals to work on different parts of the code without fear of breaking others work.

Finally, since many economics projects occur over multiple years - this will help your future self to reproduce your results and make changes without remembering all of the subtle tricks required to get things operational.


## Introduction and Setup

Much of the software engineering and continuous integration (CI) will be done through [GitHub Actions](https://github.com/features/actions).

The GitHub actions execute as isolated and fully reproducible environments on the cloud (using containers), and are initiated through various actions on the GitHub.  In particular, these are typically initiated through:
- Making commits on the main branch of a repository
- Creating a PR on a repository, or pushing commits to it.
- [Tagging a release](https://docs.github.com/en/github/administering-a-repository/releasing-projects-on-github/managing-releases-in-a-repository) of a repository, which provides a snapshot at a particular commit.

For publicly available repositories, GitHub provides free minutes for executing these actions, whereas there are limits on the execution time for private repositories - though signing up for the GitHub academic plans previous discussed (i.e, the [Student Developer Pack](https://education.github.com/pack/) or [Academic/Researcher plans](https://help.github.com/articles/about-github-education-for-educators-and-researchers/)) provides additional free minutes.

While you may think of a ``Package'' as a shared and maintained set of reusable code, with Julia it will turn out to be the most useful way to organize personal and private projects or the code associated with a particular paper.  The primary benefit of using a package workflow is **reproducibility** for yourself, your future self, and collaborators.

In addition, you will find the testing workflow as an essential element of working on even individual projects, since it will prevent you from making accidental breaking changes at points in the future - or when you come back to the code after several years.

(testing_account_setup)=
### GitHub and Codecov Account Setup

This lecture assumes you have completed the setup of VS Code and Git in the  {doc}`version control <../software_engineering/version_control>` and  {doc}`tools <../software_engineering/tools_editors>`  lectures.  In particular,
- Ensure you have setup a GitHub account, and connected it to VS Code.
- Installed and setup VS Code's Julia extension.
- Set ` git config --global user.email "you@example.com"`, `git config --global user.name "Your Name"` and `git config --global github.user "YOURGITHUBNAME"` in your terminal.


The only other service that is necessary for the complete software engineering stack is a code coverage provider.

For these lectures, visit [Codecov website](https://about.codecov.io/sign-up/).

Installation instructions are [here](https://docs.codecov.com/docs/quick-start#getting-started)

To summarize: sign up, and sign into it with `GitHub`.  You may need to provide permissions for Codecov to access GitHub, follow the provided authorization instructions.

```{figure} /_static/figures/codecov-1.png
:width: 100%
```
<!-- 
TODO: I think this has changed and we should do it later in the workflow?

Next, for any private repositories you can click "add a repository" and *enable private scope* (this allows Codecov to service your private projects).

The result should be

```{figure} /_static/figures/codecov-2.png
:width: 50%
``` 
-->

(testing_pkg_installation)=
### PkgTemplates.jl and Revise.jl

While you can create your Julia package manually, using a template will ensure that you have everything in the standard format.

If you have activated the notebook repositories, then `PkgTemplates.jl` will already be installed.

Otherwise, start a `julia` REPL outside of a particular project (or do an `] activate` to deactivate the existing project, and use the global environment) and
* Install [PkgTemplates](https://github.com/invenia/PkgTemplates.jl/) and [Revise](https://github.com/timholy/Revise.jl) with 
```{code-block} none
] add PkgTemplates Revise
```

While we typically insist on having very few packages in the global environment, and always working with a `Project.toml` file, `PkgTemplates` is primarily used without an existing project.

While the [Revise.jl](https://github.com/timholy/Revise.jl) package is optional, it is strongly encouraged since it automatically compiling functions in your package as they are changed.  See [below](editing_package_code) for more.


(project_setup)=
## Project Setup

To create a project, first choose the parent directory you wish to create the project

* In an external terminal, navigate to this parent directory.
```{note}
On Windows, given that you have installed Git you could right click on the folder in explorer and choose `Git Bash Here`.  Even better, install the [Windows Terminal](https://docs.microsoft.com/en-us/windows/terminal/get-started) and choose `Open in Windows Terminal`.
```

* Start a julia terminal with `julia`

* Then

```{code-block} none
using PkgTemplates
```

* Create a *template* for your project.

This specifies metadata like the license we'll be using (MIT by default), the location, etc.


We will create this with a number of useful options, but see [the documentation](https://invenia.github.io/PkgTemplates.jl/stable/user/#A-More-Complicated-Example-1) for more.

```{code-block} none
t = Template(;dir = ".", julia = v"1.6",
              plugins = [
                Git(; manifest=true, branch = "main"),
                Codecov(),
                GitHubActions(),
                !CompatHelper,
                !TagBot
              ])
```

```{note}
If you did not set the `github.user` in the setup, you may need to pass in `user = YOURUSERNAME` as an additional argument.  In addition, this turns off some important features (e.g. `CompatHelper` and `TagBot`) and leaves out others (e.g. `Documenter{GitHubActions}`) which you would want for a more formal package.

Alternatively, `PkgTemplates` has an interactive mode, which you can prompt with `t = Template(;interactive = true)` to choose all of your selections within the terminal.
```

* Create a specific project based off this template

```{code-block} none
t("MyProject")
```

* Open the project in VS Code by right-clicking in the explorer for the new `MyProject` folder, or exiting the Julia REPL (via `exit()`) and then
```{code-block} none
cd MyProject
code .
```

The project should open in VS Code, and look something like 

```{figure} /_static/figures/new_package_vscode.png
:width: 100%
```

### Adding a Project to GitHub

The next step is to add this project to Git version control.

* First, we will need to create an empty GitHub Repository of the same name (but with a `.jl` extension).   Choose the `+` button to "Create a new repository"

```{figure} /_static/figures/new_package_vscode_2.png
:width: 100%
```


In particular, ensure that 

* The repo you create should have the same name as the project we added (except with a `.jl` extension).
* We should leave the boxes unchecked for the `README.md`, `LICENSE`, and `.gitignore`, since these are handled by `PkgTemplates`.
* If available, Grant access to `CodeCov` for this new repository.

Then, we can now publish the generated package in VS Code to this empty repository.

Choose the icon to publish the branch as we did in previous lectures

```{figure} /_static/figures/new_package_vscode_3.png
:width: 100%
```

Do not accept the request to create a `Pull Request` when you push the branch.

At which point, if you refresh the webpage, it should be filled with the generated files
```{figure} /_static/figures/new_package_vscode_4.png
:width: 100%
```

Furthermore, you will see a badge either saying that `CI | passing` or `CI | unknown` next to another for `codecov | unknown`.

These badges provide a summary of the current state of the `main` branch relative to the GitHub Actions, as we will see below.  If a repository has broken any tests on the main branch, it will show `CI | failed` in red.

(add_to_global)=
### (Optionally) Adding the Package to the Global Environment

Optionally, you may wish to be able to use your package within other projects on your computer.

To do this, you can add it to the main environment by starting Julia in the `MyProject` folder, and without activating the project (i.e. just `julia`).

Then

```{code-block} none
] dev .
```

Given this, other julia code can use `using MyProject` and, because the global environment stacks on any project files, it will be available.

You can see the change reflected in our default package list by running `] st`

```{code-block} none
      Status `C:\Users\jesse\.julia\environments\v1.6\Project.toml`
  [7073ff75] IJulia v1.23.2
  [a361046e] MyProject v0.1.0 `..\..\..\Documents\GitHub\MyProject`
  [14b8a8f1] PkgTemplates v0.7.18
  [295af30f] Revise v3.1.19  
```

For more on the package mode, see the {doc}`tools and editors <../software_engineering/tools_editors>` lecture.

However, this step not required if you wish to use this package as a self-contained project.
## Project Structure

Let's unpack the structure of the generated project

* A hidden directory, `.git`, holds the version control information.
* The `src` directory contains the project's source code -- it should contain only one file (`MyProject.jl`), which reads
  
  ```{code-block} none
  module MyProject
  
  greet() = print("Hello World!")
  
  end # module
  ```
  
* Likewise, the `test` directory should have only one file (`runtests.jl`), which reads
  
  ```{code-block} none
  using MyProject
  using Test
  
  @testset "MyProject.jl" begin
      # Write your own tests here.
  end
  ```
  

In particular, the workflow is to export objects we want to test (`using MyProject`), and test them using Julia's `Test` module.

The other important text files for now are

* `Project.toml` and `Manifest.toml`, which contain dependency information.

In particular, the `Project.toml` contains a list of dependencies, and the `Manifest.toml` specifies their exact versions and sub-dependencies.

* The `.gitignore` file (which may display as an untitled file), which contains files and paths for `git` to ignore.

### GitHub Actions and CI

The final file is a GitHub Actions file in the `.github/workflows` folder, called `CI.yml` with the text similar to
```{code-block} none
name: CI
on:
  - push
  - pull_request
jobs:
  test:
    name: Julia ${{ matrix.version }} - ${{ matrix.os }} - ${{ matrix.arch }} - ${{ github.event_name }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        version:
          - '1.6'
          - 'nightly'
        os:
          - ubuntu-latest
        arch:
          - x64          
    steps:
      - uses: actions/checkout@v2
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.version }}
          arch: ${{ matrix.arch }}
      - uses: actions/cache@v1
        env:
          cache-name: cache-artifacts
        with:
          path: ~/.julia/artifacts
          key: ${{ runner.os }}-test-${{ env.cache-name }}-${{ hashFiles('**/Project.toml') }}
          restore-keys: |
            ${{ runner.os }}-test-${{ env.cache-name }}-
            ${{ runner.os }}-test-
            ${{ runner.os }}-
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v1
        with:
          file: lcov.info
```

This file provides the rules for continuous integration running on changes to this repository.  You will not need to modify it.

To summarize some of the features, the

* `push` and `pull_request` at the top says that any push to the repository or creation of a pull request will trigger this workflow.
* `matrix` establishes the set of operating systems, architectures, and Julia versions to test
* `actions/cache` speeds up execution by storing dependent packages and artifacts
* `julia-runtest` will execute the `test/runtests.jl` automatically and determine whether it is successful
* `codecov/codecov-action` analyzes which lines of code were executed by `test/runtests.jl` and uploads to Codecov

## Project Workflow

### Dependency Management

#### Environments

As {ref}`before <jl_packages>`, the .toml files define an *environment* for our project, or a set of files which represent the dependency information.

The actual files are written in the [TOML language](https://github.com/toml-lang/toml), which is a lightweight format to specify configuration options.

This information is the name of every package we depend on, along with the exact versions of those packages.

This information (in practice, the result of package operations we execute) will
be reflected in our `MyProject.jl` directory's TOML, once that environment is activated (selected).

This allows us to share the project with others, who can exactly reproduce the state used to build and test it.

See the [Pkg docs](https://docs.julialang.org/en/v1/stdlib/Pkg/) for more information.

#### Pkg Operations

For now, let's just try adding a dependency.  Recall the package operations described in the {doc}`tools and editors <../software_engineering/tools_editors>` lecture.

* Within VS Code, start a REPL (e.g. `> Julia: Start REPL`), type `]` to enter the Pkg mode.  Verify that the cursor says `(My Project) pkg>` to ensure that it has activated your particular project.  Otherwise, if it says `(@1.6) pkg>`, then you have launched Julia outside of VSCode or through a different mechanism, and you might need to ensure you are in the correct directory and then `] activate .` to activate the project file in it.
* Then, add in the `Expectations.jl` package

```{figure} /_static/figures/vscode_add_package.png
:width: 100%
```

After installing a large web of dependencies, this tells Julia to write the results of package operations to the activated `Project.toml` file, which should now include text such as
```{code-block} none
[deps]
Expectations = "2fe49d83-0758-5602-8f54-1f90ad0d522b"
```

From this point, anytime the project is activated, the [Expectations.jl](https://github.com/QuantEcon/Expectations.jl) package will be available with `using Expectations`.

Furthermore, different projects could use different versions of this package.   We could add additional instructions on [compatability](https://pkgdocs.julialang.org/dev/compatibility/) within the file if we choose.

But almost more importantly, the `Manifest.toml` file now contains a complete snapshot of the particular `Expectations` package and every other dependency used in your project.

For example, the following is an example snippet of one manifest here:
```{code-block} none
[[Expectations]]
deps = ["Compat", "Distributions", "FastGaussQuadrature", "LinearAlgebra", "SpecialFunctions"]
git-tree-sha1 = "0f906c5edffe266acbf471734ac942d4aa9b7235"
uuid = "2fe49d83-0758-5602-8f54-1f90ad0d522b"
version = "1.7.1"

[[FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "5829b25887e53fb6730a9df2ff89ed24baa6abf6"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.4.7"
```
This shows that that the current manifest has:
- version 1.7.1 of the `Expectations` packages
- the `Expectations` package at this version has a variety of dependencies (i.e. `deps = `) such as the `FastGaussQuadrature`.
- the `FastGaussQuadrature` package at version `0.4.7` which itself has `StaticArrays` as a dependency, etc.

The manifest file then provides a fully reproducible environment with which someone can install not only the specific version of this package, but also a feasible network of all dependencies which all fulfill the compatability requirements of each.

We have used this feature before in our setup process, where we activated a `Project.toml` and used `] instantiate`.  If there is a manifest file in the same directory as the package, then that command will install the exact set of versions listed in the manifest for all packages.  If a manifest file does not exist, then it will attempt to find a compatible set of packages that fulfill your `Project.toml` file, and generate one.

```{note}
For a typical project intended to be for reproducible research, you should always keep the `Manifest.toml` in source control so that you can reproduce every stage of the project.  This can save you when you need to roll back to an older version to track down changes.  However, when creating reusable packages largely intended for others to have as dependencies, you will typically not include the `Manifest.toml`.
```

(test_driven)=
### Test-driven Development

The basic idea of test-driven development is to create a set of tests of individual functions in `test/runtests.jl` (and files it includes), while keeping the core functionality in  `src/MyProject.jl` (and files it includes).

This will be part of a coherent set of strategies to ensure everything is reproducible since:
- Writing all of the checks on your underlying functions to be called from  `test/runtests.jl` lets you can avoid accidentally breaking old functionality.  Breaking of old code is called a [test regression](https://en.wikipedia.org/wiki/Regression_testing) and is especially problematic with research code.
- The full snapshot of all of the packages associated with a particular version of the project (i.e. a commmit) in a `Manifest.toml`, anyone can reproduce the exact environment simply from that point in the source code tree.
- Any changes to the code automatically run in the CI (i.e. the GitHub Action) after every modification, you and any collaborators will be able to automatically track changes
- The code coverage will give you a sense of how much of the code you have actually executed in your tests, which can give you a sense of how much faith should be given to the automatic tracking of changes.
- The visual badges and displays on GitHub (e.g. the CI badge and the display of checks on each PR) provides an easy way to track when problems occur

(editing_package_code)=
### Writing Code in the Package

First, in the REPL add support for the `Distributions.jl` package and Julia unit testing

```{code-block} none
] add Distributions Test
```

Which should provide output such as

```{figure} /_static/figures/vscode_add_package_2.png
:width: 100%
```


While it will add the package to the `Project.toml`, that unlike the previous package operation, this will not install any new packages.  The reason is that `Distributions.jl` was already in the manifest as a dependency of `Expectations.jl` - and hence the network of packages already supports it.


Next, modify the `src/MyProject.jl` to include

```{code-block} none
module MyProject

using Expectations, Distributions

function foo(μ = 1., σ = 2.)
    d = Normal(μ, σ)
    E = expectation(d)
    return E(x -> sin(x))
end

export foo

end # module
```

```{note}
This defines a function, `foo` in the package, and exports it so that it is available to be called directly with `foo()` after `using MyProject`.  Alternatively, you can leave off the `export foo` and instead call the function with the package name as a prefix `MyProject.foo()`.
```

Then, we can call this function within a REPL by first including our package

```{code-block} none
using MyProject
```

Then calling `foo()` with the default arguments in the REPL.  This should lead to something like

```{figure} /_static/figures/vscode_run_package_1.png
:width: 100%
```

Next, we will change the function in the package and call it again in the REPL:
* Modify the `foo` function definition to add `println("Modified foo definition")` inside the function
```{code-block} none
function foo(μ = 1., σ = 2.)
    println("Modified foo definition")
    d = Normal(μ, σ)
    E = expectation(d)
    return E(x -> sin(x))
end
```

And then call the `foo()` function again in the REPL, which shows that it is now using the modified function.

```{figure} /_static/figures/vscode_run_package_2.png
:width: 100%
```


```{note}
This behavior, where modifying code within the package is detected and immediately compiled, relies on the [installation of Revise.jl](testing_pkg_installation).  An alternative workflow without `Revise.jl` is to make modifications to the underlying package files, and then simply use `<Shift+Enter>` in the editor to recompile them.  Both approaches are fully defensible and tastes are idiosyncratic.

Regardless, if you modify functions **outside of** the package (e.g. a function in the `runtests.jl`) you will need to use the `<Shift+Enter>` to compile it, as Revise only tracks code within loaded packages.
```

Consider that when writing this function, we may be checking its behavior in the REPL.  For example, we know from theory that the symmetry of the `sin` function means the expectation of the `sin` of a mean-zero normal (i.e. `foo(0)`) should be close to zero.

To check this, we can use the `Test` package built-into julia, and added to our project file above.  The `@test` macro within it verifies a condition, and fails if it is incorrect.

In the REPL, use the `Test` package and then check that `foo(0)` is close to 0.
```{code-block} none
using Test, MyProject
@test foo(0) < 1E-16
```

Which should show that the "Test Passed".
```{figure} /_static/figures/vscode_run_package_3.png
:width: 100%
```

(editing_test_code)=
### Writing Test Code

The core of test driven development is that if a value was worth checking once, then it is probalby worth checking every time the code changes.

For this reason, the change to this workflow means moving much of the code you would use to explore in a Jupyter notebook into the `runtests.jl` (or files they `include`) themselves.

The `@testset` are just optional ways to group the tests, and are not required for organization.

- In `test/runtests.jl`, move the `@test foo(0) < 1E-16`  inside of the `@testset`, which you can rename.  The testsets are optional, but can provide easy ways to group tests and execute many at once within VS Code.
- Then `<Shift-Enter>` through each line of the `runtests.jl` file.  As you execute each line of code, it will provide the result or a checkmark to indicate success

At the end, you should have output within the REPL that the test you added passed.
```{figure} /_static/figures/vscode_run_package_4.png
:width: 100%
```

```{note}
Note that for the `runtests.jl` file, we need to manually execute lines of code as we wish to explore the results, in contrast to the changes in the package where Revise automatically compiled them.  This includes function definitions inside of these files.
```

Checks on individual functions in your test suite are called [unit tests](https://en.wikipedia.org/wiki/Unit_testing).

(executing_tests)=
### Executing Tests Locally and on CI

As you develop your project, you may find yourself adding in many small checks and organizing them into files included from runtests.  After modifying a large amount of code, you may want to run through the entire set of unit tests to ensure you didn't break anything.

To do this, the package mode in Julia has a convenient function which will create a separate environment using your Project/Manifest and execute the entire `runtests.jl` function.

If you have already activated the project, then `] test` is sufficient.  Otherwise, if you added it to the [global environment](add_to_global), you could type `] test MyProject`.

Run this test to see the output.
```{figure} /_static/figures/vscode_run_package_5.png
:width: 100%
```

This shows that the individual test passed, but also that the entire test suite was without errors.

Now that we have a functional test suite, it is a good time to upload our code to the server.  Create a commit, and push all changes to the server.

If you reload the webpage, you will notice that the commit is now on the server, but also that a small orange circle is next to the commit.  

```{figure} /_static/figures/ci_1.png
:width: 100%
```

This indicates that a GitHub Action is executing based on that push to the main branch.

You can see these actions in progress on the Actions tab, where the action may be complete by the time you open it.

```{figure} /_static/figures/ci_2.png
:width: 100%
```

If you further click on one of the jobs and select the `julia-runtest` you can see the output on the server.  It ran your test suite on its isolated environment and is showing a check mark because the tests passed.

```{figure} /_static/figures/ci_3.png
:width: 100%
```

```{figure} /_static/figures/ci_4.png
:width: 100%
```

By this point, if you navigate back to the web page you will see that the commit has a checkmark (rather than an orange circle) and the badge at the bottom says "CI | passing" to indicate that the last action was successful.

(branches_tests)=
### "Feature" Branches and Continuous Integration

The previous example showed how we the CI will automatically execution an action when you push it to the main branch.

These actions will also apply, separately, to any branches and Pull Requests (PRs) that you create, as discussed in the {doc}`previous lecture <../software_engineering/version_control>`.

Separate testing and CI is one of the primary motivations for creating a branch when collaborating on a project.

To demonstrate this

* Create a new branch on VS Code called `my-feature` (i.e., click on the `main` branch at the bottom of the screen to enter the branch selection, then choose to make a new one).

* In `MyProject.jl`, change the function in `foo` from `sin(x)` to `cos(x)`.  Note that this will change the result of `foo(0)`.

```{figure} /_static/figures/ci_5.png
:width: 100%
```
* Commit this change to your local branch with the commit message.
* Then, imagine that rather than running `]test` - which would have shown this error given our previous unit test, you pushed it to the server.  As always, do this by selecting the publish arrow next to `my-feature` at the bottom of the screen.  You can create a Pull Request in that GUI, or just go back to the webpage where it asked you if you wish to "Compare and Pull Request"


After you create the pull request, you will see it in available on the web.  After a few minutes, the unit test will execute and you will see an output such as 

```{figure} /_static/figures/ci_6.png
:width: 100%
```

Unlike in the previous example, the pull request will have a red X indicating that one of the actions failed.

If this were done by a collaborator, you could go to the "Files Changed" tab, see the code modifications for commits in this PR, and then add a comment on these inline for discussion (using the `@` to tag an individual so they are emailed)

```{figure} /_static/figures/ci_7.png
:width: 100%
```

At that point, you or your collaborators can easily switch to the branch associated with the PR (i.e. `my-feature` in the above screenshot), make changes, and test them.


If you go back to VS Code and change the `cos` back to `sin` and commit with a message (e.g. `Fixed bug`) then the PR will rerun the CI.

In this case, you will see that while the `Modified foo` broken the CI the `Fixed bug` commit passed, and has a green checkmark after it completes its run.
```{figure} /_static/figures/ci_8.png
:width: 100%
```

Furthermore, since the previous comment was made on a line of code that was changed after the comment was made, it says the comment is "Outdated".  When collaborating, the "Resolve conversation" option would let you close these sorts of issues as they are found.

### Jupyter Workflow

We can also work with the package, and activated project file, from a Jupyter notebook for analysis and exploration.

In general, Jupyter should be used sparingly as it does not support a tight workflow for collaboration since it does not have an equivalent for the CI, and changes to the Jupyter notebook cannot be discussed inline and analyzes using the workflow above.

However, if a package is fairly stable and you are working on analysis, it is a very useful tool.

Assuming that we have the `IJulia` and conda installed, as in the basic lecture note setup, if we start a terminal in VS Code we will be able to start Jupyter in the right directory.



Let's create a new output directory in our project, and run `jupyter lab` from it.

Create a new notebook `output.ipynb`

```{figure} /_static/figures/testing-output.png
:width: 100%
```

From here, we can use our package's functions as we would functions from other packages.

This lets us produce neat output documents, without pasting the whole codebase.

We can also run package operations inside the notebook

```{figure} /_static/figures/testing-notebook.png
:width: 100%
```

The change will be reflected in the `Project.toml` file.

Note that, as usual, we had to first activate `ExamplePackage` first before making our dependency changes

```{code-block} julia
name = "ExamplePackage"
uuid = "f85830d0-e1f0-11e8-2fad-8762162ab251"
authors = ["QuantEcon User <quanteconuser@gmail.com>"]
version = "0.1.0"

[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Expectations = "2fe49d83-0758-5602-8f54-1f90ad0d522b"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

There will also be changes in the Manifest as well .

Be sure to add `output/.ipynb_checkpoints` to your `.gitignore` file, so that's not checked in.

Make sure you've activated the project environment (`] activate ExamplePackage`) before you try to propagate changes.

### Collaborative Work

For someone else to get the package, they simply need to

* Run the following command

```{code-block} julia
] dev https://github.com/quanteconuser/ExamplePackage.jl.git
```

This will place the repository inside their `~/.julia/dev` folder.

* Drag-and-drop the folder to GitHub desktop in the usual way.

Recall that the path to your `~/.julia` folder is

```{code-cell} julia
DEPOT_PATH[1]
```

They can then collaborate as they would on other git repositories.

In particular, they can run

```{code-block} julia
] activate ExamplePackage
```

to load the list of dependencies, and

```{code-block} julia
] instantiate
```

to make sure they are installed on the local machine.

## Unit Testing

It's important to make sure that your code is well-tested.

There are a few different kinds of test, each with different purposes

* *Unit testing* makes sure that individual pieces of a project work as expected.
* *Integration testing* makes sure that they fit together as expected.
* *Regression testing* makes sure that behavior is unchanged over time.

In this lecture, we'll focus on unit testing.

In general, well-written unit tests (which also guard against regression, for example by comparing function output to hardcoded values) are sufficient for most small projects.

### The `Test` Module

Julia provides testing features through a built-in package called `Test`, which we get by `using Test`.

The basic object is the macro `@test`

```{code-cell} julia
using Test
@test 1 == 1
@test 1 ≈ 1
```

Tests will pass if the condition is `true`, or fail otherwise.

If a test is failing, we should flag it with `@test_broken` as below

```{code-cell} julia
@test_broken 1 == 2
```

This way, we still have access to information about the test, instead of just deleting it or commenting it out.

There are other test macros, that check for things like error handling and type-stability.

Advanced users can check the [Julia docs](https://docs.julialang.org/en/v1/stdlib/Test/).

### Example

Let's add some unit tests for the `foo()` function we defined earlier.

Our `tests/runtests.jl` file should look like this.

**As before, this should be pasted into the file directly**

```{code-block} julia
using ExamplePackage
using Test

@test foo() ≈ 0.11388071406436832
@test foo(1, 1.5) ≈ 0.2731856314283442
@test_broken foo(1, 0) # tells us this is broken
```

And run it by typing `] test` into an activated REPL (i.e., a REPL where you've run `] activate ExamplePackage`).

### Test Sets

By default, the `runtests.jl` folder starts off with a `@testset`.

This is useful for organizing different batches of tests, but for now we can simply ignore it.

To learn more about test sets, see [the docs](https://docs.julialang.org/en/v1/stdlib/Test/index.html#Working-with-Test-Sets-1/).

### Running Tests

There are a few different ways to run the tests for your package.

* Run the actual `runtests.jl`, say by hitting `shift-enter` on it in Atom.
* From a fresh (`v1.1`) REPL, run `] test ExamplePackage`.
* From an activated (`ExamplePackage`) REPL, simply run `] test` (recall that you can activate with `] activate ExamplePackage`).

## Continuous Integration with Travis

### Setup

By default, Travis should have access to all your repositories and deploy automatically.

This includes private repos if you're on a student developer pack or an academic plan (Travis detects this automatically).

To change this, go to "settings" under your GitHub profile

```{figure} /_static/figures/git-settings.png
:width: 50%
```

Click "Applications," then "Travis CI," then "Configure," and choose the repos you want to be tracked.

### Build Options

By default, Travis will compile and test your project (i.e., "build" it) for new commits and PRs for every tracked repo with a `.travis.yml` file.

We can see ours by opening it in Atom

```{code-block} none
# Documentation: http://docs.travis-ci.com/user/languages/julia/
language: julia
os:
- linux
- osx
julia:
- 1.1
- nightly
matrix:
allow_failures:
    - julia: nightly
fast_finish: true
notifications:
email: false
after_success:
- julia -e 'using Pkg; Pkg.add("Coverage"); using Coverage; Codecov.submit(process_folder())'
```

This is telling Travis to build the project in Julia, on OSX and Linux, using Julia v1.1 and the latest development build ("nightly").

It also says that if the nightly version doesn't work, that shouldn't register as a failure.

**Note** You won't need OSX unless you're building something Mac-specific, like iOS or Swift.

You can delete those lines to speed up the build, likewise for the nightly Julia version.

### Working with Builds

As above, builds are triggered whenever we push changes or open a pull request.

For example, if we push our changes to the server and then click the Travis badge (the one which says "build") on the README, we should see something like

```{figure} /_static/figures/travis-progress.png
:width: 100%
```

Note that you may need to wait a bit and/or refresh your browser.

This gives us an overview of all the builds running for that commit.

To inspect a build more closely (say, if it fails), we can click on it and expand the log options

```{figure} /_static/figures/travis-log.png
:width: 100%
```

Note that the build times here aren't informative, because we can't generally control the hardware to which our job is allocated.

We can also cancel specific jobs, either from their specific pages or by clicking the grey "x" button on the dashboard.

Lastly, we can trigger builds manually (without a new commit or PR) from the Travis overview

```{figure} /_static/figures/travis-trigger.png
:width: 100%
```

To commit *without* triggering a build, simply add "[ci skip]" somewhere inside the commit message.

### Travis and Pull Requests

One key feature of Travis is the ability to see at-a-glance whether PRs pass tests before merging them.

This happens automatically when Travis is enabled on a repository.

For an example of this feature, see [this PR](https://github.com/QuantEcon/Games.jl/pull/65/) in the `Games.jl` repository.

## Code Coverage

Beyond the success or failure of our test suite, we also want to know how much of our code the tests cover.

The tool we use to do this is called [Codecov](http://codecov.io).

### Setup

You'll find that Codecov is automatically enabled for public repos with Travis.

For private ones, you'll need to first get an access token.

Add private scope in the Codecov website, just like we did for Travis.

Navigate to the repo settings page (i.e., `https://codecov.io/gh/quanteconuser/ExamplePackage.jl/settings` for our repo) and copy the token.

Next, go to your Travis settings and add an environment variable as below

```{figure} /_static/figures/travis-settings.png
:width: 100%
```

### Interpreting Results

Click the Codecov badge to see the build page for your project.

This shows us that our tests cover 50% of our functions in `src//`.

**Note:** To get a more detailed view, we can click the `src//` and the resultant filename.

**Note:** Codecov may take a few minutes to run for the first time

```{figure} /_static/figures/codecov.png
:width: 100%
```

This shows us precisely which methods (and parts of methods) are untested.

## Pull Requests to External Julia Projects

As mentioned in {doc}`version control <../software_engineering/version_control>`, sometimes we'll want to work on external repos that are also Julia projects.

* `] dev` the git URL (or package name, if the project is a registered Julia package), which will both clone the git repo to `~/.julia/dev` and sync it with the Julia package manager.

For example, running

```{code-block} julia
] dev Expectations
```

will clone the repo `https://github.com/quantecon/Expectations.jl` to `~/.julia/dev/Expectations`.

Make sure you do this from the base Julia environment (i.e., after running `] activate` without arguments).

As a reminder, you can find the location of your `~/.julia` folder (called the "user depot"), by running

```{code-cell} julia
DEPOT_PATH[1]
```

The `] dev` command will also add the target to the package manager, so that whenever we run `using Expectations`, Julia will load our cloned copy from that location

```{code-block} julia
using Expectations
pathof(Expectations) # points to our git clone
```

```{figure} /_static/figures/testing-fork.png
:width: 100%
```

```{figure} /_static/figures/testing-repo-settings.png
:width: 75%
```

* Drag that folder to GitHub Desktop.
* The next step is to fork the original (external) package from its website (i.e., `https://github.com/quantecon/Expectations.jl`) to your account (`https://github.com/quanteconuser/Expectations.jl` in our case).
* Edit the settings in GitHub Desktop (from the "Repository" dropdown) to reflect the new URL.
  
  Here, we'd change the highlighted text to read `quanteconuser`, or whatever our GitHub ID is.
* If you make some changes in a text editor and return to GitHub Desktop, you'll see something like.

**Note:** As before, we're editing the files directly in `~/.julia/dev`, as opposed to cloning the repo again.

```{figure} /_static/figures/testing-commit.png
:width: 100%
```

Here, for example, we're revising the README.

* Clicking "commit to master" (recall that the checkboxes next to each file indicate whether it's to be committed) and then pushing (e.g., hitting "push" under the "Repository" dropdown) will add the committed changes to your account.

To confirm this, we can check the history on our account [here](https://github.com/quanteconuser/Expectations.jl/commits/master); for more on working with git repositories, see the {doc}`version control <../software_engineering/version_control>` lecture.

```{figure} /_static/figures/testing-expectations.png
:width: 100%
```

The green check mark indicates that Travis tests passed for this commit.

* Clicking "new pull request" from the pull requests tab will show us a snapshot of the changes, and let us create a pull request for project maintainers to review and approve.

```{figure} /_static/figures/testing-pr2.png
:width: 100%
```

For more on PRs, see the relevant section of the {doc}`version control <../software_engineering/version_control>` lecture.

For more on forking, see the docs on [GitHub Desktop](https://help.github.com/desktop/guides/contributing-to-projects/cloning-a-repository-from-github-to-github-desktop/) and [the GitHub Website](https://guides.github.com/activities/forking/).

### Case with Write Access

If you have write access to the repo, we can skip the preceding steps about forking and changing the URL.

You can use `] dev` on a package name or the URL of the package

```{code-block} julia
] dev Expectations
```

or `] dev https://github.com/quanteconuser/Expectations.jl.git` as an example for an unreleased package by URL.

Which will again clone the repo to `~/.julia/dev`, and use it as a Julia package.

```{code-block} julia
using Expectations
pathof(Expectations) # points to our git clone
```

Next, drag that folder to GitHub Desktop as before.

Then, in order to work with the package locally, all we need to do is open the `~/.julia/dev/Expectations` in a text editor (like Atom)

```{figure} /_static/figures/testing-atom-package.png
:width: 100%
```

From here, we can edit this package just like we created it ourselves and use GitHub Desktop to track versions of our package files (say, after `] up`, or editing source code, `] add Package`, etc.).

### Removing a Julia Package

To "un-dev" a Julia package (say, if we want to use our old `Expectations.jl`), you can simply run

```{code-block} julia
] free Expectations
```

To delete it entirely, simply run

```{code-block} julia
] rm Expectations
```

From a REPL where that package is in the active environment.

## Benchmarking

Another goal of testing is to make sure that code doesn't slow down significantly from one version to the next.

We can do this using tools provided by the `BenchmarkTools.jl` package.

See the {doc}`need for speed <../software_engineering/need_for_speed>` lecture for more details.

## Additional Notes

* The [JuliaCI](https://github.com/JuliaCI/) organization provides more Julia utilities for continuous integration and testing.

## Review

To review the workflow for creating, versioning, and testing a new project end-to-end.

1. Create the local package directory using the `PkgTemplates.jl`.
1. Add that package to the Julia package manager, by opening a Julia REPL in the `~/.julia/dev/ExamplePackage.jl`, making sure the active environment is the default one `(v1.1)`, and hitting `] dev .`.
1. Drag-and-drop that folder to GitHub Desktop.
1. Create an empty repository with the same name on the GitHub server.
1. Push from GitHub Desktop to the server.
1. Open **the original project folder** (e.g., `~/.julia/dev/ExamplePackage.jl`) in Atom.
1. Make changes, test, iterate on it, etc. As a rule, functions like should live in the `src/` directory once they're stable, and you should export them from that file with `export func1, func2`. This will export all methods of `func1`, `func2`, etc.
1. Commit them in GitHub Desktop as you go (i.e., you can and should use version control to track intermediate states).
1. Push to the server, and see the Travis and Codecov results (note that these may take a few minutes the first time).

## Exercises

### Exercise 1

Following the {ref}`instructions for a new project <project_setup>`, create a new package on your github account called `NewtonsMethod.jl`.

In this package, you should create a simple package to do Newton's Method using the code you did in the
{ref}`Newton's method <jbe_ex8a>` exercise in {doc}`Introductory Examples <../getting_started_julia/julia_by_example>`.

In particular, within your package you should have two functions

* `newtonroot(f, f′; x₀, tol = 1E-7, maxiter = 1000)`
* `newtonroot(f; x₀, tol = 1E-7, maxiter = 1000)`

Where the second function uses Automatic Differentiation to call the first.

The package should include

* implementations of those functions in the `/src` directory
* comprehensive set of tests
* project and manifest files to replicate your development environment
* automated running of the tests with Travis CI in GitHub

For the tests, you should have at the very minimum

* a way to handle non-convergence (e.g. return back `nothing` as discussed in {ref}`error handling <error_handling>`
* several `@test` for the root of a known function, given the `f` and analytical `f'` derivatives
* tests of those roots using the automatic differentiation version of the function
* test of finding those roots with a `BigFloat` and not just a `Float64`
* test of non-convergence for a function without a root (e.g. $f(x) = 2 + x^2$ )
* test to ensure that the `maxiter` is working (e.g. what happens if you call `maxiter = 5`
* test to ensure that `tol` is working

And anything else you can think of.  You should be able to run `] test` for the project to check that the test-suite is running, and then ensure that it is running automatically on Travis CI.

Push a commit to the repository which breaks one of the tests and see what the Travis CI reports after running the build.

### Exercise 2

Watch the youtube video [Developing Julia Packages](https://www.youtube.com/watch?v=QVmU29rCjaA) from Chris Rackauckas.  The demonstration goes through many of the same concepts as this lecture, but with more background in test-driven development and providing more details for open-source projects..

