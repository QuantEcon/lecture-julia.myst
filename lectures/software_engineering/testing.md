---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.8
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

Benefits include:

* Specifying dependencies (and their versions) so that your project works across Julia setups and over time.
* Being able to load your project's functions from outside without copy/pasting.
* Writing tests that run locally, *and automatically on the GitHub server*.
* Having GitHub test your project across operating systems, Julia versions, etc.

**Note:** Throughout this lecture, important points and sequential workflow steps are listed as bullets.

The goal of this style of coding is to ensure that [test driven development](test_driven) is possible, which will help ensure that your research is reproducible by yourself, your future self, and others - while avoiding accidental mistakes that can silently break old code and change results.

Furthermore, it makes collaboration on code much more feasible - allowing individuals to work on different parts of the code without fear of breaking others' work.

Finally, since many economics projects occur over multiple years - this will help your future self to reproduce your results and make changes without remembering all of the subtle tricks required to get things operational.


## Introduction and Setup

Much of the software engineering and continuous integration (CI) will be done through [GitHub Actions](https://github.com/features/actions).

The GitHub actions execute as isolated and fully reproducible environments on the cloud (using containers), and are initiated through various actions on the GitHub repo.  In particular, these are typically initiated through:
- Making commits on the main branch of a repository.
- Creating a PR on a repository, or pushing commits to it.
- [Tagging a release](https://docs.github.com/en/github/administering-a-repository/releasing-projects-on-github/managing-releases-in-a-repository) of a repository, which provides a snapshot at a particular commit.

For publicly available repositories, GitHub provides free minutes for executing these actions, whereas there are limits on the execution time for private repositories - though signing up for the GitHub academic plans previously discussed (i.e, the [Student Developer Pack](https://education.github.com/pack/) or [Academic/Researcher plans](https://help.github.com/articles/about-github-education-for-educators-and-researchers/)) provides additional free minutes.

While you may think of a ''Package'' as a shared and maintained set of reusable code, with Julia it will turn out to be the most useful way to organize personal and private projects or the code associated with a particular paper.  The primary benefit of using a package workflow is **reproducibility** for yourself, your future self, and collaborators.

In addition, you will find the testing workflow as an essential element of working on even individual projects, since it will prevent you from making accidental breaking changes at points in the future - or when you come back to the code after several years.

(testing_account_setup)=
### GitHub and Codecov Account Setup

This lecture assumes you have completed the setup of VS Code and Git in the  {doc}`version control <../software_engineering/version_control>` and  {doc}`tools <../software_engineering/tools_editors>`  lectures.  In particular,
- Ensure you have setup a GitHub account, and connected it to VS Code.
- Installed and setup VS Code's Julia extension.
- Set ` git config --global user.email "you@example.com"`, `git config --global user.name "Your Name"` and `git config --global github.user "YOURGITHUBNAME"` in your terminal.


The only other service that is necessary for the complete software engineering stack is a code coverage provider.

For these lectures, visit [Codecov website](https://about.codecov.io/sign-up/).

Installation instructions are [here](https://docs.codecov.com/docs/quick-start#getting-started).

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
```{code-block} julia
] add PkgTemplates Revise
```

While we typically insist on having very few packages in the global environment, and always working with a `Project.toml` file, `PkgTemplates` is primarily used without an existing project.

While the [Revise.jl](https://github.com/timholy/Revise.jl) package is optional, it is strongly encouraged since it automatically compiling functions in your package as they are changed.  See [below](editing_package_code) for more.


(project_setup)=
## Project Setup

To create a project, first choose the parent directory where you wish to create the project

* In an external terminal, navigate to this parent directory.
```{note}
On Windows, given that you have installed Git you could right click on the folder in explorer and choose `Git Bash Here`.  Even better, install the [Windows Terminal](https://docs.microsoft.com/en-us/windows/terminal/get-started) and choose `Open in Windows Terminal`.
```

* Start a julia terminal with `julia`

* Then run

```{code-block} julia
using PkgTemplates
```

* Create a *template* for your project.

This specifies metadata like the license we'll be using (MIT by default), the location, etc.


We will create this with a number of useful options, but see [the documentation](https://invenia.github.io/PkgTemplates.jl/stable/user/#A-More-Complicated-Example-1) for more.

```{code-block} julia
t = Template(;dir = ".", julia = v"1.8",
              plugins = [
                Git(; manifest=true, branch = "main"),
                Codecov(),
                GitHubActions(),
                !CompatHelper,
                !TagBot
              ])
```

```{note}
If you did not set the `github.user` in the setup, you may need to pass in `user = "YOURUSERNAME"` as an additional argument in the first line.  In addition, this turns off some important features (e.g. `CompatHelper` and `TagBot`) and leaves out others (e.g. `Documenter{GitHubActions}`) which you would want for a more formal package.

Alternatively, `PkgTemplates` has an interactive mode, which you can prompt with `t = Template(;interactive = true)` to choose all of your selections within the terminal.
```

* Create a specific project based off this template

```{code-block} julia
t("MyProject")
```

* Open the project in VS Code by right-clicking in the explorer for the new `MyProject` folder, or exiting the Julia REPL (via `exit()`) and then
```{code-block} bash
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

Furthermore, you will see a badge either saying `CI | passing` or `CI | unknown` next to another for `codecov | unknown`.

These badges provide a summary of the current state of the `main` branch relative to the GitHub Actions, as we will see below.  If a repository has broken any tests on the main branch, it will show `CI | failed` in red.

(add_to_global)=
### (Optionally) Adding the Package to the Global Environment

Optionally, you may wish to be able to use your package within other projects on your computer.

To do this, you can add it to the main environment by starting Julia in the `MyProject` folder, and without activating the project (i.e. just `julia`).

Then

```{code-block} julia
] dev .
```

Given this, other julia code can use `using MyProject` and, because the global environment stacks on any project files, it will be available.

You can see the change reflected in our default package list by running `] st`

```{code-block} bash
      Status `C:\Users\jesse\.julia\environments\v1.8\Project.toml`
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
  
  ```{code-block} julia
  module MyProject
  
  greet() = print("Hello World!")
  
  end # module
  ```
  
* Likewise, the `test` directory should have only one file (`runtests.jl`), which reads
  
  ```{code-block} julia
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
```{code-block} yaml
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
          - '1.8'
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

As {ref}`before <jl_packages>`, the `.toml` files define an *environment* for our project, or a set of files which represent the dependency information.

The actual files are written in the [TOML language](https://github.com/toml-lang/toml), which is a lightweight format to specify configuration options.

This information is the name of every package we depend on, along with the exact versions of those packages.

This information (in practice, the result of package operations we execute) will
be reflected in our `MyProject.jl` directory's TOML, once that environment is activated (selected).

This allows us to share the project with others, who can exactly reproduce the state used to build and test it.

See the [Pkg docs](https://docs.julialang.org/en/v1/stdlib/Pkg/) for more information.

#### Pkg Operations

For now, let's just try adding a dependency.  Recall the package operations described in the {doc}`tools and editors <../software_engineering/tools_editors>` lecture.

* Within VS Code, start a REPL (e.g. `> Julia: Start REPL`), type `]` to enter the Pkg mode.  Verify that the cursor says `(My Project) pkg>` to ensure that it has activated your particular project.  Otherwise, if it says `(@1.8) pkg>`, then you have launched Julia outside of VSCode or through a different mechanism, and you might need to ensure you are in the correct directory and then `] activate .` to activate the project file in it.
* Then, add in the `Expectations.jl` package

```{figure} /_static/figures/vscode_add_package.png
:width: 100%
```

After installing a large web of dependencies, this tells Julia to write the results of package operations to the activated `Project.toml` file, which should now include text such as
```toml
[deps]
Expectations = "2fe49d83-0758-5602-8f54-1f90ad0d522b"
```

From this point, anytime the project is activated, the [Expectations.jl](https://github.com/QuantEcon/Expectations.jl) package will be available with `using Expectations`.

Furthermore, different projects could use different versions of this package.   We could add additional instructions on [compatability](https://pkgdocs.julialang.org/dev/compatibility/) within the file if we choose.

But almost more importantly, the `Manifest.toml` file now contains a complete snapshot of the particular `Expectations` package and every other dependency used in your project.

For example, the following is an example snippet of one manifest here:
```toml
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
- With the full snapshot of all of the packages associated with a particular version of the project (i.e. a commmit) in a `Manifest.toml`, anyone can reproduce the exact environment simply from that point in the source code tree.
- Any changes to the code automatically run in the CI (i.e. the GitHub Action) after every modification, so you and any collaborators will be able to automatically track changes.
- The code coverage will give you a sense of how much of the code you have actually executed in your tests, which can give you a sense of how much faith should be given to the automatic tracking of changes.
- The visual badges and displays on GitHub (e.g. the CI badge and the display of checks on each PR) provides an easy way to track when problems occur.

(editing_package_code)=
### Writing Code in the Package

First, in the REPL add support for the `Distributions.jl` package and Julia unit testing

```{code-block} julia
] add Distributions Test
```

Which should provide output such as

```{figure} /_static/figures/vscode_add_package_2.png
:width: 100%
```


While it will add the package to the `Project.toml`, unlike the previous package operation, this will not install any new packages.  The reason is that `Distributions.jl` was already in the manifest as a dependency of `Expectations.jl` - and hence the network of packages already supports it.


Next, modify the `src/MyProject.jl` to include

```{code-block} julia
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

```{code-block} julia
using MyProject
```

Then calling `foo()` with the default arguments in the REPL.  This should lead to something like

```{figure} /_static/figures/vscode_run_package_1.png
:width: 100%
```

Next, we will change the function in the package and call it again in the REPL:
* Modify the `foo` function definition to add `println("Modified foo definition")` inside the function
```{code-block} julia
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
```{code-block} julia
using Test, MyProject
@test foo(0) < 1E-16
```

Which should show that the "Test Passed".
```{figure} /_static/figures/vscode_run_package_3.png
:width: 100%
```

(editing_test_code)=
### Writing Test Code

The core of test driven development is that if a value was worth checking once, then it is probably worth checking every time the code changes.

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

The previous example showed how the CI will automatically execution an action when you push it to the main branch.

These actions will also apply, separately, to any branches and Pull Requests (PRs) that you create, as discussed in the {doc}`previous lecture <../software_engineering/version_control>`.

Separate testing and CI is one of the primary motivations for creating a branch when collaborating on a project.

To demonstrate this

* Create a new branch on VS Code called `my-feature` (i.e., click on the `main` branch at the bottom of the screen to enter the branch selection, then choose to make a new one).

* In `MyProject.jl`, change the function in `foo` from `sin(x)` to `cos(x)`.  Note that this will change the result of `foo(0)`.

```{figure} /_static/figures/ci_5.png
:width: 100%
```
* Commit this change to your local branch with the commit message.
* Then, imagine that rather than running `] test` - which would have shown this error given our previous unit test, you pushed it to the server.  As always, do this by selecting the publish arrow next to `my-feature` at the bottom of the screen.  You can create a Pull Request in that GUI, or just go back to the webpage where it asked you if you wish to "Compare and Pull Request".


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

In this case, you will see that while the `Modified foo` broke the CI, the `Fixed bug` commit passed, and has a green checkmark after it completes its run.
```{figure} /_static/figures/ci_8.png
:width: 100%
```

Furthermore, since the previous comment was made on a line of code that was changed after the comment was made, it says the comment is "Outdated".  When collaborating, the "Resolve conversation" option would let you close these sorts of issues as they are found.

Reselect your `main` branch in VS Code to make further changes there rather than on the `my-feature` branch.


### Code Coverage

At this point, the `Codecov` badge in the README likely says `codecov | unknown` since the repository has not been added there.

Click on that badge, which should take you to the `Codecov` website where you can login if required, and select particular commits or PRs.

For example, on our previous PR we can see that it detected the change from the `cos` back to `sin` and that the coverage is 100% (i.e., every line of code in the package is executed by a test)

```{figure} /_static/figures/codecov_1.png
:width: 100%
```


To see the impact of additional code in the package, go back to your `main` branch, add in a new function such as 
```{code-block} julia
function bar()
    return 10
end
```
and then commit and push to the server.

After the CI executes, you will notice that the unit tests still pass despite the additional function.

However, if you click on the code coverage badge again and look at the main branch (or that particular commit) you will see that the code coverage has decreased substantially.

```{figure} /_static/figures/codecov_2.png
:width: 100%
```

Furthermore, it highlights in red the new code in this commit which led to the lowered code coverage.  In this case, the `bar` function had nothing in `runtests.jl` that was calling it.  This can help you ensure that all of the code paths (not just entire functions) are called in one test or another in your unit tests.

### Jupyter Workflow

We can also work with the package, and activated project file, from a Jupyter notebook for analysis and exploration.

In general, Jupyter should be used sparingly as it does not support a tight workflow for collaboration since it does not have an equivalent for the CI, and changes to the Jupyter notebook cannot be discussed inline and analyzed using the workflow above.

However, if source code is fairly stable and you are working on just the analysis and visualization of results (where introducing bugs is unlikely), it is a very useful tool.

Assuming that we have the `IJulia` package and conda installed, as in the basic lecture note setup, if we start a terminal in VS Code we will be able to start Jupyter in the right directory.

- Create a new terminal for your operating system by choosing the `+` button on the terminals pane, then type `jupyter lab` to start Jupyter, then open the webpage

```{figure} /_static/figures/vscode_jupyter_1.png
:width: 100%
```

- Create a new notebook in Jupyter in the main folder (or a subfolder) and add code to use `MyProject` and call `foo`.

```{figure} /_static/figures/vscode_jupyter_2.png
:width: 100%
```

Crucially, note that the `MyProject` package and anything else in the `Project.toml` file is available since Jupyter automatically activates the project file local to jupyter notebooks.

Be sure to add `.ipynb_checkpoints/*` to your `.gitignore` file, so that's not checked in.


While you would typically modify the Jupyter notebooks after the core code is stable, with `Revise.jl` it is possible to have it automatically load changes to the package itself.

To do this
- Execute `using Revise` prior to the `using MyProject` (note that VS Code's REPL does the `using Revise` automatically, if it is available).
- Then run `foo(0)` to see the old code
- Then modify the text of the package itself, such as changing the string in the `foo` function
- Run `foo(0)` again to see the change.


```{figure} /_static/figures/vscode_jupyter_3.png
:width: 100%
```

## More on Writing Tests

As discussed, there are a few different kinds of test, each with different purposes

* *Unit testing* makes sure that individual pieces of a project work as expected.
* *Integration testing* makes sure that they fit together as expected.
* *Regression testing* makes sure that behavior is unchanged over time.

In general, well-written unit tests (which also guard against regression, for example by comparing function output to hardcoded values) are sufficient for most small projects.

### The `Test` Module

Julia provides testing features through a built-in package called `Test`, which we get by `using Test`.

The basic object is the macro `@test`

```{code-cell} julia
using Test
@test 1 == 1
```

Since floating points can often only be compared within machine precision, you can typically only compare these approximately with either `isapprox` or `≈` (which can auto-completed in VS Code with `\approx<TAB>`)
```{code-cell} julia
@test 1.0 ≈ 1.0
x = 2.0
@test isapprox(x, 2.0)
@test x ≈ 2.0
```

The default comparison for `isapprox` is that the relative tolerance (i.e. $\|x - y\| \leq rtol \max(\|x\|, \|y\|)$) depends on the particular type, but defaults to the sqrt rate of the machine epsilon.
For example,
```{code-cell} julia
@show √eps(Float64)
@show √eps(BigFloat);
```

For cases where you want to change the relative tolerance or add in an absolute tolerance (i.e. $\|x - y\| \leq atol$) use the appropriate keywords
```{code-cell} julia
x = 100.0 + 1E-6  # within the tolerance
@test x ≈ 100.0 rtol=1E-7  # note < the 1E-6 difference passes due to relative scaling
y = 1E-7
@test 0.0 ≈ y atol=1E-6  # absolute tolerance!
```

```{note}
The relative tolerance is always preferred for comparisons since it is dimensionless, whereas the absolute tolerance is related to the norm of the objects themselves.  As the documentation for `isapprox` says: `x - y ≈ 0, atol=1e-9`is an absurdly small tolerance if `x` is the radius of the Earthin meters, but an absurdly large tolerance if `x` is the radius of a Hydrogen atom.

Because of this, there is no default `atol` if the norms of the values are zero.  And `rtol` is undefined, so you will need to use judgement on the specific problem to form a threshold.
```

The `isapprox` comparison works for any datatypes that implement a `norm`, for example
```{code-cell} julia
x = [1.0, 2.0]
@test x ≈ [1.0, 2.0]
```

Tests will pass if the condition is `true`, or fail otherwise.  Failing tests will be highlighted in both the CI and output within the terminal,

```{code-cell} julia
---
tags: [raises-exception, remove-output]
---
@test 1 == 2
```

Finally, sometimes you will have broken tests which are known to fail, but you do not want to remove.

This is good practice for code in progress that you intend to fix, but do not want to forget about, as commented out tests are easily forgotten.

To do this, flag it with `@test_broken` as below

```{code-cell} julia
@test_broken 1 == 2
```

This way, we still have access to information about the test, instead of just deleting it or commenting it out, but the CI will not fail.

There are other test macros, that check for things like error handling and type-stability.  Advanced users can check the [Julia docs](https://docs.julialang.org/en/v1/stdlib/Test/).

### Test Sets

By default, the `runtests.jl` folder starts off with a `@testset`.

This is useful for organizing different batches of tests, and executing them all at once.

```{code-cell} julia
@testset "my tests" begin
  @test 1 == 1
  @test 2 == 2
  @test_broken 1 == 2
end;
```
By using `<Shift+Enter>` in VS Code or Jupyter on the testset, you will execute them all.  You may want to execute only parts of them during development by commenting out the `@testset` and `end` and execute sequentially until the suite passes.


To learn more about test sets, see [the docs](https://docs.julialang.org/en/v1/stdlib/Test/index.html#Working-with-Test-Sets-1/).

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
* automated running of the tests with GitHub Actions

For the tests, you should have at the very minimum

* a way to handle non-convergence (e.g. return back `nothing` as discussed in {ref}`error handling <error_handling>`)
* several `@test` for the root of a known function, given the `f` and analytical `f'` derivatives
* tests of those roots using the automatic differentiation version of the function
* a test of finding those roots with a `BigFloat` and not just a `Float64`
* a test of non-convergence for a function without a root (e.g. $f(x) = 2 + x^2$ )
* a test to ensure that the `maxiter` is working (e.g. what happens if you call `maxiter = 5`)
* a test to ensure that `tol` is working

And anything else you can think of.  You should be able to run `] test` for the project to check that the test-suite is running, and then ensure that it is running automatically on GitHub Actions CI.

Push a commit to the repository which breaks one of the tests and see what the GitHub Actions CI reports after running the build.

<!-- 
# TODO: I think this is out of date, but mabye there aer better ones now?

### Exercise 2

Watch the youtube video [Developing Julia Packages](https://www.youtube.com/watch?v=QVmU29rCjaA) from Chris Rackauckas.  The demonstration goes through many of the same concepts as this lecture, but with more background in test-driven development and providing more details for open-source projects.. 
-->

