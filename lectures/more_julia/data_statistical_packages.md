---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.5
---

(data_statistical_packages)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Data and Statistics Packages

```{contents} Contents
:depth: 2
```

## Overview

This lecture explores some of the key packages for working with data and doing statistics in Julia.

In particular, we will examine the `DataFrame` object in detail (i.e., construction, manipulation, querying, visualization, and nuances like missing data).

While Julia is not an ideal language for pure cookie-cutter statistical analysis, it has many useful packages to provide those tools as part of a more general solution.

This list is not exhaustive, and others can be found in organizations such as [JuliaStats](https://github.com/JuliaStats), [JuliaData](https://github.com/JuliaData/), and  [QueryVerse](https://github.com/queryverse).


```{code-cell} julia
---
tags: [hide-output]
---
using LinearAlgebra, Statistics
using DataFrames, RDatasets, DataFramesMeta, CategoricalArrays, Query, VegaLite
using GLM
```

## DataFrames

A useful package for working with data is [DataFrames.jl](https://github.com/JuliaStats/DataFrames.jl).

The most important data type provided is a `DataFrame`, a two dimensional array for storing heterogeneous data.

Although data can be heterogeneous within a `DataFrame`, the contents of the columns must be homogeneous
(of the same type).

This is analogous to a `data.frame` in R, a `DataFrame` in Pandas (Python) or, more loosely, a spreadsheet in Excel.

There are a few different ways to create a DataFrame.

### Constructing and Accessing a DataFrame

The first is to set up columns and construct a dataframe by assigning names

```{code-cell} julia
using DataFrames, RDatasets  # RDatasets provides good standard data examples from R

# note use of missing
commodities = ["crude", "gas", "gold", "silver"]
last_price = [4.2, 11.3, 12.1, missing]
df = DataFrame(commod = commodities, price = last_price)
```

Columns of the `DataFrame` can be accessed by name using `df.col`, as below

```{code-cell} julia
df.price
```

Note that the type of this array has values `Union{Missing, Float64}` since it was created with a `missing` value.

```{code-cell} julia
df.commod
```

The `DataFrames.jl` package provides a number of methods for acting on `DataFrame`'s, such as `describe`.

```{code-cell} julia
DataFrames.describe(df)
```

While often data will be generated all at once, or read from a file, you can add to a `DataFrame` by providing the key parameters.

```{code-cell} julia
nt = (commod = "nickel", price= 5.1)
push!(df, nt)
```

Named tuples can also be used to construct a `DataFrame`, and have it properly deduce all types.

```{code-cell} julia
nt = (t = 1, col1 = 3.0)
df2 = DataFrame([nt])
push!(df2, (t=2, col1 = 4.0))
```

In order to modify a column, access the mutating version by the symbol `df[!, :col]`.

```{code-cell} julia
df[!, :price]
```

Which allows modifications, like other mutating `!` functions in julia.

```{code-cell} julia
df[!, :price] *= 2.0  # double prices
```

As discussed in the next section, note that the {ref}`fundamental types <missing>`, is propagated, i.e. `missing * 2 === missing`.

### Working with Missing

As we discussed in {ref}`fundamental types <missing>`, the semantics of `missing` are that mathematical operations will not silently ignore it.

In order to allow `missing` in a column, you can create/load the `DataFrame`
from a source with `missing`'s, or call `allowmissing!` on a column.

```{code-cell} julia
allowmissing!(df2, :col1) # necessary to add in a for col1
push!(df2, (t=3, col1 = missing))
push!(df2, (t=4, col1 = 5.1))
```

We can see the propagation of `missing` to caller functions, as well as a way to efficiently calculate with non-missing data.

```{code-cell} julia
@show mean(df2.col1)
@show mean(skipmissing(df2.col1))
```

And to replace the `missing`

```{code-cell} julia
df2.col1  .= coalesce.(df2.col1, 0.0) # replace all missing with 0.0
```

### Manipulating and Transforming DataFrames

One way to do an additional calculation with a `DataFrame` is to tuse the `@transform` macro from `DataFramesMeta.jl`.

```{code-cell} julia
using DataFramesMeta
f(x) = x^2
df2 = @transform(df2, col2 = f.(:col1))
```

### Categorical Data

For data that is [categorical](https://juliadata.github.io/DataFrames.jl/stable/man/categorical/)

```{code-cell} julia
using CategoricalArrays
id = [1, 2, 3, 4]
y = ["old", "young", "young", "old"]
y = CategoricalArray(y)
df = DataFrame(id=id, y=y)
```

```{code-cell} julia
levels(df.y)
```

### Visualization, Querying, and Plots

The `DataFrame` (and similar types that fulfill a standard generic interface) can fit into a variety of packages.

One set of them is the [QueryVerse](https://github.com/queryverse).

**Note:** The QueryVerse, in the same spirit as R's tidyverse, makes heavy use of the pipeline syntax `|>`.

```{code-cell} julia
x = 3.0
f(x) = x^2
g(x) = log(x)

@show g(f(x))
@show x |> f |> g; # pipes nest function calls
```

To give an example directly from the source of the LINQ inspired [Query.jl](http://www.queryverse.org/Query.jl/stable/)

```{code-cell} julia
using Query

df = DataFrame(name=["John", "Sally", "Kirk"], age=[23., 42., 59.], children=[3,5,2])

x = @from i in df begin
    @where i.age>50
    @select {i.name, i.children}
    @collect DataFrame
end
```

While it is possible to just use the `Plots.jl` library, there may be better options for displaying tabular data -- such as [VegaLite.jl](https://github.com/queryverse/VegaLite.jl).

```{code-cell} julia
using RDatasets, VegaLite
iris = dataset("datasets", "iris")

iris |> @vlplot(
    :point,
    x=:PetalLength,
    y=:PetalWidth,
    color=:Species
)
```

## Statistics and Econometrics

While Julia is not intended as a replacement for R, Stata, and similar specialty languages, it has a growing number of packages aimed at statistics and econometrics.

Many of the packages live in the [JuliaStats organization](https://github.com/JuliaStats/).

A few to point out

* [StatsBase](https://github.com/JuliaStats/StatsBase.jl) has basic statistical functions such as geometric and harmonic means, auto-correlations, robust statistics, etc.
* [StatsFuns](https://github.com/JuliaStats/StatsFuns.jl) has a variety of mathematical functions and constants such as pdf and cdf of many distributions, softmax, etc.

### General Linear Models

To run linear regressions and similar statistics, use the [GLM](http://juliastats.github.io/GLM.jl/latest/) package.

```{code-cell} julia
using GLM

x = randn(100)
y = 0.9 .* x + 0.5 * rand(100)
df = DataFrame(x=x, y=y)
ols = lm(@formula(y ~ x), df) # R-style notation
```

To display the results in a useful tables for LaTeX and the REPL, use
[RegressionTables](https://github.com/jmboehm/RegressionTables.jl/) for output
similar to the Stata package esttab and the R package stargazer.

```{code-cell} julia
using RegressionTables
regtable(ols)
# regtable(ols,  renderSettings = latexOutput()) # for LaTex output
```

### Fixed Effects

While Julia may be overkill for estimating a simple linear regression,
fixed-effects estimation with dummies for multiple variables are much more computationally intensive.

For a 2-way fixed-effect, taking the example directly from the documentation using [cigarette consumption data](https://github.com/johnmyleswhite/RDatasets.jl/blob/master/doc/plm/rst/Cigar.rst)

```{code-cell} julia
using FixedEffectModels
cigar = dataset("plm", "Cigar")
cigar.StateCategorical =  categorical(cigar.State)
cigar.YearCategorical =  categorical(cigar.Year)
fixedeffectresults = reg(cigar, @formula(Sales ~ NDI + fe(StateCategorical) + fe(YearCategorical)),
                            weights = :Pop, Vcov.cluster(:State))
regtable(fixedeffectresults)
```

