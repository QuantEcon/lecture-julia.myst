---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.10
---

(types_methods)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# The Need for Speed

```{contents} Contents
:depth: 2
```

## Overview

Computer scientists often classify programming languages according to the following two categories.

*High level languages* aim to maximize productivity by

* being easy to read, write and debug
* automating standard tasks (e.g., memory management)
* being interactive, etc.

*Low level languages* aim for speed and control, which they achieve by

* being closer to the metal (direct access to CPU, memory, etc.)
* requiring a relatively large amount of information from the user (e.g., all data types must be specified)

Traditionally we understand this as a trade off

* high productivity or high performance
* optimized for humans or optimized for machines

One of the great strengths of Julia is that it pushes out the curve, achieving
both high productivity and high performance with relatively little fuss.

The word "relatively" is important here, however...

In simple programs, excellent performance is often trivial to achieve.

For longer, more sophisticated programs, you need to be aware of potential stumbling blocks.

This lecture covers the key points.

### Requirements

You should read our {doc}`earlier lecture <../more_julia/generic_programming>` on types, methods and multiple dispatch before this one.


```{code-cell} julia
using LinearAlgebra, Statistics
```

## Understanding Multiple Dispatch in Julia

This section provides more background on how methods, functions, and types are connected.

### Methods and Functions

The precise data type is important, for reasons of both efficiency and mathematical correctness.

For example consider 1 + 1 vs. 1.0 + 1.0 or [1 0] + [0 1].

On a CPU, integer and floating point addition are different things, using a different set of instructions.

Julia handles this problem by storing multiple, specialized versions of functions like addition, one for each data type or set of data types.

These individual specialized versions are called **methods**.

When an operation like addition is requested, the Julia compiler inspects the type of data to be acted on and hands it out to the appropriate method.

This process is called **multiple dispatch**.

Like all "infix" operators, 1 + 1 has the alternative syntax +(1, 1)

```{code-cell} julia
+(1, 1)
```

This operator + is itself a function with multiple methods.

We can investigate them using the @which macro, which shows the method to which a given call is dispatched

```{code-cell} julia
x, y = 1.0, 1.0
@which +(x, y)
```

We see that the operation is sent to the `+` method that specializes in adding
floating point numbers.

Here's the integer case

```{code-cell} julia
x, y = 1, 1
@which +(x, y)
```

This output says that the call has been dispatched to the + method
responsible for handling integer values.

(We'll learn more about the details of this syntax below)

Here's another example, with complex numbers

```{code-cell} julia
x, y = 1.0 + 1.0im, 1.0 + 1.0im
@which +(x, y)
```

Again, the call has been dispatched to a + method specifically designed for handling the given data type.

#### Adding Methods

It's straightforward to add methods to existing functions.

For example, we can't at present add an integer and a string in Julia (i.e. `100 + "100"` is not valid syntax).

This is sensible behavior, but if you want to change it there's nothing to stop you.

```{code-cell} none
import Base: +  # enables adding methods to the + function
+(x::Integer, y::String) = x + parse(Int, y)
@show +(100, "100")
@show 100 + "100";  # equivalent
```

The above code is not executed to avoid any chance of a [method invalidation](https://julialang.org/blog/2020/08/invalidations/), where are a source of compile-time latency.

### Understanding the Compilation Process

We can now be a little bit clearer about what happens when you call a function on given types.

Suppose we execute the function call `f(a, b)` where `a` and `b`
are of concrete types `S` and `T` respectively.

The Julia interpreter first queries the types of `a` and `b` to obtain the tuple `(S, T)`.

It then parses the list of methods belonging to `f`, searching for a match.

If it finds a method matching `(S, T)` it calls that method.

If not, it looks to see whether the pair `(S, T)` matches any method defined for *immediate parent types*.

For example, if `S` is `Float64` and `T` is `ComplexF32` then the
immediate parents are `AbstractFloat` and `Number` respectively

```{code-cell} julia
supertype(Float64)
```

```{code-cell} julia
supertype(ComplexF32)
```

Hence the interpreter looks next for a method of the form `f(x::AbstractFloat, y::Number)`.

If the interpreter can't find a match in immediate parents (supertypes) it proceeds up the tree, looking at the parents of the last type it checked at each iteration.

* If it eventually finds a matching method, it invokes that method.
* If not, we get an error.

This is the process that leads to the following error (since we only added the `+` for adding `Integer` and `String` above)

```{code-cell} julia
---
tags: [raises-exception]
---
@show (typeof(100.0) <: Integer) == false
100.0 + "100"
```

Because the dispatch procedure starts from concrete types and works upwards, dispatch always invokes the *most specific method* available.

For example, if you have methods for function `f` that handle

1. `(Float64, Int64)` pairs
1. `(Number, Number)` pairs

and you call `f` with `f(0.5, 1)` then the first method will be invoked.

This makes sense because (hopefully) the first method is optimized for
exactly this kind of data.

The second method is probably more of a "catch all" method that handles other
data in a less optimal way.

Here's another simple example, involving a user-defined function

```{code-cell} julia
function q(x)  # or q(x::Any)
    println("Default (Any) method invoked")
end

function q(x::Number)
    println("Number method invoked")
end

function q(x::Integer)
    println("Integer method invoked")
end
```

Let's now run this and see how it relates to our discussion of method dispatch
above

```{code-cell} julia
q(3)
```

```{code-cell} julia
q(3.0)
```

```{code-cell} julia
q("foo")
```

Since `typeof(3) <: Int64 <: Integer <: Number`, the call `q(3)` proceeds up the tree to `Integer` and invokes `q(x::Integer)`.

On the other hand, `3.0` is a `Float64`, which is not a subtype of  `Integer`.

Hence the call `q(3.0)` continues up to `q(x::Number)`.

Finally, `q("foo")` is handled by the function operating on `Any`, since `String` is not a subtype of `Number` or `Integer`.

### Analyzing Function Return Types

For the most part, time spent "optimizing" Julia code to run faster is about ensuring the compiler can correctly deduce types for all functions.

The macro `@code_warntype` gives us a hint

```{code-cell} julia
x = [1, 2, 3]
f(x) = 2x
@code_warntype f(x)
```

The `@code_warntype` macro compiles `f(x)` using the type of `x` as an example -- i.e., the `[1, 2, 3]` is used as a prototype for analyzing the compilation, rather than simply calculating the value.

Here, the `Body::Array{Int64,1}` tells us the type of the return value of the
function, when called with types like `[1, 2, 3]`, is always a vector of integers.

In contrast, consider a function potentially returning `nothing`, as in {doc}`this lecture <../getting_started_julia/fundamental_types>`

```{code-cell} julia
f(x) = x > 0.0 ? x : nothing
@code_warntype f(1)
```

This states that the compiler determines the return type when called with an integer (like `1`) could be one of two different types, `Body::Union{Nothing, Int64}`.

A final example is a variation on the above, which returns the maximum of `x` and `0`.

```{code-cell} julia
f(x) = x > 0.0 ? x : 0.0
@code_warntype f(1)
```

Which shows that, when called with an integer, the type could be that integer or the floating point `0.0`.

On the other hand, if we use change the function to return `0` if x <= 0, it is type-unstable with  floating point.

```{code-cell} julia
f(x) = x > 0.0 ? x : 0
@code_warntype f(1.0)
```

The solution is to use the `zero(x)` function which returns the additive identity element of type `x`.

On the other hand, if we change the function to return `0` if `x <= 0`, it is type-unstable with  floating point.

```{code-cell} julia
@show zero(2.3)
@show zero(4)
@show zero(2.0 + 3im)

f(x) = x > 0.0 ? x : zero(x)
@code_warntype f(1.0)
```

## Foundations

Let's think about how quickly code runs, taking as given

* hardware configuration
* algorithm (i.e., set of instructions to be executed)

We'll start by discussing the kinds of instructions that machines understand.

### Machine Code

All instructions for computers end up as *machine code*.

Writing fast code --- expressing a given algorithm so that it runs quickly --- boils down to producing efficient machine code.

You can do this yourself, by hand, if you want to.

Typically this is done by writing [assembly](https://en.wikipedia.org/wiki/Assembly_language), which is a symbolic representation of machine code.

Here's some assembly code implementing a function that takes arguments $a, b$ and returns $2a + 8b$

```{code-block} asm
    pushq   %rbp
    movq    %rsp, %rbp
    addq    %rdi, %rdi
    leaq    (%rdi,%rsi,8), %rax
    popq    %rbp
    retq
    nopl    (%rax)
```

Note that this code is specific to one particular piece of hardware that we use --- different machines require different machine code.

If you ever feel tempted to start rewriting your economic model in assembly, please restrain yourself.

It's far more sensible to give these instructions in a language like Julia,
where they can be easily written and understood.

```{code-cell} julia
function f(a, b)
    y = 2a + 8b
    return y
end
```

or Python

```{code-block} python
def f(a, b):
    y = 2 * a + 8 * b
    return y
```

or even C

```{code-block} c
int f(int a, int b) {
    int y = 2 * a + 8 * b;
    return y;
}
```

In any of these languages we end up with code that is much easier for humans to write, read, share and debug.

We leave it up to the machine itself to turn our code into machine code.

How exactly does this happen?

### Generating Machine Code

The process for turning high level code into machine code differs across
languages.

Let's look at some of the options and how they differ from one another.

#### AOT Compiled Languages

Traditional compiled languages like Fortran, C and C++ are a reasonable option for writing fast code.

Indeed, the standard benchmark for performance is still well-written C or Fortran.

These languages compile down to efficient machine code because users are forced to provide a lot of detail on data types and how the code will execute.

The compiler therefore has ample information for building the corresponding machine code ahead of time (AOT) in a way that

* organizes the data optimally in memory and
* implements efficient operations as required for the task in hand

At the same time, the syntax and semantics of C and Fortran are verbose and unwieldy when compared to something like Julia.

Moreover, these low level languages lack the interactivity that's so crucial for scientific work.

#### Interpreted Languages

Interpreted languages like Python generate machine code "on the fly", during program execution.

This allows them to be flexible and interactive.

Moreover, programmers can leave many tedious details to the runtime environment, such as

* specifying variable types
* memory allocation/deallocation, etc.

But all this convenience and flexibility comes at a cost: it's hard to turn
instructions written in these languages into efficient machine code.

For example, consider what happens when Python adds a long list of numbers
together.

Typically the runtime environment has to check the type of these objects one by one before it figures out how to add them.

This involves substantial overheads.

There are also significant overheads associated with accessing the data values themselves, which might not be stored contiguously in memory.

The resulting machine code is often complex and slow.

#### Just-in-time compilation

Just-in-time (JIT) compilation is an alternative approach that marries some of
the advantages of AOT compilation and interpreted languages.

The basic idea is that functions for specific tasks are compiled as requested.

As long as the compiler has enough information about what the function does,
it can in principle generate efficient machine code.

In some instances, all the information is supplied by the programmer.

In other cases, the compiler will attempt to infer missing information on the fly based on usage.

Through this approach, computing environments built around JIT compilers aim to

* provide all the benefits of high level languages discussed above and, at the same time,
* produce efficient instruction sets when functions are compiled down to machine code

## JIT Compilation in Julia

JIT compilation is the approach used by Julia.

In an ideal setting, all information necessary to generate efficient native machine code is supplied or inferred.

In such a setting, Julia will be on par with machine code from low level languages.

### An Example

Consider the function

```{code-cell} julia
function f(a, b)
    y = (a + 8b)^2
    return 7y
end
```

Suppose we call `f` with integer arguments (e.g., `z = f(1, 2)`).

The JIT compiler now knows the types of `a` and `b`.

Moreover, it can infer types for other variables inside the function

* e.g., `y` will also be an integer

It then compiles a specialized version of the function to handle integers and
stores it in memory.

We can view the corresponding machine code using the @code_native macro

```{code-cell} julia
@code_native f(1, 2)
```

If we now call `f` again, but this time with floating point arguments, the JIT compiler will once more infer types for the other variables inside the function.

* e.g., `y` will also be a float

It then compiles a new version to handle this type of argument.

```{code-cell} julia
@code_native f(1.0, 2.0)
```

Subsequent calls using either floats or integers are now routed to the appropriate compiled code.

### Potential Problems

In some senses, what we saw above was a best case scenario.

Sometimes the JIT compiler produces messy, slow machine code.

This happens when type inference fails or the compiler has insufficient information to optimize effectively.

The next section looks at situations where these problems arise and how to get around them.

## Fast and Slow Julia Code

To summarize what we've learned so far, Julia provides a platform for generating highly efficient machine code with relatively little effort by combining

1. JIT compilation
1. Optional type declarations and type inference to pin down the types of variables and hence compile efficient code
1. Multiple dispatch to facilitate specialization and optimization of compiled code for different data types

But the process is not flawless, and hiccups can occur.

The purpose of this section is to highlight potential issues and show you how
to circumvent them.

### BenchmarkTools

The main Julia package for benchmarking is [BenchmarkTools.jl](https://www.github.com/JuliaCI/BenchmarkTools.jl).

Below, we'll use the `@btime` macro it exports to evaluate the performance of Julia code.

As mentioned in an {doc}`earlier lecture <../software_engineering/testing>`, we can also save benchmark results to a file and guard against performance regressions in code.

For more, see the package docs.

### Global Variables

Global variables are names assigned to values outside of any function or type definition.

The are convenient and novice programmers typically use them with abandon.

But global variables are also dangerous, especially in medium to large size programs, since

* they can affect what happens in any part of your program
* they can be changed by any function

This makes it much harder to be certain about what some  small part of a given piece of code actually commands.

Here's a [useful discussion on the topic](http://wiki.c2.com/?GlobalVariablesAreBad).

When it comes to JIT compilation, global variables create further problems.

The reason is that the compiler can never be sure of the type of the global
variable, or even that the type will stay constant while a given function runs.

To illustrate, consider this code, where `b` is global

```{code-cell} julia
b = 1.0
function g(a)
    global b
    tmp = a
    for i in 1:1_000_000
        tmp = tmp + a + b
    end
    return tmp
end
```

The code executes relatively slowly and uses a huge amount of memory.

```{code-cell} julia
using BenchmarkTools

@btime g(1.0)
```

If you look at the corresponding machine code you will see that it's a mess.

```{code-cell} julia
@code_native g(1.0)
```

If we eliminate the global variable like so

```{code-cell} julia
function g(a, b)
    tmp = a
    for i in 1:1_000_000
        tmp = tmp + a + b
    end
    return tmp
end
```

then execution speed improves dramatically.  Furthermore, the number of allocations has dropped to zero.

```{code-cell} julia
@btime g(1.0, 1.0)
```

Note that if you called `@time` instead, the first call would be slower as it would need to compile the function.  Conversely, `@btime` discards the first call and runs it multiple times.

More information is available with `@benchmark` instead,
```{code-cell} julia
@benchmark g(1.0, 1.0)
```

Also, the machine code is simple and clean

```{code-cell} julia
@code_native g(1.0, 1.0)
```

Now the compiler is certain of types throughout execution of the function and
hence can optimize accordingly.

If global variations are strictly needed (and they almost never are) then you can declare them with a `const` to declare to Julia that the type never changes (the value can).  For example, 

```{code-cell} julia
const b_const = 1.0
function g_const(a)
    global b_const
    tmp = a
    for i in 1:1_000_000
        tmp = tmp + a + b_const
    end
    return tmp
end
@btime g_const(1)
```

Now the compiler can again generate efficient machine code.

However, global variables within a function is almost always a bad idea.  Instead, the `b_const` should be passed as a parameter to the function.

### Composite Types with Abstract Field Types

Another scenario that trips up the JIT compiler is when composite types have
fields with abstract types.

We met this issue {ref}`earlier <spec_field_types>`, when we discussed AR(1) models.

Let's experiment, using, respectively,

* an untyped field
* a field with abstract type, and
* parametric typing

As we'll see, the last of these options gives us the best performance, while still maintaining significant flexibility.

Here's the untyped case

```{code-cell} julia
struct Foo_any
    a
end
```

Here's the case of an abstract type on the field `a`

```{code-cell} julia
struct Foo_abstract
    a::Real
end
```

Finally, here's the parametrically typed case (where the `{T <: Real}` is not necessary for performance, and could simply be `{T}`

```{code-cell} julia
struct Foo_concrete{T <: Real}
    a::T
end
```

Now we generate instances

```{code-cell} julia
fg = Foo_any(1.0)
fa = Foo_abstract(1.0)
fc = Foo_concrete(1.0)
```

In the last case, concrete type information for the fields is embedded in the object

```{code-cell} julia
typeof(fc)
```

This is significant because such information is detected by the compiler.

#### Timing

Here's a function that uses the field `a` of our objects

```{code-cell} julia
function f(foo)
    tmp = foo.a
    for i in 1:1_000_000
        tmp = i + foo.a
    end
    return tmp
end
```

Let's try timing our code, starting with the case without any constraints:

```{code-cell} julia
@btime f($fg)
```

The timing is not very impressive.

Here's the nasty looking machine code

```{code-cell} julia
@code_native f(fg)
```

The abstract case is almost identical,

```{code-cell} julia
@btime f($fa)
```

Note the large memory footprint.

The machine code is also long and complex, although we omit details.

Finally, let's look at the parametrically typed version

```{code-cell} julia
@btime f($fc)
```

Which is improbably small - since a runtime of 1-2 nanoseconds without any allocations suggests no computations really took place.

A hint is in the simplicity of the corresponding machine code

```{code-cell} julia
@code_native f(fc)
```

This machine code has none of the hallmark assembly instructions associated with a loop, in particular loops (e.g. `for` in julia) end up as jumps in the machine code (e.g. `jne`).

Here, the compiler was smart enough to realize that only the final step in the loop matters, i.e. it could generate the equivalent to `f(a) = 1_000_000 + foo.a` and then it can return that directly and skip the loop.  These sorts of code optimizations are only possible if the compiler is able to use a great deal of information about the types.


Finally, note that if we compile a slightly different version of the function, which doesn't actually return the value
```{code-cell} julia
function f_no_return(foo)
    for i in 1:1_000_000
        tmp = i + foo.a
    end
end
@code_native f_no_return(fc)
```
We see that the code is even simpler.  In effect, if figured out that because `tmp` wasn't returned, and the `foo.a` could have no side effects (since it knows the type of `a`), that it doesn't even need to execute any of the code in the function.

### Type Inference

Consider the following function, which essentially does the same job as Julia's `sum()` function but acts only on floating point data

```{code-cell} julia
function sum_float_array(x::AbstractVector{<:Number})
    sum = 0.0
    for i in eachindex(x)
        sum += x[i]
    end
    return sum
end
```

Calls to this function run very quickly

```{code-cell} julia
x_range = range(0, 1, length = 100_000)
x = collect(x_range)
typeof(x)
```

```{code-cell} julia
@btime sum_float_array($x)
```

When Julia compiles this function, it knows that the data passed in as `x` will be an array of 64 bit floats.  Hence it's known to the compiler that the relevant method for `+` is always addition of floating point numbers.

But consider a version without that type annotation

```{code-cell} julia
function sum_array(x)
    sum = 0.0
    for i in eachindex(x)
        sum += x[i]
    end
    return sum
end
@btime sum_array($x)
```

Note that this has the same running time as the one with the explicit types.  In julia, there is (almost never) performance gain from declaring types, and if anything they can make things worse by limited the potential for specialized algorithms.  See {doc}`generic programming <../more_julia/generic_programming>` for more.

As an example within Julia code, look at the built-in sum for array

```{code-cell} julia
@btime sum($x)
```

Versus the underlying range

```{code-cell} julia
@btime sum($x_range)
```

Note that the difference in speed is enormous---suggesting it is better to keep things in their more structured forms as long as possible.  You can check the underlying source used for this with `@which sum(x_range)` to see the specialized algorithm used.

#### Type Inferences

Here's the same function minus the type annotation in the function signature

```{code-cell} julia
function sum_array(x)
    sum = 0.0
    for i in eachindex(x)
        sum += x[i]
    end
    return sum
end
```

When we run it with the same array of floating point numbers it executes at a
similar speed as the function with type information.

```{code-cell} julia
@btime sum_array($x)
```

The reason is that when `sum_array()` is first called on a vector of a given
data type, a newly compiled version of the function is produced to handle that
type.

In this case, since we're calling the function on a vector of floats, we get a compiled version of the function with essentially the same internal representation as `sum_float_array()`.

#### An Abstract Container

Things get tougher for the interpreter when the data type within the array is imprecise.

For example, the following snippet creates an array where the element type is `Any`

```{code-cell} julia
x = Any[1 / i for i in 1:1e6];
```

```{code-cell} julia
eltype(x)
```

Now summation is much slower and memory management is less efficient.

```{code-cell} julia
@btime sum_array($x)
```

## Further Comments

Here are some final comments on performance.

### Summary and Tips

Use functions to segregate operations into logically distinct blocks.

Data types will be determined at function boundaries.

If types are not supplied then they will be inferred.

If types are stable and can be inferred effectively your functions will run fast.

### Further Reading

A good next stop for further reading is the [relevant part](https://docs.julialang.org/en/v1/manual/performance-tips/) of the Julia documentation.

