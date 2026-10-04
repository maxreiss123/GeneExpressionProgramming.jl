# Installation Guide

## Julia Requirements

Julia 1.10, the long-term support release, or newer: `Project.toml` requires it (`julia = "1.10"`). The continuous integration installs, loads and tests the package on the lowest Julia that entry allows and on the latest release, with the dependencies resolved for each as `Pkg.add` resolves them. The repository's `Manifest.toml` was generated with Julia 1.12.1; to try the repository on an older Julia, resolve without it (delete it, then `Pkg.instantiate()`). Julia is available from the [official website](https://julialang.org/downloads/), and [juliaup](https://github.com/JuliaLang/juliaup) keeps several versions side by side (`juliaup add 1.10`, then `julia +1.10`).

## Installation Methods

The latest registered release:

```julia
using Pkg
Pkg.add("GeneExpressionProgramming")
```

These pages follow the current code in the repository, which may be ahead of that release. To install it:

```julia
using Pkg
Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git")
```

To modify the source, install it in development mode, which clones the repository into your Julia development directory:

```julia
using Pkg
Pkg.develop(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git")
```

or add a local copy:

```julia
using Pkg
Pkg.add(path="/path/to/GeneExpressionProgramming.jl")
```

From a clone, the scripts in `examples/` include the sources directly and run with `julia --project=. --threads=4 examples/<script>.jl`. The `tutorial` folder holds a notebook that runs on Google Colab: a first search, then the constants of a model tuned against a loss of your own.

## Dependencies

Pkg installs the dependencies with the package. Expressions are evaluated by the package's own batched evaluator, so there is no separate expression engine. The dependencies doing the heavy lifting are:

- **Tensors.jl**: vector and tensor types and operations for `GepTensorRegressor`
- **Optim.jl**: Nelder-Mead optimisation of the numeric constants of the best model
- **NearestNeighbors.jl**: the kd-tree behind the unit repair's dimension lookups
- **ThreadsX.jl**: parallel unit repair
- **Random123.jl**: independent, reproducible random streams per parallel task
- **LRUCache.jl**: the fitness cache that detects duplicate expressions
- **StatsBase.jl**: weighted sampling of symbols when genes are generated
- **Distributions.jl**: the random constants of `GepTensorRegressor`
- **CSV.jl / DataFrames.jl / JSON.jl**: reading data files and unit descriptions in the examples and paper scripts

Tensors.jl, CSV.jl and DataFrames.jl are dependencies of the package; add them to your own environment to call them directly (for example `using Tensors` to build tensor data).

Plots.jl is not a dependency. Install it to draw results:

```julia
using Pkg
Pkg.add("Plots")
```

The scripts in `examples/` print their results; `Main_min_example.jl` and `Main_min_with_csv.jl` also plot them when Plots.jl is installed.

## Threads

The fitness evaluation, the genetic operators and the unit repair run in parallel on the threads Julia is started with:

```bash
julia --threads=auto your_script.jl
# or
JULIA_NUM_THREADS=8 julia your_script.jl
```

Evaluation buffers are allocated once per thread id (`thread_slots()`, which covers the interactive thread that Julia 1.12 starts next to the worker threads), so a custom loss can evaluate in `ctxs[Threads.threadid()]` of `thread_contexts(...)` without sizing anything itself.

The fitness loop runs one worker per thread, and each takes the next unscored individual as soon as it is free: a loss whose cost varies between individuals (a solver that converges quickly for some models and slowly for others) keeps every thread busy until the last one is scored. A call of the loss stays on one thread from start to finish and no other call shares its thread id meanwhile, so `ctxs[Threads.threadid()]` is safe even when the loss waits, e.g. on an external process.

## Verification

```julia
using GeneExpressionProgramming

regressor = GepRegressor(2)
println("Basic regressor created: ", typeof(regressor))
```

## Next Steps

1. [Getting Started](getting-started.md): a first symbolic regression
2. [Core Concepts](core-concepts.md): how the search works
3. [Examples](examples/basic-regression.md): complete workflows

For help, open an issue on the [GitHub repository](https://github.com/maxreiss123/GeneExpressionProgramming.jl/issues).
