#=
Separate the one-off Julia costs from the cost of actually solving the problem.

Reported (and written to results/precompile.json):

  deps_precompile_s     precompilation of the package dependencies; one-off per
                        environment, taken from the Pkg log of the initial
                        `Pkg.instantiate()` (pass it via --deps).
  package_precompile_s  precompilation of GeneExpressionProgramming itself,
                        measured here by invalidating its cache and rebuilding.
  load_s                `using GeneExpressionProgramming` in a process that has not
                        loaded it yet, with the cache warm -- paid once per process.
  ttfx_s                first `fit!` call (JIT of the evolutionary loop).
  second_fit_s          the identical call again: the compilation-free cost.

The benchmark scripts keep all of the above out of their reported solve times by running
one throw-away `fit!` before the timed one.

    julia --project=. paper/site_benchmark/measure_precompile.jl [--deps 205.0]
=#

using Pkg
using Printf
using JSON

const HERE = @__DIR__
const SRC = normpath(joinpath(HERE, "..", "..", "src", "GeneExpressionProgramming.jl"))

deps_s = NaN
for (i, a) in enumerate(ARGS)
    a == "--deps" && (global deps_s = parse(Float64, ARGS[i+1]))
end

# invalidate only this package's cache, then rebuild it
touch(SRC)
t_pkg = @elapsed Pkg.precompile(; io=devnull)

t_load = @elapsed (@eval using GeneExpressionProgramming)

using Random, Statistics
Random.seed!(1)
x = randn(3, 200)
y = vec(2.0 .* x[1, :] .- x[2, :] .* x[3, :])

mk() = GepRegressor(3; gene_count=2, head_len=4)
t_first = @elapsed fit!(mk(), 2, 40, x, y; loss_fun="mse")
t_second = @elapsed fit!(mk(), 2, 40, x, y; loss_fun="mse")

res = Dict(
    "deps_precompile_s" => deps_s,
    "package_precompile_s" => t_pkg,
    "load_s" => t_load,
    "ttfx_s" => t_first,
    "second_fit_s" => t_second,
    "julia_version" => string(VERSION),
    "threads" => Threads.nthreads(),
)
@printf("deps precompile   : %s\n", isnan(deps_s) ? "n/a" : @sprintf("%.1f s", deps_s))
@printf("package precompile: %.1f s\n", t_pkg)
@printf("load (warm cache) : %.1f s\n", t_load)
@printf("first fit! (JIT)  : %.1f s\n", t_first)
@printf("second fit!       : %.2f s\n", t_second)

open(joinpath(HERE, "results", "precompile.json"), "w") do f
    JSON.print(f, res)
end
