#=
Allocation and time of one `fit!` of the scalar `GepRegressor` as the data lengthens: 25
generations at population 800, three features, 1 000 to 100 000 samples. Each size is
fitted once untimed first, so the numbers exclude compilation. The bytes are Julia's own
allocation counter (`@timed`), so they compare evaluators within Julia, not across
runtimes.

    julia --project=. --threads=4 paper/site_benchmark/measure_allocation.jl [--out file.json]
=#

using GeneExpressionProgramming
using Random
using JSON
using Printf

const SAMPLES = (1_000, 5_000, 20_000, 100_000)

function fit_once(n)
    Random.seed!(1)
    x = randn(3, n)
    y = @. x[1, :] * x[2, :] - 0.5 * x[3, :]
    reg = GepRegressor(3; entered_non_terminals=[:+, :-, :*, :/])
    return @timed fit!(reg, 25, 800, x, y; loss_fun="mse")
end

function main(args)
    out = ""
    for (i, a) in enumerate(args)
        a == "--out" && (out = args[i+1])
    end
    mb, secs = Float64[], Float64[]
    for n in SAMPLES
        fit_once(n)                                     # compile, untimed
        t = fit_once(n)
        push!(mb, round(t.bytes / 2^20))
        push!(secs, round(t.time; digits=2))
        @printf("%7d samples: %6.0f MB allocated, %.2f s\n", n, mb[end], secs[end])
    end
    res = Dict("samples" => collect(SAMPLES), "batched_buffers" => mb,
        "seconds_batched_buffers" => secs, "threads" => Threads.nthreads(),
        "julia_version" => string(VERSION))
    isempty(out) ? JSON.print(stdout, res) : open(io -> JSON.print(io, res, 1), out, "w")
end

main(ARGS)
