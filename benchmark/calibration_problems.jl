#=
The calibration problems of benchmark/screened_constants.jl and
benchmark/swarm_variants.jl: losses of a few constants, most of them calibrations of the
coefficients of an ODE against trajectories, where every loss call solves the ODE.
`PROBLEMS` holds (name, loss, minimum, starts), the starts a function of a random number
generator. Included after the package.
=#

using LinearAlgebra
using Random
using Statistics

rosenbrock(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

# a quadratic with condition number `cond`, its axes rotated at random, minimum 0 at a
# point between -1 and 1
function rotated_quadratic(n, cond; seed=1)
    Q = Matrix(qr(randn(MersenneTwister(seed), n, n)).Q)
    A = Q * Diagonal(exp.(range(0, log(cond); length=n))) * Q'
    center = collect(range(-1, 1; length=n))
    return x -> dot(x .- center, A * (x .- center))
end

# the states every `every` steps of classical Runge-Kutta, or nothing if they blow up
function rk4(rhs, x0, dt, steps, every)
    x = copy(x0)
    out = zeros(length(x0), steps ÷ every)
    for s in 1:steps
        k1 = rhs(x)
        k2 = rhs(x .+ 0.5dt .* k1)
        k3 = rhs(x .+ 0.5dt .* k2)
        k4 = rhs(x .+ dt .* k3)
        x = x .+ dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        (all(isfinite, x) && maximum(abs, x) < 1e6) || return nothing
        s % every == 0 && (out[:, s÷every] .= x)
    end
    return out
end

# logistic growth dx/dt = a x - b x^2 from 8 initial values
const LOGISTIC_X0 = collect(range(0.05, 3.5; length=8))
logistic_rhs(p) = x -> p[1] .* x .- p[2] .* x .^ 2
simulate_logistic(p) = rk4(logistic_rhs(p), LOGISTIC_X0, 0.004, 2000, 25)

# Lotka-Volterra dx/dt = a x - b x y, dy/dt = d x y - c y from 3 initial states
const LV_X0 = [0.5, 1.0, 1.5, 0.5, 2.0, 1.5]
function lv_rhs(p)
    return z -> begin
        x, y = z[1:2:end], z[2:2:end]
        dz = similar(z)
        dz[1:2:end] .= p[1] .* x .- p[2] .* x .* y
        dz[2:2:end] .= p[4] .* x .* y .- p[3] .* y
        dz
    end
end
simulate_lv(p) = rk4(lv_rhs(p), LV_X0, 0.01, 800, 10)

# the loss of a calibration: the mean squared distance of the simulated states from the
# data, and 1e3 for a simulation that blows up
function calibration(simulate, data)
    return p -> begin
        sim = simulate(p)
        isnothing(sim) ? 1e3 : mean(abs2, sim .- data)
    end
end

const LOGISTIC_TRUE = [1.3, 0.54]
const LV_TRUE = [1.1, 0.9, 1.0, 1.2]
const LOGISTIC_DATA = simulate_logistic(LOGISTIC_TRUE)
const LV_DATA = simulate_lv(LV_TRUE)
const LOGISTIC_NOISY = LOGISTIC_DATA .+ 0.02 .* randn(MersenneTwister(3), size(LOGISTIC_DATA))
const LV_NOISY = LV_DATA .+ 0.02 .* randn(MersenneTwister(4), size(LV_DATA))

# the Lotka-Volterra coefficients `free` calibrated, the others known
function lv_calibration(free, data)
    simulate = p -> simulate_lv(setindex!(copy(LV_TRUE), p, free))
    return calibration(simulate, data)
end

# the minimum of a loss with noisy data, by a long run of Nelder-Mead from the truth
reference(f, x) = simplex_search(f, x; method=ScreenedNelderMead(screen=false,
    max_evaluations=20_000, tolerance=1e-14)).minimum

# the oscillators x'' = -k x - c x' - a x^3 from x = 1 at rest, coefficients (k, c[, a])
oscillator(p) = z -> [z[2], -p[1] * z[1] - p[2] * z[2] - (length(p) > 2 ? p[3] : 0.0) * z[1]^3]
simulate_oscillator(p) = rk4(oscillator(p), [1.0, 0.0], 0.02, 3000, 10)
const OSCILLATOR_TRUE = [4.0, 0.02, 0.5]
oscillator_start(n) = rng -> OSCILLATOR_TRUE[1:n] .* (0.7 .+ 0.6 .* rand(rng, n))

logistic_start(rng) = [0.8, 0.3] .+ rand(rng, 2) .* [1.0, 0.5]
lv_start(free) = rng -> LV_TRUE[free] .* (0.75 .+ 0.5 .* rand(rng, length(free)))
box_start(n) = rng -> 4 .* rand(rng, n) .- 2

# name, loss, its minimum, starts
const PROBLEMS = let
    logistic_noisy = calibration(simulate_logistic, LOGISTIC_NOISY)
    lv_noisy = lv_calibration(1:4, LV_NOISY)
    [
        ("logistic growth, 2 coefficients", calibration(simulate_logistic, LOGISTIC_DATA),
            0.0, logistic_start),
        ("logistic growth, 2 coefficients, noisy data", logistic_noisy,
            reference(logistic_noisy, LOGISTIC_TRUE), logistic_start),
        ("Lotka-Volterra, 2 coefficients", lv_calibration(1:2, LV_DATA), 0.0, lv_start(1:2)),
        ("Lotka-Volterra, 3 coefficients", lv_calibration(1:3, LV_DATA), 0.0, lv_start(1:3)),
        ("Lotka-Volterra, 4 coefficients", lv_calibration(1:4, LV_DATA), 0.0, lv_start(1:4)),
        ("Lotka-Volterra, 4 coefficients, noisy data", lv_noisy, reference(lv_noisy, LV_TRUE),
            lv_start(1:4)),
        ("oscillator, 2 coefficients, several minima",
            calibration(simulate_oscillator, simulate_oscillator(OSCILLATOR_TRUE[1:2])), 0.0,
            oscillator_start(2)),
        ("Duffing oscillator, 3 coefficients, several minima",
            calibration(simulate_oscillator, simulate_oscillator(OSCILLATOR_TRUE)), 0.0,
            oscillator_start(3)),
        ("Rosenbrock, 2 constants", rosenbrock, 0.0, box_start(2)),
        ("quadratic, 3 constants, condition 100", rotated_quadratic(3, 100.0), 0.0, box_start(3)),
        ("quadratic, 5 constants, condition 10", rotated_quadratic(5, 10.0), 0.0, box_start(5)),
        ("quadratic, 6 constants, condition 30", rotated_quadratic(6, 30.0; seed=2), 0.0,
            box_start(6)),
        ("quadratic, 8 constants, condition 10", rotated_quadratic(8, 10.0; seed=3), 0.0,
            box_start(8)),
    ]
end
