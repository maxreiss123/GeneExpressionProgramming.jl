#=
What `dimension_homogene` promises: the model a flagged individual is scored as is
dimensionally homogeneous. Checked against the functions themselves rather than against the
unit rules alone: a rule is sound if rescaling the units of the operands rescales the result
as the rule's dimension says, and a homogeneous model rescales like its target. Covered: every
scalar unit rule, the tensor rules of `inv` and `hadamard`, the gene-by-gene contract that a
model scored as the weighted sum of its genes (linear scaling) needs, and the check that a
repair must pass before its individual is flagged.
=#

using Test
using Random
using Statistics
using LinearAlgebra
using Tensors

const SBPH = GeneExpressionProgramming.SBPUtils
const RWH = GeneExpressionProgramming.RegressionWrapper
const GEH = GeneExpressionProgramming.GepEntities

h7(v...) = Float16[v..., zeros(Float16, 7 - length(v))...]
# the factor a quantity of dimension `d` takes when the base units are rescaled by `λ`
unit_factor(d, λ) = prod(λ[k]^Float64(d[k]) for k in eachindex(λ))

@testset "Homogeneity of flagged models" begin
    @testset "scalar unit rules hold under a rescaling of the units" begin
        # rule F of f is sound iff f(λ^d1 x, λ^d2 y) == λ^F(d1, d2) f(x, y) whenever F is
        # consistent; `floor(λx) ≠ λ floor(x)`, for one, so `floor` must refuse units
        Random.seed!(3)
        pool = [h7(), h7(1), h7(0, 1, -1), h7(1, 2, -2), h7(0, 2), h7(0, -1)]
        unsound = Symbol[]
        for (name, F) in RWH.FUNCTION_LIB_FORWARD_COMMON
            f = GeneExpressionProgramming.GepUtils.FUNCTION_LIB_COMMON[name]
            unary = GeneExpressionProgramming.GepUtils.ARITY_LIB_COMMON[name] == 1
            for _ in 1:40
                λ = rand(7) .* 3 .+ 0.2
                d1 = rand(pool)
                d2 = rand() < 0.5 ? d1 : rand(pool)
                # arguments in (0.1, 0.9), in every function's domain (acosh: (1.1, 1.9))
                x = rand() * 0.8 + 0.1 + (name === :acosh)
                y = rand() * 0.8 + 0.1
                Fd = unary ? F(copy(d1)) : F(copy(d1), copy(d2))
                SBPH.is_feasible(SBPH.canonical_dim(Fd)) || continue
                v = unary ? f(x) : f(x, y)
                v2 = unary ? f(x * unit_factor(d1, λ)) :
                     f(x * unit_factor(d1, λ), y * unit_factor(d2, λ))
                if !isapprox(v2, v * unit_factor(Fd, λ); rtol=1e-9, atol=1e-12)
                    push!(unsound, name)
                    break
                end
            end
        end
        @test isempty(unsound)
        # the rules that used to keep the operand's dimension
        @test RWH.FUNCTION_LIB_FORWARD_COMMON[:sign](h7(0, 1)) == h7()
        @test SBPH.has_inf16(RWH.FUNCTION_LIB_FORWARD_COMMON[:floor](h7(0, 1)))
        @test RWH.FUNCTION_LIB_FORWARD_COMMON[:round](h7()) == h7()
    end

    @testset "tensor rules of inv and hadamard" begin
        # [order, kg, m, s, K, mol, A, cd]; the order is never rescaled
        u(o, kg, m) = Float16[o, kg, m, 0, 0, 0, 0, 0]
        λ = [1.0, 1.7, 2.3, 1, 1, 1, 1, 1]
        si(d) = unit_factor(d[2:end], λ[2:end])
        Random.seed!(4)
        A = Tensor{2,3}(rand(3, 3) + 3I)
        B = Tensor{2,3}(rand(3, 3))
        dA, dB = u(2, 1, -1), u(2, 0, 2)

        f_inv = RWH.FUNCTION_LIB_FORWARD_COMMON_TENSOR[:inv]
        @test f_inv(copy(dA)) == u(2, -1, 1)
        @test inv(A * si(dA)) ≈ inv(A) * si(f_inv(copy(dA)))
        @test SBPH.has_inf16(f_inv(u(0, 1, 0)))              # no inverse of a scalar
        f_had = RWH.FUNCTION_LIB_FORWARD_COMMON_TENSOR[:hadamard]
        @test f_had(copy(dA), copy(dB)) == u(2, 1, 1)
        @test (A * si(dA)) ∘ (B * si(dB)) ≈ (A ∘ B) * si(f_had(copy(dA), copy(dB)))
        @test SBPH.has_inf16(f_had(u(2, 0, 0), u(1, 0, 0)))  # orders must agree

        # the backward rules give operands that the forward rules take to the requirement
        E = u(2, 1, 1)
        @test f_inv(RWH.FUNCTION_LIB_BACKWARD_COMMON_TENSOR[:inv](copy(E))) == E
        b_had = RWH.FUNCTION_LIB_BACKWARD_COMMON_TENSOR[:hadamard]
        for (l, r) in ((dA, dB), (u(0, 0, 0), dB), (u(0, 0, 0), u(1, 0, 0)))
            p, q = b_had(copy(l), copy(r), copy(E))
            @test f_had(p, q) == E
        end
    end

    # three features m, s and kg; the target is a force
    dims = Dict{Symbol,Vector{Float16}}(:x1 => h7(0, 1), :x2 => h7(0, 0, 1), :x3 => h7(1))
    target = h7(1, 1, -2)
    fixture(conns; nt=[:+, :-, :*, :/]) = GepRegressor(3; considered_dimensions=dims,
        entered_non_terminals=nt, gene_connections=conns, gene_count=3, head_len=5,
        max_permutations_lib=3000, rounds=4)

    @testset "gene dimensions" begin
        reg = fixture([:+, :-, :*, :/])
        dto = reg.token_dto_
        sym(f) = only(k for (k, v) in reg.toolbox_.callbacks if string(v) == f)
        dimdict = dto.tokenLib.physical_dimension_dict[]
        x(d) = only(k for (k, v) in dimdict if v == d && !haskey(reg.toolbox_.callbacks, k))
        m, s, kg = x(h7(0, 1)), x(h7(0, 0, 1)), x(h7(1))
        # * joins genes [kg m], [1] and [s^-2] into a force: homogeneous as a whole, but the
        # weighted sum of the genes is not
        prod_form = Int8[sym("*"), sym("*"), sym("*"), kg, m, sym("/"), kg, kg,
            sym("/"), sym("/"), kg, s, sym("*"), kg, s]
        @test is_dimensionally_homogeneous(prod_form, target, dto)
        @test gene_dimensions(prod_form, 3, dto) == [h7(1, 1), h7(), h7(0, 0, -2)]
        @test !is_gene_wise_homogeneous(prod_form, target, dto, 3)
        # three force genes joined by + meet both
        force = Int8[sym("/"), sym("*"), kg, m, sym("*"), s, s]
        sum_form = vcat(Int8[sym("+"), sym("-")], force, force, force)
        @test is_gene_wise_homogeneous(sum_form, target, dto, 3)
        @test is_dimensionally_homogeneous(sum_form, target, dto)
        # a malformed string has no gene dimensions
        @test gene_dimensions(sum_form[1:end-1], 3, dto) === nothing
        @test !is_gene_wise_homogeneous(sum_form[1:end-1], target, dto, 3)
    end

    @testset "gene-wise repair" begin
        Random.seed!(5)
        for conns in ([:+, :-, :*, :/], [:*, :/])
            reg = fixture(conns)
            tb, dto = reg.toolbox_, reg.token_dto_
            corr, check = RWH.make_dimension_contract(tb, dto, target, 10; gene_wise=true)
            pop = GEH.generate_population(150, tb)
            repaired = 0
            for c in pop
                GEH.compile_expression!(c; force_compile=true)
                check(c.expression_raw) && continue
                before = copy(c.genes)
                _, ok = corr(c.genes, tb.gen_start_indices, c.expression_raw, 0)
                if !ok
                    @test c.genes == before
                    continue
                end
                repaired += 1
                GEH.compile_expression!(c; force_compile=true)
                @test check(c.expression_raw)
                @test is_gene_wise_homogeneous(c.expression_raw, target, dto, tb.gene_count)
                # the genes stay ordinary genes, and + or - join them where offered
                for part in GEH._karva_raw(c; split=true)[2:end]
                    @test length(part) <= 2 * tb.head_len + 1
                    @test SBPH.last_operator_position(dto.index, collect(part)) <= tb.head_len
                end
                @test all(c.genes[k] in tb.gene_connections for k in 1:tb.gene_count-1)
                if :+ in conns
                    @test is_dimensionally_homogeneous(c.expression_raw, target, dto)
                end
            end
            @test repaired >= 100
        end
    end

    @testset "a repair is flagged only if its result passes the check" begin
        Random.seed!(6)
        reg = fixture([:+, :-, :*, :/])
        tb, dto = reg.toolbox_, reg.token_dto_
        check = e -> is_dimensionally_homogeneous(e, target, dto)
        pop = GEH.generate_population(60, tb)
        foreach(c -> GEH.compile_expression!(c; force_compile=true), pop)
        RWH.make_lib_seeder(tb, dto, target, 0.5)(pop)   # half of them born homogeneous
        # a repair that claims success without changing anything
        claims = (genes, starts, expression, generation) -> (Float16(0), true)
        GeneExpressionProgramming.GepRegression.perform_correction_callback!(pop, 1, 1, 1.0,
            claims; homogeneity_check=check)
        @test count(c -> c.dimension_homogene, pop) > 0
        @test all(c -> !c.dimension_homogene || check(c.expression_raw), pop)
        # what the claim did not fix is not scored
        @test all(c -> c.dimension_homogene || c.fitness == tb.fitness_reset[1], pop)
    end

    @testset "linear scaling holds every gene to the target" begin
        # the scaled model is the weighted sum of its genes; with * or / among the
        # connectors, the connected expression can have the target while the genes do not
        Random.seed!(7)
        x = abs.(randn(3, 120)) .+ 0.5
        y = vec(x[3, :] .* x[1, :] ./ x[2, :] .^ 2)
        reg = fixture([:+, :-, :*, :/])
        dto = reg.token_dto_
        flagged = Ref(0)
        off = Ref(0)
        logger = function (population, epoch, selected)
            for c in population
                c.dimension_homogene || continue
                flagged[] += 1
                (is_gene_wise_homogeneous(c.expression_raw, target, dto, 3) &&
                 is_dimensionally_homogeneous(c.expression_raw, target, dto)) || (off[] += 1)
            end
        end
        fit!(reg, 15, 200, x, y; target_dimension=target, linear_scaling=true,
            file_logger_callback=logger)
        @test flagged[] > 1000
        @test off[] == 0
        # the fitted models rescale like a force when metres, seconds and kilograms do
        λ = [1.9, 2.3, 0.6, 1, 1, 1, 1]
        x2 = copy(x)
        for (i, d) in enumerate((h7(0, 1), h7(0, 0, 1), h7(1)))
            x2[i, :] .*= unit_factor(d, λ)
        end
        for c in reg.best_models_
            @test c.dimension_homogene
            @test c(x2) ≈ unit_factor(target, λ) .* c(x) rtol = 1e-9
        end
    end

    @testset "tensor path: gene_wise_dimension" begin
        # Maxwell stress tensor terminals with SI units, `*` among the connectors
        u(o, kg, m, sec, A) = Float16[o, kg, m, sec, 0, 0, A, 0]
        tdims = [u(0, 2, 2, -6, -2), u(0, 2, 0, -4, -2), u(0, -1, -3, 4, 2),
            u(0, 1, 1, -2, -2), u(2, 2, 2, -6, -2), u(2, 2, 0, -4, -2), u(2, 0, 0, 0, 0)]
        ttarget = u(2, 1, -1, -2, 0)
        Random.seed!(8)
        reg = GepTensorRegressor(7; problem_dimension=3, entered_non_terminals=[:+, :-, :*, :/],
            entered_terminal_nums=[0.5], gene_connections=[:+, :-, :*], gene_count=3,
            head_len=4, max_permutations_lib=2000, rounds=3,
            considered_dimensions=Dict{Symbol,Vector{Float16}}(
                Symbol("x$i") => tdims[i] for i in 1:7))
        dto = reg.token_dto_
        flagged = Ref(0)
        off = Ref(0)
        logger = function (population, epoch, selected)
            for c in population
                c.dimension_homogene || continue
                flagged[] += 1
                is_gene_wise_homogeneous(c.expression_raw, ttarget, dto, 3) || (off[] += 1)
            end
        end
        # only the dimension bookkeeping is under test, so the loss scores at random
        loss = (elem, validate) -> ((isnan(mean(elem.fitness)) || validate) &&
                                    (elem.fitness = (rand(),)); nothing)
        fit!(reg, 8, 200, loss; target_dimension=ttarget, gene_wise_dimension=true,
            file_logger_callback=logger)
        @test flagged[] > 500
        @test off[] == 0
    end

    @testset "invariants and a tensor basis, the tensor marked in the 7th slot" begin
        # an OpenFOAM-style closure: invariants I1..I4 (I3 and I4 in metres) and tensor basis
        # terms T1..T4, the 7th slot marking the tensor; the target m^3 T asks every term for
        # three metres and one tensor factor
        dims8 = Dict{Symbol,Vector{Float16}}(
            :x1 => h7(), :x2 => h7(), :x3 => h7(0, 1), :x4 => h7(0, 1),
            :x5 => h7(0, 0, 0, 0, 0, 0, 1), :x6 => h7(0, 1, 0, 0, 0, 0, 1),
            :x7 => h7(0, 1, 0, 0, 0, 0, 1), :x8 => h7(0, 1, 0, 0, 0, 0, 1))
        t8 = h7(0, 3, 0, 0, 0, 0, 1)
        # metres doubled and the tensor slot quadrupled: a homogeneous model's output takes a
        # factor 2^3 * 4 = 32, and exactly so, since powers of two commute with rounding
        scale8 = [unit_factor(dims8[Symbol("x$i")], [1, 2, 1, 1, 1, 1, 4]) for i in 1:8]
        rescales(c, ctx1, ctx2) = begin
            p1, p2 = c(ctx1), c(ctx2)
            (p1 isa AbstractVector && p2 isa AbstractVector) || return true   # not evaluable
            p1, p2 = copy(p1), copy(p2)
            all(i -> !(isfinite(p1[i]) && isfinite(p2[i])) || p2[i] ≈ 32 * p1[i], eachindex(p1))
        end
        Random.seed!(9)
        reg = GepRegressor(8; considered_dimensions=dims8, gene_count=3, head_len=6)
        tb, dto = reg.toolbox_, reg.token_dto_
        corr, check = RWH.make_dimension_contract(tb, dto, t8, 10)
        x = rand(8, 32) .+ 0.5
        ctx1, ctx2 = GEH.buffer_context(tb, x), GEH.buffer_context(tb, x .* scale8)
        repaired = failed = 0
        for c in GEH.generate_population(300, tb)
            GEH.compile_expression!(c; force_compile=true)
            check(c.expression_raw) && continue
            _, ok = corr(c.genes, tb.gen_start_indices, c.expression_raw, 0)
            ok || (failed += 1; continue)
            repaired += 1
            GEH.compile_expression!(c; force_compile=true)
            @test check(c.expression_raw)
            @test rescales(c, ctx1, ctx2)
        end
        @test repaired >= 0.95 * (repaired + failed)

        # a scaled search on a ground truth linear in the tensor terms: every flagged
        # individual, weighted genes and all, rescales like the target
        x = rand(8, 200) .+ 0.5
        y = x[1, :] .* x[3, :] .* x[4, :] .* x[6, :] .+
            0.5 .* x[2, :] .* x[3, :] .^ 2 .* x[7, :] .- x[3, :] .* x[4, :] .^ 2 .* x[5, :]
        ctx1, ctx2 = GEH.buffer_context(tb, x), GEH.buffer_context(tb, x .* scale8)
        flagged = Ref(0)
        off = Ref(0)
        logger = function (population, epoch, selected)
            for c in population
                c.dimension_homogene || continue
                flagged[] += 1
                rescales(c, ctx1, ctx2) || (off[] += 1)
            end
        end
        fit!(reg, 12, 200, x, y; target_dimension=t8, linear_scaling=true,
            file_logger_callback=logger)
        @test flagged[] > 1000
        @test off[] == 0
        @test reg.best_models_[1].dimension_homogene
    end
end
