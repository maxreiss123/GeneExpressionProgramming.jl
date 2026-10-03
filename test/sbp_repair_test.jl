#=
The semantic backpropagation (SBP) repair: the library index and reach set, individual
repair moves, and the gene structure of written-back repairs.
=#

using Test
using Random

const SBPR = GeneExpressionProgramming.SBPUtils

# two features, m and s, and the four arithmetic operators plus sqrt
function repair_fixture(; rounds=3, head_len=4, gene_count=2, gene_connections=[:+, :-, :*, :/],
    non_terminals=[:+, :-, :*, :/, :sqrt])
    dims = Dict{Symbol,Vector{Float16}}(
        :x1 => Float16[0, 1, 0, 0, 0, 0, 0],     # m
        :x2 => Float16[0, 0, 1, 0, 0, 0, 0])     # s
    reg = GepRegressor(2; considered_dimensions=dims, entered_non_terminals=non_terminals,
        gene_connections=gene_connections, gene_count=gene_count, head_len=head_len,
        rounds=rounds, max_permutations_lib=5000)
    tb = reg.toolbox_
    dto = reg.token_dto_
    sym(f) = only(k for (k, v) in tb.callbacks if string(v) == f)
    dimdict = dto.tokenLib.physical_dimension_dict[]
    x1 = only(k for (k, v) in dimdict if v == dims[:x1] && !haskey(tb.callbacks, k))
    x2 = only(k for (k, v) in dimdict if v == dims[:x2] && !haskey(tb.callbacks, k))
    return reg, tb, dto, sym, x1, x2
end

d7(v...) = Float16[v..., zeros(Float16, 7 - length(v))...]

@testset "SBP repair engine" begin
    Random.seed!(7)

    @testset "library index" begin
        reg, tb, dto, sym, x1, x2 = repair_fixture()
        ix = dto.index
        # every stored entry is in prefix order and has the dimension it is filed under
        for ((k, len), bucket) in ix.entries, e in bucket
            @test length(e) == len
            @test SBPR.same_dim(SBPR.expression_dimension(e, dto.tokenLib), ix.dims[k])
        end
        # every reach expression has its dimension and the recorded last operator position
        for k in eachindex(ix.reach_dims)
            e = ix.reach_expr[k]
            @test SBPR.same_dim(SBPR.expression_dimension(e, dto.tokenLib), ix.reach_dims[k])
            @test SBPR.last_operator_position(ix, e) == ix.reach_lastop[k]
        end
        # the kd-tree answers with the dimension itself first when it is reachable
        m_per_s = d7(0, 1, -1)
        @test ix.reach_dims[first(SBPR.nearest_reach(ix, m_per_s, 3))] == m_per_s
        @test ix.terminals[d7(0, 1)] == [x1]
    end

    @testset "negative zero finds its key" begin
        reg, tb, dto, sym, x1, x2 = repair_fixture()
        negzero = Float16[-0.0, 1, 0, 0, 0, 0, 0]
        @test !haskey(dto.index.terminals, negzero)       # the raw vector misses
        @test haskey(dto.index.terminals, SBPR.canonical_dim(negzero))
    end

    @testset "requirements pass through unary operators" begin
        # sqrt(x1 * x1) has dimension m; m^0.5 s^0.5 needs the operand to become m s,
        # which one terminal swap under the product gives
        reg, tb, dto, sym, x1, x2 = repair_fixture()
        expr = Int8[sym("sqrt"), sym("*"), x1, x1]
        tree = create_compute_tree(expr, dto)
        target = d7(0, 0.5, 0.5)
        @test propagate_necessary_changes!(tree, target; cycles=1)
        @test calculate_vector_dimension!(tree) == target
        repaired = flatten_dependents(tree)
        @test repaired[1] == sym("sqrt")              # the unary operator was kept
        @test length(repaired) == 4                   # ... and kept its length
    end

    @testset "a terminal swap under a product succeeds at once" begin
        # x1 * x1 -> m s: one operand must become s
        reg, tb, dto, sym, x1, x2 = repair_fixture()
        tree = create_compute_tree(Int8[sym("*"), x1, x1], dto)
        @test propagate_necessary_changes!(tree, d7(0, 1, 1); cycles=1)
        @test sort(flatten_dependents(tree)[2:3]) == sort([x1, x2])
    end

    @testset "compositions reach past the library" begin
        # a one-round library stops at two symbols, so m s (three) is not in it; the reach
        # set composes it from two library entries
        reg, tb, dto, sym, x1, x2 = repair_fixture(; rounds=1)
        ix = dto.index
        target = d7(0, 1, 1)
        @test !haskey(ix.id, target)
        @test haskey(ix.reach_id, target)
        expr = SBPR.random_expression(ix, target; max_len=9)
        @test SBPR.same_dim(SBPR.expression_dimension(expr, dto.tokenLib), target)
        # an exact seed of that dimension can be sampled all the same
        seed = sample_lib_expression(target, dto; max_len=9, exact_only=true)
        @test seed !== nothing
        @test is_dimensionally_homogeneous(seed, target, dto)
    end

    @testset "targets split across genes" begin
        # a head of one holds at most `op t t`, two terminals; m^2 s^2 needs four, so both
        # genes under a * connector, which the repair has to put in place of the +
        reg, tb, dto, sym, x1, x2 = repair_fixture(; head_len=1, gene_count=2,
            gene_connections=[:+, :*])
        target = d7(0, 2, 2)
        genes = Int8[sym("+"), x1, x2, x2, x1, x2, x2]
        c = Chromosome(copy(genes), tb, true)
        @test !is_dimensionally_homogeneous(c.expression_raw, target, dto)
        dist, ok = correct_genes!(c.genes, tb.gen_start_indices, c.expression_raw, target, dto;
            cycles=5, head_len=tb.head_len, gene_len=2 * tb.head_len + 1,
            connectors=tb.gene_connections)
        @test ok
        compile_expression!(c; force_compile=true)
        @test is_dimensionally_homogeneous(c.expression_raw, target, dto)
        @test c.genes[1] == sym("*")
    end

    @testset "written-back genes stay ordinary genes" begin
        reg, tb, dto, sym, x1, x2 = repair_fixture(; head_len=4, gene_count=3)
        gene_len = 2 * tb.head_len + 1
        target = d7(0, 2, -1)
        pop = generate_population(300, tb)
        attempted = 0
        repaired = 0
        for c in pop
            compile_expression!(c; force_compile=true)
            is_dimensionally_homogeneous(c.expression_raw, target, dto) && continue
            attempted += 1
            before = copy(c.genes)
            _, ok = correct_genes!(c.genes, tb.gen_start_indices, c.expression_raw, target, dto;
                cycles=5, head_len=tb.head_len, gene_len=gene_len, connectors=tb.gene_connections)
            if !ok
                @test c.genes == before                 # a failed repair leaves no trace
                continue
            end
            repaired += 1
            compile_expression!(c; force_compile=true)
            @test is_dimensionally_homogeneous(c.expression_raw, target, dto)
            for part in GeneExpressionProgramming.GepEntities._karva_raw(c; split=true)[2:end]
                @test length(part) <= gene_len
                @test SBPR.last_operator_position(dto.index, collect(part)) <= tb.head_len
            end
            @test all(c.genes[k] in tb.gene_connections for k in 1:tb.gene_count-1)
        end
        @test attempted > 100
        @test repaired >= 0.9 * attempted
    end
end
