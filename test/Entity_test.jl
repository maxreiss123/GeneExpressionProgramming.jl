using Test
using OrderedCollections
using Random
using Random123
operators = (binary=[+, -, *, /], unary=[sqrt])

# two genes of head length 3, joined by +, over +, *, sqrt, the feature x1 and the
# constant 1.0
function create_test_toolbox()
    

    symbols = OrderedDict{Int8,Int8}(
        1 => 2,  # +
        2 => 2,  # *
        3 => 1,  # sqrt
        4 => 0,  # feature x1
        5 => 0   # constant 1.0
    )
    
    callbacks = Dict{Int8,Function}(
        1 => +,
        2 => *,
        3 => sqrt
    )
    
    nodes = OrderedDict{Int8,Any}(
        4 => InputSelector(1, "x1"),
        5 => 1.0
    )
    
    gep_probs = Dict{String,AbstractFloat}(
        "mutation_prob" => 0.2,
        "mutation_rate" => 0.1,
        "inversion_prob" => 0.1,
        "one_point_cross_over_prob" => 0.3,
        "two_point_cross_over_prob" => 0.3,
        "dominant_fusion_prob" => 0.2,
        "rezessiv_fusion_prob" => 0.2,
        "fusion_prob" => 0.2,
        "fusion_rate" => 0.1,
        "rezessiv_fusion_rate" => 0.1
    )
    
    return Toolbox(2, 3, symbols, Int8[1], callbacks, nodes, gep_probs; master_rng=Threefry4x(UInt64, (UInt64(0), UInt64(0), UInt64(0), UInt64(0))))
end

@testset "SymbolicEntities Tests" begin
    @testset "Basic Setup" begin
        toolbox = create_test_toolbox()
        
        @test toolbox.gene_count == 2
        @test toolbox.head_len == 3
        @test length(toolbox.headsyms) == 2+1+2  # every symbol may appear in the head
        @test length(toolbox.tailsyms) == 2  # x1 and 1.0
    end

    @testset "Chromosome Creation" begin
        toolbox = create_test_toolbox()
        Random.seed!(42)
        chromosome = generate_chromosome(toolbox)
        
        @test chromosome isa Chromosome
        @test length(chromosome.genes) == (toolbox.gene_count - 1 + 
            toolbox.gene_count * (2 * toolbox.head_len + 1))
        @test chromosome.compiled == true
        @test chromosome.dimension_homogene == false
        @test isnan(chromosome.fitness[1])
    end

    @testset "Function Compilation" begin
        toolbox = create_test_toolbox()
        Random.seed!(42)
        chromosome = generate_chromosome(toolbox)
        compile_expression!(chromosome, force_compile=true)
        
        @test chromosome.compiled == true
        @test !isempty(chromosome.expression_raw)
        
        # the compiled chromosome evaluates on data (one feature, one sample)
        if !isempty(chromosome.expression_raw)
            result = chromosome(reshape([1.0], 1, 1))
            @test typeof(result[1]) <: Real
        end
    end
    
    @testset "Karva strings" begin
        # `_karva_raw` against the rule it implements, written out: per gene, the active
        # part ends at the first zero of the cumulative sum of the arities, all but the
        # first reduced by one (the open argument slots); a gene without one is taken whole
        function karva_reference(c)
            tb = c.toolbox
            gene_len = 2 * tb.head_len + 1
            parts = [c.genes[1:tb.gene_count-1]]
            for g in 1:tb.gene_count
                start = tb.gene_count + (g - 1) * gene_len
                gene = c.genes[start:start+gene_len-1]
                slots = [Int(tb.arrity_by_id[x]) for x in gene]
                slots[2:end] .-= 1
                k = findfirst(==(0), cumsum(slots))
                push!(parts, gene[1:something(k, gene_len)])
            end
            return parts
        end
        Random.seed!(7)
        for gene_count in 1:4, head_len in (1, 3, 6)
            tb = GepRegressor(2; entered_non_terminals=[:+, :-, :*, :sqrt],
                gene_count=gene_count, head_len=head_len).toolbox_
            for c in generate_population(50, tb)
                ref = karva_reference(c)
                @test GeneExpressionProgramming.GepEntities._karva_raw(c) == reduce(vcat, ref)
                @test collect.(GeneExpressionProgramming.GepEntities._karva_raw(c; split=true)) == ref
                @test c.expression_raw == reduce(vcat, ref)
            end
        end
        # a gene cut short of its full length is an error, not a read past the end
        c = generate_chromosome(create_test_toolbox())
        resize!(c.genes, length(c.genes) - 1)
        @test_throws BoundsError GeneExpressionProgramming.GepEntities._karva_raw(c)
    end

    @testset "Population Generation" begin
        toolbox = create_test_toolbox()
        population_size = 10
        population = generate_population(population_size, toolbox)
        
        @test length(population) == population_size
        @test all(x -> x isa Chromosome, population)
        @test length(unique([p.genes for p in population])) == population_size
    end
end
# `compiled_function` was the field that held a chromosome's compiled expression; scripts
# written then print it or call it on data, and a population logger reads it every epoch
@testset "compiled_function, as older code reads it" begin
    Random.seed!(4)
    x = randn(2, 25)
    y = x[1, :] .* x[2, :]
    reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=2, head_len=3)
    c = generate_population(1, reg.toolbox_)[1]
    m = c.compiled_function
    @test string(m) == equation_string(c)
    @test isequal(m(x), c(x))
    @test isequal(m(permutedims(x)', reg.operators_), c(x))    # samples as rows, transposed
    @test :compiled_function in propertynames(c)
    @test string(Chromosome(copy(c.genes), reg.toolbox_, false).compiled_function) ==
          "(not compiled)"
    # the plain fields are still inferred exactly
    @test Base.return_types(ch -> ch.genes, (Chromosome,)) == Any[Vector{Int8}]
    @test Base.return_types(ch -> ch.compiled, (Chromosome,)) == Any[Bool]

    ctxs = thread_contexts(reg.toolbox_, x)
    function loss(elem, validate)
        (isnan(sum(elem.fitness)) || validate) || return
        p = elem(ctxs[Threads.threadid()])
        elem.fitness = p isa AbstractVector ? (sum(abs2, p .- y) / length(y),) : (Inf,)
    end
    io = IOBuffer()
    logger(population, epoch, _) =
        foreach(ch -> println(io, epoch, "; ", ch.compiled_function, "; ", ch.fitness), population)
    fit!(reg, 3, 60, loss; file_logger_callback=logger)
    @test count(==('\n'), String(take!(io))) == 3 * 60
end
