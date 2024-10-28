module SBPUtils

const SMALLEST_TREE_SEGMENT = 3
const FAILURE_RECURSION_SIZE = -21
const STD_DIM_SIZE = 7
const ZERO_DIM = zeros(Float16, STD_DIM_SIZE)
const EMPTY_DIM = Float16[]

using OrderedCollections
using Random
using StaticArrays

export TokenLib, TokenDto, LibEntry, TempComputeTree
export create_lib, create_compute_tree, propagate_necessary_changes!, calculate_vector_dimension!, flush!, calculate_vector_dimension!, flatten_dependents
export propagate_necessary_changes!, correct_genes!
export equal_unit_forward, mul_unit_forward, div_unit_forward, zero_unit_forward, sqr_unit_backward, sqr_unit_forward, arbitrary_unit_forward
export zero_unit_backward, mul_unit_backward, div_unit_backward, equal_unit_backward  
export get_feature_dims_json, get_target_dim_json, retrieve_coeffs_based_on_similarity
export ZERO_DIM

function equal_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}) 
    if isempty(u1) || isempty(u2)
        return EMPTY_DIM
    end
    @inbounds return all(u1 .== u2) ? u1 : EMPTY_DIM
end

function equal_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16})
    return expected_dim, expected_dim
end


function arbitrary_unit_forward(u1::Vector{Float16})
    return u1
end

function mul_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}) 
    if isempty(u1) || isempty(u2)
        return EMPTY_DIM
    end
    return u1 .+ u2
end

function mul_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}) 
    if isempty(u2) && isempty(u1)
        if 0.5 < rand()
            lr = expected_dim
            rr = ZERO_DIM
        else
            rr = expected_dim
            lr = ZERO_DIM
        end
        return lr, rr
    elseif isempty(u2)
        return u1, expected_dim .- u1
    elseif isempty(u1)
        return expected_dim .- u2, u2
    elseif sum(abs.(expected_dim .- (u1 .- u2))) == 0
        tree.symbol = tree.tokenDto.point_operations[2]
        return EMPTY_DIM, EMPTY_DIM
    else
        if isapprox(u1, u2, atol=eps(T))
            lr = expected_dim .- expected_dim .÷ 2
            rl = expected_dim .- lr
            return lr, rl
        elseif isapprox(u1, expected_dim, atol=eps(T))
            return u1, expected_dim .- u1
        else
            return expected_dim .- u2, u2
        end
    end
end


function div_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}) 
    if isempty(u1) || isempty(u2)
        return EMPTY_DIM
    end
    return u1 .- u2
end


function div_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}) 
    if isempty(u2) && isempty(u1)
        if 0.5 < rand()
            lr = expected_dim
            rr = ZERO_DIM
        else
            rr = -expected_dim
            lr = ZERO_DIM
        end
        return lr, rr
    elseif isempty(u2)
        return u1, .-(expected_dim .+ u1)
    elseif isempty(u1)
        return expected_dim .+ u2, u2
    elseif sum(abs.(expected_dim .- (u1 .+ u2))) == 0
        tree.symbol = tree.tokenDto.point_operations[1]
        return EMPTY_DIM, EMPTY_DIM
    else
        if isapprox(u1, u2, atol=eps(T))
            lr = expected_dim .- expected_dim .÷ 2
            rl = .-(expected_dim .+ lr)
            return lr, rl
        elseif isapprox(u1, expected_dim, atol=eps(T))
            return u1, .-(expected_dim .+ u1)
        else
            return expected_dim .+ u2, u2
        end
    end
end


function zero_unit_forward(u1::Vector{Float16})
    if isempty(u1)
        return EMPTY_DIM
    end
    @inbounds return all(u1 .== 0) ? ZERO_DIM .* u1 : EMPTY_DIM
end

function zero_unit_backward(u1::Vector{Float16}) 
    return ZERO_DIM
end

function sqr_unit_forward(u1::Vector{Float16}) 
    return 2.0 .* u1
end

function sqr_unit_backward(u1::Vector{Float16}) 
    return 0.5 .* u1
end


mutable struct TokenLib
    physical_dimension_dict::Ref{OrderedDict{Int8,Vector{Float16}}}
    physical_operation_dict::Ref{OrderedDict{Int8,Function}}
    symbol_arity_mapping::Ref{OrderedDict{Int8,Int8}}

    function TokenLib(physical_dimension_dict::OrderedDict{Int8,Vector{Float16}},
        physical_operation_dict::OrderedDict{Int8,Function},
        symbol_arity_mapping::OrderedDict{Int8,Int8})
        new(Ref(physical_dimension_dict), Ref(physical_operation_dict), Ref(symbol_arity_mapping))
    end
end

function get_arity(elem::TokenLib, item::Int8)
    return elem.symbol_arity_mapping[][item]
end

function get_physical_dimension(elem::TokenLib, item::Int8)
    return elem.physical_dimension_dict[][item]
end

function get_physical_operation(elem::TokenLib, item::Int8)
    return elem.physical_operation_dict[][item]
end


mutable struct LibEntry
    elements::Vector{Int8}
    physical_dimension::Vector{Float16}
    arity_potential::Int
    homogene::Bool
    tokenLib::TokenLib

    #=
        physical_dimension_dict: Int is related to the corresponding feature dimension
        Correponding operation_dict: provides the correspnding operation for converting the unit 
    =#
    function LibEntry(symbol_ref::TokenLib)
        new(Int8[], Float16[], 0, true, symbol_ref)
    end
end


function Base.:(==)(a::LibEntry, b::LibEntry)
    if length(a.elements) != length(b.elements)
        return false
    end
    @simd for i in eachindex(a.elements)
        @inbounds if a.elements[i] != b.elements[i]
            return false
        end
    end
    return true
end


function Base.hash(entry::LibEntry, h::UInt)
    return hash(entry.elements, h)
end

function Base.length(entry::LibEntry)
    return length(entry.elements)
end

function Base.show(io::IO, entry::LibEntry)
    print(io, "LibEntry(elements=$(entry.elements), physical_dimension=$(entry.physical_dimension), arity_potential=$(entry.arity_potential))")
end


function Base.copy(entry::LibEntry)
    new_entry = LibEntry(entry.tokenLib)
    new_entry.elements = copy(entry.elements)
    new_entry.physical_dimension = copy(entry.physical_dimension)
    new_entry.arity_potential = entry.arity_potential
    new_entry.homogene = entry.homogene
    return new_entry
end

#Hint inline is a directive to copy paste code at a certain position-> mitigate function calls and therefore heap shifting
@inline function Base.append!(entry::LibEntry, item::Int8)
    arity = get_arity(entry.tokenLib, item)

    if entry.arity_potential == 0 && arity == 0
        entry.arity_potential += 1
        push!(entry.elements, item)
        if haskey(entry.tokenLib.physical_dimension_dict[], item)
            entry.physical_dimension = convert(Vector{Float16}, get_physical_dimension(entry.tokenLib, item))
            entry.homogene = true
        end

    elseif entry.arity_potential == 1
        if arity == 1 && sanity_check(entry, item)
            try
                operation = get_physical_operation(entry.tokenLib, item)
                physical_dimension = convert(Vector{Float16}, operation(entry.physical_dimension))
                if !isempty(physical_dimension)
                    push!(entry.elements, item)
                    entry.physical_dimension = physical_dimension
                    entry.homogene = true
                end
            catch e
                @warn "Issue in Lib. could not apply unary symbol [append-method]: $e"
                entry.homogene = false
            end
        elseif arity == 0
            push!(entry.elements, item)
            entry.arity_potential += 1
            entry.homogene = false
        end
    elseif entry.arity_potential == 2 && arity == 2
        try
            operation = get_physical_operation(entry.tokenLib, item)
            last_dim = get_physical_dimension(entry.tokenLib, entry.elements[end])
            temp_dim = convert(Vector{Float16}, operation(last_dim, entry.physical_dimension))
            if !isempty(temp_dim)
                entry.arity_potential = 1
                push!(entry.elements, item)
                entry.physical_dimension = temp_dim
                entry.homogene = true
            end
        catch e
            @warn "Issue in Lib. could not apply binary symbol [append-method]: $e"
            @warn entry 
            @warn get_physical_operation(entry.tokenLib, item)
            entry.homogene = false
        end
    end
end


function clean!(entry::LibEntry)
    if !entry.homogene && !isempty(entry.elements)
        pop!(entry.elements)
        entry.arity_potential = 1
        entry.homogene = true
    end
end


function sanity_check(entry::LibEntry, item::Int8; max_occurence::Int=2)
    if isempty(entry.elements) || entry.elements[end] == item
        return false
    else
        count = 0
        for i in length(entry.elements):-1:1
            if entry.elements[i] == item
                count += 1
            end
            if count >= max_occurence
                return false
            end
        end
    end
    return true
end

function create_lib(tokenLib::TokenLib, features::Vector{Int8},
    functions::Vector{Int8},
    constants::Vector{Int8};
    rounds::Int=25, max_permutations::Int=10000)

    lib = Set{LibEntry}()
    for feature in features
        entry = LibEntry(tokenLib)
        append!(entry, feature)
        push!(lib, entry)
    end

    search_space = vcat(functions, features, constants)
    new_entries_local = [Vector{LibEntry}() for _ in 1:Threads.nthreads()]

    @inbounds for round in 1:rounds
        for entries in new_entries_local
            empty!(entries) #thread storage
        end

        lib_array = collect(lib)
        Threads.@threads for i in eachindex(lib_array)
            entry = lib_array[i]
            local_entries = new_entries_local[Threads.threadid()]
            for item in search_space
                new_entry = copy(entry)
                append!(new_entry, item)
                if !(new_entry in lib) # Check if it's not already in lib
                    push!(local_entries, new_entry)
                end
            end
        end

        new_entries = reduce(vcat, new_entries_local)
        unique!(new_entries)
        if length(new_entries) > max_permutations
            shuffle!(new_entries)
            resize!(new_entries, max_permutations)
        end

        union!(lib, new_entries)

        if isempty(new_entries)
            println("No new entries generated. Stopping early.")
            break
        end
    end

    organized_lib = reorganize_lib(lib)
    return organized_lib
end

function reorganize_lib(old_lib::Set{LibEntry})
    local_libs = [OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}}() for _ in 1:Threads.nthreads()]

    old_lib_array = collect(old_lib)  # Convert Set to Array
    Threads.@threads for i in eachindex(old_lib_array)
        entry = old_lib_array[i]
        clean!(entry)
        key = (entry.physical_dimension, length(entry.elements))
        local_lib = local_libs[Threads.threadid()]

        if haskey(local_lib, key)
            push!(local_lib[key], entry.elements)
        else
            local_lib[key] = [entry.elements]
        end
    end

    merged_lib = OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}}()
    for local_lib in local_libs
        for (key, value) in local_lib
            if haskey(merged_lib, key)
                append!(merged_lib[key], value)
            else
                merged_lib[key] = value
            end
        end
    end
    return merged_lib
end

mutable struct TokenDto
    tokenLib::TokenLib
    point_operations::Vector{Int8}
    lib::Ref{OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}}}
    inverse_operation::Dict{Int8,Function}
    gene_count::Int
    head_len::Int

    function TokenDto(tokenLib, point_operations, lib, inverse_operation, gene_count; head_len=-1)
        new(tokenLib, point_operations, Ref(lib), inverse_operation, gene_count, head_len)
    end

end


mutable struct TempComputeTree
    symbol::Int8
    depend_on::Vector{Union{TempComputeTree,Int8}}
    vector_dimension::Vector{Float16}
    tokenDto::TokenDto
    depend_on_total_number::Int
    exchange_len::Int

    function TempComputeTree(symbol::Int8,
        depend_on::Vector{T}=Union{TempComputeTree,Int8}[],
        vector_dimension::Vector{Float16}=Float16[],
        tokenDto::TokenDto=nothing) where {T}
        new(symbol,
            convert(Vector{Union{TempComputeTree,Int8}}, depend_on),
            vector_dimension,
            tokenDto,
            length(depend_on),
            -1)
    end
end

function flatten_dependents(tree::TempComputeTree)
    ret_val = [tree.symbol]
    for elem in tree.depend_on
        if elem isa TempComputeTree
            append!(ret_val, flatten_dependents(elem))
        else
            push!(ret_val, elem)
        end
    end
    tree.depend_on_total_number = length(ret_val)
    return ret_val
end

function flush!(tree::TempComputeTree)
    tree.vector_dimension = []
end

function calculate_vector_dimension!(tree::TempComputeTree)
    tdto = tree.tokenDto
    tokenLib = tdto.tokenLib
    point_operations = tdto.point_operations

    function_op = tokenLib.physical_operation_dict[][tree.symbol]
    dims = map(elem -> elem isa TempComputeTree ? calculate_vector_dimension!(elem) : get_physical_dimension(tokenLib, elem), tree.depend_on)
    tree.vector_dimension = function_op(dims...)


    #needs to be revised
    if length(dims) == 2 && isempty(tree.vector_dimension)
        tree.symbol = rand(point_operations)
        function_op = tokenLib.physical_operation_dict[][tree.symbol]
        tree.vector_dimension = function_op(dims...)
    end

    return tree.vector_dimension
end


function calculate_contribution(tree::TempComputeTree, expected_dim::Vector{Float16})
    left_dim = tree.depend_on[1] isa TempComputeTree ? tree.depend_on[1].vector_dimension : get_physical_dimension(tree.tokenDto.tokenLib, tree.depend_on[1])
    right_dim = tree.depend_on[2] isa TempComputeTree ? tree.depend_on[2].vector_dimension : get_physical_dimension(tree.tokenDto.tokenLib, tree.depend_on[2])
    return tree.tokenDto.inverse_operation[tree.symbol](left_dim, right_dim, expected_dim)
end

function check_crit_up!(len_rest::Int, expected_dim::Vector{Float16}, tree::TempComputeTree)
    @inbounds for elem in len_rest:-1:SMALLEST_TREE_SEGMENT
        if haskey(tree.tokenDto.lib[], (expected_dim, elem))
            tree.exchange_len=elem
            return true
        end
    end
    return false
end



function enforce_changes!(tree::TempComputeTree, expected_dim::Vector{Float16}, index::Int)
    exchange_symbol = retrieve_exchange_from_lib(tree, expected_dim, 1)
    if isnothing(exchange_symbol) || isempty(exchange_symbol)
        return false
    end
    tree.depend_on[index] = rand(exchange_symbol)
    calculate_vector_dimension!(tree)
    return calculate_distance(tree.vector_dimension, expected_dim) < eps(Float16)

end


function enforce_changes!(tree::TempComputeTree, expected_dim::Vector{Float16}; flexible::Bool=true)
    extraction_len = tree.exchange_len > -1 ? tree.exchange_len : tree.depend_on_total_number+1
        exchange = retrieve_exchange_from_lib(tree,
            expected_dim,
            extraction_len,
            flexible)
        isnothing(exchange) && return false
        exchange = reverse(exchange)
        new_tree = create_compute_tree(exchange, tree.tokenDto)
        if new_tree isa TempComputeTree
            tree.symbol = new_tree.symbol
            tree.depend_on = new_tree.depend_on
            calculate_vector_dimension!(tree)
        end
        return calculate_distance(tree.vector_dimension, expected_dim) < eps(Float16)
end


function retrieve_exchange_from_lib(tree::TempComputeTree,
    dimension_vector::Vector{Float16},
    len_substitution::Int,
    flexible::Bool=false)
    min_len = min(tree.depend_on_total_number, SMALLEST_TREE_SEGMENT)
    possible_matches = find_closest(dimension_vector, len_substitution, tree.tokenDto.lib[];
        flexible=flexible, min_len=min_len)
    if isnothing(possible_matches) || isempty(possible_matches)
        return nothing
    else
        return possible_matches
    end
end

function calculate_distance(k1::Vector{Float16}, k2::Vector{Float16})
    if isempty(k2) || isempty(k1)
        return Inf16
    end
    return sqrt(sum((k1 .- k2) .^ 2))
end


function find_closest(distance::Vector{Float16},
    expression_len::Int,
    lib::OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}};
    flexible::Bool=false,
    min_len::Int=1)

    exact_key = (distance, expression_len)
    haskey(lib, exact_key) && return rand(lib[exact_key])
    matching_length_keys = filter(key -> key[2] == expression_len, collect(keys(lib)))

    if !isempty(matching_length_keys)
        _, closest_key = findmin(key -> calculate_distance(distance, key[1]), matching_length_keys)
        return rand(lib[matching_length_keys[closest_key]])
    end

    if flexible
        all_keys = sort!(collect(keys(lib)), by=key -> abs(key[2] - expression_len))
        for key in all_keys
            if min_len <= key[2] <= expression_len
                sim = calculate_darityistance(distance, key[1])
                sim < eps(Float16) && return rand(lib[key])
            end
        end
    end
    return nothing
end

function propagate_necessary_changes!(
    tree::TempComputeTree,
    expected_dim::Vector{Float16},
    distance_to_change::Int=0
)
    if distance_to_change < FAILURE_RECURSION_SIZE
        @warn "Reached max depth"
        return false
    end

    if !isempty(tree.vector_dimension) && isapprox(tree.vector_dimension, expected_dim, atol=eps(Float16))
        return true
    end

    if check_crit_up!(tree.depend_on_total_number+1, expected_dim, tree) && distance_to_change <= 0
        return enforce_changes!(tree, expected_dim)
    end

    if length(tree.depend_on) == 2
        ret_val = handle_binary_operation(tree, expected_dim, distance_to_change)
    elseif length(tree.depend_on) == 1
        ret_val = handle_unary_operation(tree, expected_dim, distance_to_change)
    end
    calculate_vector_dimension!(tree)
    return ret_val && isapprox(tree.vector_dimension, expected_dim, atol=eps(Float16))
end


function handle_binary_operation(
    tree::TempComputeTree,
    expected_dim::Vector{Float16},
    distance_to_change::Int
)

    residuals = calculate_contribution(tree, expected_dim)
    if isempty(residuals[1]) || isempty(residuals[2])
        return true
    end

    ret_val = true
    for (index, (elem, residual_dimension)) in enumerate(zip(tree.depend_on, residuals))
        if elem isa TempComputeTree
            ret_val &= propagate_necessary_changes!(elem, convert.(Float16, residual_dimension), distance_to_change - 1)
        else
            ret_val &= enforce_changes!(tree, convert.(Float16, residual_dimension), index)
        end
    end
    return ret_val
end

function handle_unary_operation(
    tree::TempComputeTree,
    expected_dim::Vector{Float16},
    distance_to_change::Int
)
    if tree.depend_on[1] isa TempComputeTree && tree.symbol in keys(tree.tokenDto.inverse_operation)
        inverse_op = tree.tokenDto.inverse_operation[tree.symbol]
        expected_dim_new = inverse_op(tree.vector_dimension)
        return propagate_necessary_changes!(tree.depend_on[1], expected_dim_new, distance_to_change - 1)
    else
        return enforce_changes!(tree, expected_dim)
    end
end


function create_compute_tree(expression::Vector{Int8}, tokenDto::TokenDto, initial_state::Bool=false)
    expression_list = reverse(expression)
    stack = Union{TempComputeTree,Int8}[]

    if length(expression) == 1
        return expression[1]
    end

    for (index, symbol) in enumerate(expression_list)
        arity = get_arity(tokenDto.tokenLib, symbol)

        if arity == 1
            if isempty(stack)
                return nothing
            end
            op1 = pop!(stack)
            computeTree = TempComputeTree(symbol, [op1], Float16[], tokenDto)
            push!(stack, computeTree)
        elseif arity == 2
            if length(stack) < 2
                return nothing
            end
            op1 = pop!(stack)
            op2 = pop!(stack)
            computeTree = TempComputeTree(symbol, [op1, op2], Float16[], tokenDto)
            push!(stack, computeTree)
        else
            push!(stack, symbol)
        end
    end


    if isempty(stack)
        return nothing
    end

    root = stack[end]
    calculate_vector_dimension!(root)

    root.depend_on_total_number = length(flatten_dependents(root))
    return root
end

function edit_gene_from_compute_tree!(gene::Vector{Int8}, compute_tree::Union{TempComputeTree,Int8}, start_index::Int)
    traits = compute_tree isa TempComputeTree ? flatten_dependents(compute_tree) : [compute_tree]
    @inbounds for index in eachindex(traits)
        gene[start_index+index-1] = traits[index]
    end
end


@inline function correct_genes!(genes::Vector{Int8}, start_indices::Vector{Int}, expression::Vector{Int8},
    target_dimension::Vector{Float16}, token_dto::TokenDto; cycles::Int=5)
    gene_count = token_dto.gene_count
    tree = create_compute_tree(expression, token_dto, true)
    for _ in 1:cycles
        try
            if propagate_necessary_changes!(tree, target_dimension, gene_count-1)
                break
            end
            calculate_vector_dimension!(tree)
        catch
            tree = nothing 
            return Inf16, false
        end
    end
    distance = calculate_distance(target_dimension, calculate_vector_dimension!(tree))

    temp_tree = tree
    if distance < eps(Float16)
        for (count, index) in enumerate(reverse(start_indices))
            if index != start_indices[1]
                genes[count] = temp_tree.symbol
                edit_gene_from_compute_tree!(genes, temp_tree.depend_on[2], index)
                temp_tree = temp_tree.depend_on[1]
            else
                edit_gene_from_compute_tree!(genes, temp_tree, index)
            end
        end
    end

    return distance, distance < eps(Float16)
end

function get_feature_dims_json(json_data::Dict{String,Any}, features::Vector{String}, case_name::String; dims_identifier::String="dims")
    target_entry = json_data[case_name]
    ret_val = Dict{String,Vector{Float16}}()
    for (index, entry) in enumerate(target_entry[dims_identifier])
        ret_val[features[index]] = convert.(Float16, entry)
    end
    return ret_val
end

function is_close_to_target(target_dim::Vector{Float16}, value_dim::Vector{Float16}, tolerance::Float16=Float16(0.5))
    weighting = [t != 0 ? Float16(0.25) : Float16(2.0) for t in target_dim]
    value = sum(abs.(target_dim .- value_dim) .* weighting)
    return value < tolerance
end


function get_target_dim_json(json_data::Dict{String,Any}, case_name::String; dims_identifier::String="targetdims")
    target_entry = json_data[case_name]
    return convert.(Float16, target_entry[dims_identifier])
end



function retrieve_coeffs_based_on_similarity(target_dim::Vector{Float16},
    physical_constants::Dict{String,Tuple{T,Vector{Float16}}}; tolerance::Float16=Float16(100.0)) where {T<:AbstractFloat}

    ret_val = Dict{String,Vector{Float16}}()
    for (_, (val, dim)) in physical_constants
        if is_close_to_target(target_dim, dim, tolerance)
            ret_val[string(val)] = dim
        end
    end
    return ret_val
end

end

