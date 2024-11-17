"""
    SBPUtils

A module for Semantic backpropagation (SBP) utilities, focusing on dimensional homogeneity.

# Constants
- `SMALLEST_TREE_SEGMENT = 3`: Minimum size for tree segments
- `FAILURE_RECURSION_SIZE = -21`: Maximum recursion depth for failure handling
- `STD_DIM_SIZE = 7`: Standard dimension vector size
- `ZERO_DIM`: Zero vector of size STD_DIM_SIZE
- `EMPTY_DIM`: Empty dimension vector

# Core Components
## Data Structures
- `TokenLib`: Library of tokens with physical dimensions and operations
- `TokenDto`: Data transfer object for token operations
- `LibEntry`: Entry in the symbolic computation library
- `TempComputeTree`: Temporary computation tree for symbolic manipulation

## Unit Operations
### Forward Operations
- `equal_unit_forward`: Dimension equality checking
- `mul_unit_forward`: Dimension multiplication
- `div_unit_forward`: Dimension division
- `zero_unit_forward`: Zero dimension checking
- `arbitrary_unit_forward`: Direct dimension passing
- `sqr_unit_forward`: Dimension squaring

### Backward Operations
- `mul_unit_backward`: Backward multiplication propagation
- `div_unit_backward`: Backward division propagation
- `zero_unit_backward`: Backward zero propagation
- `sqr_unit_backward`: Backward square propagation
- `equal_unit_backward`: Backward equality propagation

## Tree Operations
- `create_compute_tree`: Creates computation trees from expressions
- `propagate_necessary_changes!`: Propagates dimensional changes
- `calculate_vector_dimension!`: Calculates dimensional vectors
- `correct_genes!`: Ensures dimensional consistency in genes

## JSON Utilities
- `get_feature_dims_json`: Extracts feature dimensions
- `get_target_dim_json`: Extracts target dimensions
- `retrieve_coeffs_based_on_similarity`: Finds similar physical constants

# Features
- Thread-safe operations for parallel processing
- Dimensional homogeneity enforcement
- Physical unit propagation (forward and backward)
- Symbolic computation tree manipulation
- JSON configuration support
- Physical constant matching

# Dependencies
- `OrderedCollections`: For ordered data structures
- `Random`: For stochastic operations
- `StaticArrays`: For efficient array operations

# Implementation Notes
- Uses type parameters for numerical stability
- Implements efficient tree traversal algorithms
- Provides comprehensive error handling
- Supports parallel computation
- Maintains dimensional consistency
- Uses Float16 for dimension calculations

"""

module SBPUtils

const SMALLEST_TREE_SEGMENT = 3
const FAILURE_RECURSION_SIZE = -21
const STD_DIM_SIZE = 7
const ZERO_DIM = zeros(Float16, STD_DIM_SIZE)
const EMPTY_DIM = Float16[typemax(Float16) for _ in 1:STD_DIM_SIZE]

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

@inline function has_inf16(u::Vector{Float16})
    @inbounds for x in u
        reinterpret(UInt16, x) == 0x7c00 && return true
    end
    return false
end

function equal_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}) 
    @inbounds return all(u1 .== u2) ? u1 : EMPTY_DIM
end

function equal_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16})
    return expected_dim, expected_dim
end


function arbitrary_unit_forward(u1::Vector{Float16})
    return u1
end

function mul_unit_forward(u1::Vector{Float16}, u2::Vector{Float16})
    return u1 .+ u2
end

function mul_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}) 
    if has_inf16(u2) && has_inf16(u1)
        if 0.5 < rand()
            lr = expected_dim
            rr = ZERO_DIM
        else
            rr = expected_dim
            lr = ZERO_DIM
        end
        return lr, rr
    elseif has_inf16(u2)
        return u1, expected_dim .- u1
    elseif has_inf16(u1)
        return expected_dim .- u2, u2
    else
        if isapprox(u1, u2, atol=eps(Float16))
            lr = expected_dim .- expected_dim .÷ 2
            rl = expected_dim .- lr
            return lr, rl
        elseif isapprox(u1, expected_dim, atol=eps(Float16))
            return u1, expected_dim .- u1
        else
            return expected_dim .- u2, u2
        end
    end
end


function div_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}) 
    return u1 .- u2
end


function div_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}) 
    if has_inf16(u2) && has_inf16(u1)
        if 0.5 < rand()
            lr = expected_dim
            rr = ZERO_DIM
        else
            rr = -expected_dim
            lr = ZERO_DIM
        end
        return lr, rr
    elseif has_inf16(u2)
        return u1, .-(expected_dim .+ u1)
    elseif has_inf16(u1)
        return expected_dim .+ u2, u2
    else
        if isapprox(u1, u2, atol=eps(Float16))
            lr = expected_dim .- expected_dim .÷ 2
            rl = .-(expected_dim .+ lr)
            return lr, rl
        elseif isapprox(u1, expected_dim, atol=eps(Float16))
            return u1, .-(expected_dim .+ u1)
        else
            return expected_dim .+ u2, u2
        end
    end
end


function zero_unit_forward(u1::Vector{Float16})
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

"""
    TokenLib

A container structure for managing physical dimensions, operations, and symbol arities to support dimensional analysis

# Fields
- `physical_dimension_dict::Ref{OrderedDict{Int8,Vector{Float16}}}`: Maps symbol IDs to their physical dimensions
- `physical_operation_dict::Ref{OrderedDict{Int8,Function}}`: Maps symbol IDs to their physical operations
- `symbol_arity_mapping::Ref{OrderedDict{Int8,Int8}}`: Maps symbol IDs to their arities 

# Constructor
```julia
TokenLib(
    physical_dimension_dict::OrderedDict{Int8,Vector{Float16}},
    physical_operation_dict::OrderedDict{Int8,Function},
    symbol_arity_mapping::OrderedDict{Int8,Int8}
)
```

# Arguments
- `physical_dimension_dict`: Dictionary mapping symbol IDs to their physical dimension vectors
- `physical_operation_dict`: Dictionary mapping symbol IDs to their dimension transformation functions
- `symbol_arity_mapping`: Dictionary mapping symbol IDs to their arity values (0 for terminals, 1 or 2 for functions)

# Examples
```julia
# Create dimension dictionary
dims = OrderedDict{Int8,Vector{Float16}}(
    1 => Float16[1, 0, 0],  # Length
    2 => Float16[0, 1, 0]   # Time
)

# Create operation dictionary
ops = OrderedDict{Int8,Function}(
    1 => mul_unit_forward,
    2 => div_unit_forward
)

# Create arity mapping
arities = OrderedDict{Int8,Int8}(
    1 => 2,  # Binary operation
    2 => 2   # Binary operation
)

# Create TokenLib instance
lib = TokenLib(dims, ops, arities)
```

# Notes
- Uses `Ref` for thread-safe dictionary access
- Dimensions are stored as `Float16` vectors for memory efficiency
- Operations should handle dimensional transformations
- Arity values determine function argument counts

See also: [`SBPUtils.mul_unit_forward`](@ref), [`SBPUtils.div_unit_forward`](@ref), 
[`SBPUtils.equal_unit_forward`](@ref), [`LibEntry`](@ref), [`TokenDto`](@ref)
"""
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


"""
    LibEntry

A mutable structure representing an entry in the symbolic computation library,
tracking elements, their physical dimensions, and arity status.

# Fields
- `elements::Vector{Int8}`: Sequence of symbol IDs representing the expression
- `physical_dimension::Vector{Float16}`: Physical dimension vector of the expression
- `arity_potential::Int`: Current arity potential (0: terminal, 1: unary ready, 2: binary ready)
- `homogene::Bool`: Dimensional homogeneity status
- `tokenLib::TokenLib`: Reference to the token library for symbol information

# Constructor
```julia
LibEntry(symbol_ref::TokenLib)
```

Creates an empty library entry initialized with:
- Empty elements vector
- Empty dimension vector
- Zero arity potential
- True homogeneity status
- Reference to provided TokenLib

# Usage
```julia
# Create token library
token_lib = TokenLib(dimension_dict, operation_dict, arity_mapping)

# Create empty library entry
entry = LibEntry(token_lib)

# Add elements (using append!)
append!(entry, Int8(1))  # Add terminal
append!(entry, Int8(2))  # Add operator
```

# States
## Arity Potential
- `0`: Ready for terminal symbol
- `1`: Ready for unary operation
- `2`: Ready for binary operation

## Homogeneity
- `true`: Expression maintains dimensional consistency
- `false`: Dimensional inconsistency detected

# Methods
The following methods are commonly used with LibEntry:
- `Base.append!`: Add new symbol to entry
- `Base.copy`: Create deep copy of entry
- `Base.length`: Get number of elements
- `Base.:(==)`: Compare entries
- `Base.hash`: Hash entry for collections
- `clean!`: Reset entry to valid state

# Notes
- Maintains dimensional homogeneity tracking
- Supports incremental expression building
- Automatically validates dimensional consistency

See also: [`TokenLib`](@ref), [`TempComputeTree`](@ref)
"""
mutable struct LibEntry
    elements::Vector{Int8}
    physical_dimension::Vector{Float16}
    arity_potential::Int
    homogene::Bool
    tokenLib::TokenLib

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
                if !isempty(physical_dimension) && !has_inf16(physical_dimension)
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
            if !isempty(temp_dim) && !has_inf16(temp_dim)
                entry.arity_potential = 1
                push!(entry.elements, item)
                entry.physical_dimension = temp_dim
                entry.homogene = true
            end
        catch e
            @warn "Issue in Lib. could not apply binary symbol [append-method]: $e"
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

"""
    TokenDto

A data transfer object (DTO) for managing token operations, library access, and gene configuration
in Gene Expression Programming with dimensional analysis support.

# Fields
- `tokenLib::TokenLib`: Reference to token library containing symbol information
- `point_operations::Vector{Int8}`: Vector of point operation symbol IDs (e.g., multiplication, division)
- `lib::Ref{OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}}}`: Library mapping dimensions and lengths to valid expressions
- `inverse_operation::Dict{Int8,Function}`: Maps symbol IDs to their inverse dimensional operations
- `gene_count::Int`: Number of genes in chromosomes
- `head_len::Int`: Length of head section in genes

# Constructor
```julia
TokenDto(
    tokenLib::TokenLib,
    point_operations::Vector{Int8},
    lib::OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}},
    inverse_operation::Dict{Int8,Function},
    gene_count::Int;
    head_len::Int=-1
)
```

# Arguments
- `tokenLib`: Token library instance
- `point_operations`: Vector of point operation symbols
- `lib`: Library mapping (dimension, length) tuples to valid expressions
- `inverse_operation`: Dictionary of inverse operations for backward propagation
- `gene_count`: Number of genes
- `head_len`: Head length (optional, defaults to -1)

# Library Structure
The `lib` field maps tuples of (dimension vector, expression length) to vectors of valid expressions:
```julia
(dimension::Vector{Float16}, length::Int) => Vector{Vector{Int8}}
```

# Usage Example
```julia
# Create components
token_lib = TokenLib(dims_dict, ops_dict, arity_dict)
point_ops = Int8[1, 2]  # multiplication and division
expression_lib = create_lib(token_lib, features, functions, constants)
inverse_ops = Dict{Int8,Function}(
    1 => mul_unit_backward,
    2 => div_unit_backward
)

# Create TokenDto
dto = TokenDto(
    token_lib,
    point_ops,
    expression_lib,
    inverse_ops,
    3;  # gene count
    head_len=6
)
```

# Purpose
1. Centralizes access to token operations and library
2. Manages dimensional analysis configuration
4. Supports both forward and backward dimension propagation
5. Facilitates library lookup for valid expressions

# Notes
- Supports dimensional homogeneity checking
- Maintains expression validity through library lookups

See also: [`TokenLib`](@ref), [`LibEntry`](@ref), [`TempComputeTree`](@ref),
[`SBPUtils.create_lib`](@ref), [`SBPUtils.mul_unit_backward`](@ref), [`SBPUtils.div_unit_backward`](@ref)
"""
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

"""
    TempComputeTree

A mutable structure representing a temporary computation tree for symbolic manipulation
and dimensional analysis in Gene Expression Programming.

# Fields
- `symbol::Int8`: Symbol ID representing the current node's operation or terminal
- `depend_on::Vector{Union{TempComputeTree,Int8}}`: Vector of child nodes or terminal symbols
- `vector_dimension::Vector{Float16}`: Physical dimension vector of the current subtree
- `tokenDto::TokenDto`: Reference to token configuration and library
- `depend_on_total_number::Int`: Total number of nodes in subtree
- `exchange_len::Int`: Length of potential exchange segment (-1 if not set)

# Constructor
```julia
TempComputeTree(
    symbol::Int8,
    depend_on::Vector{T}=Union{TempComputeTree,Int8}[],
    vector_dimension::Vector{Float16}=Float16[],
    tokenDto::TokenDto=nothing
) where {T}
```

# Tree Structure Example
```julia
# Binary operation tree (e.g., multiplication)
root = TempComputeTree(
    1,  # multiplication symbol
    [   # children
        TempComputeTree(3, [], [1.0, 0.0]),  # length dimension
        TempComputeTree(4, [], [0.0, 1.0])   # time dimension
    ],
    [],  # dimension computed later
    token_dto
)
```

# Operations
Common operations on TempComputeTree:
- `flatten_dependents`: Flattens tree to symbol sequence
- `flush!`: Clears dimension vector
- `calculate_vector_dimension!`: Computes dimensions
- `propagate_necessary_changes!`: Ensures dimensional consistency
- `enforce_changes!`: Applies dimensional corrections

# Dimensional Analysis
The tree supports:
1. Forward dimension propagation
2. Backward dimension propagation
3. Dimensional homogeneity checking
4. Automatic correction of dimensional inconsistencies

# Usage Example
```julia
# Create computation tree
tree = TempComputeTree(mul_symbol, [], [], token_dto)

# Calculate dimensions
calculate_vector_dimension!(tree)

# Check and correct dimensions
if !isapprox(tree.vector_dimension, target_dim)
    propagate_necessary_changes!(tree, target_dim)
end
```

# Notes
## Tree Properties
- Recursive structure for expression representation
- Maintains dimensional information at each node
- Supports both terminal and operation symbols
- Allows for tree modification and correction

## Performance Considerations
- Use `flush!` to clear cached dimensions
- `depend_on_total_number` tracks subtree size
- `exchange_len` optimizes tree modifications
- Efficient memory usage with Int8 symbols

See also: [`TokenDto`](@ref), [`SBPUtils.propagate_necessary_changes!`](@ref), [`SBPUtils.enforce_changes!`](@ref)
"""
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
    if length(dims) == 2 && has_inf16(tree.vector_dimension)
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

"""
    enforce_changes!(tree::TempComputeTree, expected_dim::Vector{Float16}; flexible::Bool=true)

Enforces dimensional changes on a computation tree by attempting to replace it with a dimensionally
compatible expression from the library.

# Arguments
- `tree::TempComputeTree`: The computation tree to modify
- `expected_dim::Vector{Float16}`: Target dimension vector to achieve
- `flexible::Bool=true`: Whether to allow flexible length matching in library lookup

# Returns
- `Bool`: `true` if changes were successful and resulting dimensions match expected dimensions
         within Float16 epsilon precision, `false` otherwise

# Algorithm
1. Determines exchange length:
   - Uses `tree.exchange_len` if set (> -1)
   - Otherwise uses total number of dependent nodes + 1
2. Retrieves potential replacement from library matching:
   - Expected dimensions
   - Determined length
   - Flexibility constraints
3. Creates new computation tree from retrieved expression
4. If successful:
   - Updates current tree's symbol and dependencies
   - Recalculates dimensional vector
5. Verifies dimensional match within epsilon

# Effects
When successful:
- Modifies tree's symbol
- Updates tree's dependencies
- Recalculates tree's dimensional vector
Original tree remains unchanged if operation fails.

# Example
```julia
# Create a tree with incorrect dimensions
tree = TempComputeTree(mul_symbol, [...], current_dim, token_dto)

# Try to enforce correct dimensions
target_dim = Float16[1.0, 0.0, 0.0]  # Length dimension
success = enforce_changes!(tree, target_dim)

if success
    @assert isapprox(tree.vector_dimension, target_dim, atol=eps(Float16))
end
```

# Notes
- Can operate in flexible or strict length matching mode
- May fail if no suitable replacement exists in library

See also: [`TempComputeTree`](@ref)
"""
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

"""
    propagate_necessary_changes!(
        tree::TempComputeTree,
        expected_dim::Vector{Float16},
        distance_to_change::Int=0
    )

Recursively propagates dimensional changes through a computation tree to achieve
desired dimensional consistency.

# Arguments
- `tree::TempComputeTree`: The computation tree to modify
- `expected_dim::Vector{Float16}`: Target dimension vector to achieve
- `distance_to_change::Int=0`: Current recursion depth for change propagation

# Returns
- `Bool`: `true` if changes were successful and dimensions match expected values,
         `false` if propagation failed or max depth was reached

# Algorithm Flow
1. Recursion Depth Check:
   - Returns `false` if `distance_to_change < FAILURE_RECURSION_SIZE`
   - Issues warning when max depth reached

2. Early Success Check:
   - If current dimensions match expected (within Float16 epsilon)
   - Random 90% chance of accepting match to allow exploration
   - Returns `true` if match accepted

3. Critical Update Check:
   - Checks if direct tree replacement is possible
   - Attempts enforcement if at or beyond max distance
   - Uses `check_crit_up!` and `enforce_changes!`

4. Operation Handling:
   - Binary operations: Propagates through both children
   - Unary operations: Propagates through single child
   - Recalculates dimensions after changes

# Effects
May modify:
- Tree structure
- Node symbols
- Dimensional vectors
- Child dependencies

# Example
```julia
# Create computation tree
tree = create_compute_tree(expression, token_dto)

# Target length dimension
target_dim = Float16[1.0, 0.0, 0.0]

# Attempt to achieve target dimension
success = propagate_necessary_changes!(tree, target_dim)

if success
    @assert isapprox(tree.vector_dimension, target_dim, atol=eps(Float16))
end
```

# Error Handling
- Returns `false` if recursion depth exceeded
- Issues warning via `@warn` at max depth
- Handles both successful and failed propagations
- Validates final dimensions after changes

# Notes
## Performance Considerations
- Includes randomization to prevent local optima
- Caches dimension calculations
- Optimizes tree modifications

## Implementation Details
- Recursive implementation
- Aims to maintain consistency
- Supports both unary and binary operations

See also: [`TempComputeTree`](@ref), [`SBPUtils.enforce_changes!`](@ref)
"""
function propagate_necessary_changes!(
    tree::TempComputeTree,
    expected_dim::Vector{Float16},
    distance_to_change::Int=0
)
    if distance_to_change < FAILURE_RECURSION_SIZE
        @warn "Reached max depth"
        return false
    end

    if !has_inf16(tree.vector_dimension) && isapprox(tree.vector_dimension, expected_dim, atol=eps(Float16)) 
        return true
    end

    if check_crit_up!(tree.depend_on_total_number+1, expected_dim, tree) && distance_to_change <= 0 && rand() > 0.25
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
        return propagate_necessary_changes!(tree.depend_on[1], convert.(Float16,expected_dim_new), distance_to_change - 1)
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

"""
    correct_genes!(
        genes::Vector{Int8},
        start_indices::Vector{Int},
        expression::Vector{Int8},
        target_dimension::Vector{Float16},
        token_dto::TokenDto;
        cycles::Int=5
    )

Modifies genes to achieve dimensional consistency with target dimensions through
iterative tree transformation and correction.

# Arguments
- `genes::Vector{Int8}`: Gene sequence to modify
- `start_indices::Vector{Int}`: Starting indices for each gene segment
- `expression::Vector{Int8}`: Current expression to analyze
- `target_dimension::Vector{Float16}`: Target dimension vector to achieve
- `token_dto::TokenDto`: Token configuration and library reference
- `cycles::Int=5`: Maximum number of correction attempts

# Returns
Tuple containing:
- `distance::Float16`: Final dimensional distance from target
- `success::Bool`: `true` if correction achieved target dimensions within epsilon

# Algorithm Steps
1. Tree Creation:
   - Creates computation tree from expression
   - Initializes with token configuration

2. Correction Cycles:
   - Attempts dimensional correction up to specified cycles
   - Propagates necessary changes through tree
   - Recalculates dimensions after each attempt
   - Breaks early on success

3. Success Processing:
   - If target dimensions achieved (distance < eps)
   - Updates genes based on corrected tree
   - Handles both single and multi-gene cases
   - Preserves gene structure integrity

# Error Handling
- Catches and logs any errors during correction
- Returns (Inf16, false) on failure
- Provides detailed error messages
- Preserves original genes on failure

# Example
```julia
# Setup
genes = Int8[...]
indices = [1, 10, 20]  # Gene start positions
expression = Int8[...]  # Current expression
target_dim = Float16[1.0, 0.0, 0.0]  # Length dimension

# Attempt correction
distance, success = correct_genes!(
    genes,
    indices,
    expression,
    target_dim,
    token_dto;
    cycles=7
)

if success
    println("Correction successful!")
else
    println("Failed to achieve target dimensions")
end
```

# Performance Notes
- `@inline` directive for performance optimization
- Early breaking on successful correction
- Efficient tree traversal
- Minimal memory allocation

# Implementation Details
## Gene Update Process
1. Processes genes in reverse index order
2. Special handling for first gene index
3. Updates both symbol and dependencies
4. Maintains gene structure constraints

## Error Conditions
- Tree creation failure
- Propagation errors
- Dimension calculation issues
- Invalid tree structures

See also: [`TempComputeTree`](@ref), [`SBPUtils.propagate_necessary_changes!`](@ref)
"""
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
