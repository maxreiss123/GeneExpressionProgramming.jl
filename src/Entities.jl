module SymbolicEntities

using OrderedCollections

export AbstractSymbol, FunctionalSymbol, BasicSymbol, SymbolConfig

abstract type AbstractSymbol end

struct BasicSymbol <: AbstractSymbol
    representation::Union{String,Real}
    unit::Vector{Float16}
    index::Int
    arity::Int8
    feature::Bool
end

# Concrete type for functional symbols
struct FunctionalSymbol <: AbstractSymbol
    representation::String
    unit::Vector{Float16}
    index::Int
    arity::Int8
    arithmetic_operation::Function
    forward_function::Union{Function, Nothing}
    reverse_function::Union{Function, Nothing}
end


struct SymbolConfig
    basic_symbols::OrderedDict{Int8, BasicSymbol}
    constant_symbols::OrderedDict{Int8, BasicSymbol}
    functional_symbols::OrderedDict{Int8, FunctionalSymbol}
    callbacks::Dict{Int8,Function}
    operators_djl::Any
    nodes_djl::OrderedDict{Int8, Any}
    symbol_arity_mapping::OrderedDict{Int8, Int8}
    physical_operation_dict::Union{OrderedDict{Int8, Function},Nothing}
    physical_dimension_dict::OrderedDict{Int8, Vector{Float16}}
    features_idx::Vector{Int8}
    functions_idx::Vector{Int8}
    constants_idx::Vector{Int8}
    point_operations_idx::Vector{Int8}
    inverse_operations::Union{Dict{Int8, Function},Nothing}
end



end