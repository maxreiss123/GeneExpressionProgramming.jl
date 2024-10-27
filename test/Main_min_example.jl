include("../src/VGeneExpressionProgramming.jl")

using .VGeneExpressionProgramming
using DynamicExpressions
using OrderedCollections
using BenchmarkTools
using Random

Random.seed!(1)
#Example Call
#Define utilized syms as Ordered Dict: Symbol:Arity
#utilized_syms = OrderedDict{String,Int8}(1"+" => 2, 2"*" => 2, 3"-" => 2, 4"/" => 2, 5"exp" => 1, 7"x_0" => 0, "2" => 0, "0"=> 0, "x_1" => 0)
utilized_syms = OrderedDict{Int8,Int8}(1 => 2, 2 => 2, 3 => 2, 4 => 2, 5 => 1,6 => 0, 7 => 0, 8 => 0, 9 => 0)
#Create connection between genes 
connection_syms = Int8[1, 2]

#Define all the elements for the dynamic.jl
operators =  OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[exp])

callbacks = Dict{Int8,Function}(
        3 => (-),
        4 => (/),
        2 => (*),
        1 => (+),
        5 => (exp)
)
nodes = OrderedDict{Int8,Any}(
    6 => Node{Float64}(feature=1),
    7 => Node{Float64}(feature=2),
    8 => 2,
    9 => 0
)

gep_params = Dict{String, AbstractFloat}(
    "one_point_cross_over_prob" => 0.6,
    "two_point_cross_over_prob" => 0.5,
    "mutation_prob" => 1,
    "mutation_rate" => 0.05,
    "dominant_fusion_prob" => 0.1,
    "dominant_fusion_rate" => 0.2,
    "rezessiv_fusion_prob" => 0.1,
    "rezessiv_fusion_rate" => 0.2,
    "fusion_prob" => 0.0,
    "fusion_rate" => 0.0,
    "inversion_prob" => 0.1
)

#Generate some data
x_data = randn(Float64, 2, 1000)
y_data = @. x_data[1,:] * x_data[1,:] + x_data[1,:] * x_data[2,:] - 2 * x_data[2,:] * x_data[2,:]

#call the function -> return value yields the best:

 best,history =runGep(1000, 1000,4,10,utilized_syms,operators, callbacks, nodes, x_data,y_data, connection_syms, gep_params;
    loss_fun_str="mse", opt_method_const=:cg, hof=1)
@show string(best[1].fitness)
@show string(best[1].compiled_function)
