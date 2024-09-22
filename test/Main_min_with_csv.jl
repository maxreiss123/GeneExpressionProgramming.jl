include("../src/VGEP.jl")
include("../src/Util.jl")


using .VGEP
using .VGEPUtils
using DynamicExpressions
using OrderedCollections
using BenchmarkTools
using CSV
using DataFrames
using Random

Random.seed!(1)
#Example Call
#Define utilized syms as Ordered Dict: Symbol:Arity

#Symbol representation: Here, each symbol belongs either to a terminal or non-terminal with a corresponding arity
utilized_syms = OrderedDict{Int8,Int8}(1 => 2, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6=>2, 7=>1, 8=>0, 9=>0, 10=>0, 11=>0)

#Create a connection between genes here (times, plus)  
connection_syms = Int8[1, 2]

#Define all the elements of the dynamic.jl
operators =  OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[exp])

callbacks = Dict{Int8,Function}(
        3 => (-),
        4 => (/),
        2 => (*),
        1 => (+),
        5 => (max),
        6 => (min),
        7 => (exp)
)
nodes = OrderedDict{Int8,Any}(
    8 => Node{Float64}(feature=1),
    9 => Node{Float64}(feature=2),
    10 => 2,
    11 => 0
)

gep_params = Dict{String, AbstractFloat}(
    "one_point_cross_over_prob" => 0.6,
    "two_point_cross_over_prob" => 0.5,
    "mutation_prob" => 1,
    "mutation_rate" => 0.25,
    "dominant_fusion_prob" => 0.1,
    "dominant_fusion_rate" => 0.2,
    "rezessiv_fusion_prob" => 0.1,
    "rezessiv_fusion_rate" => 0.2,
    "fusion_prob" => 0.1,
    "fusion_rate" => 0.2,
    "inversion_prob" => 0.1
)


# considers only the n-th row for the fitness evaluation
consider = 1

# Data file, here is expected to be a csv, where the columns are in the order x1,x2...xk, y 
data = Matrix(CSV.read("test.csv", DataFrame))

# Get the number of columns
num_cols = size(data, 2)

# Shuffle the data
data = data[shuffle(1:size(data, 1)), :]

# Split the data into train and test sets
split_point = floor(Int, size(data, 1) * 0.75)
data_train = data[1:split_point, :]
data_test = data[(split_point + 1):end, :]

# Set the consideration factor
consider = 1

# Prepare training data
x_data = Float64.(data_train[1:consider:end, 1:(num_cols-1)])
y_data = Float64.(data_train[1:consider:end, num_cols])

# Prepare test data
x_data_test = Float64.(data_test[1:consider:end, 1:(num_cols-1)])
y_data_test = Float64.(data_test[1:consider:end, num_cols])

epochs = 1000
pop_size = 1000
gene_count = 3
head_len = 4

best=run_GEP(epochs, pop_size, gene_count, head_len, utilized_syms,operators, callbacks, nodes, x_data',y_data, connection_syms, gep_params;
    loss_fun_str="mse",x_data_test=x_data_test', y_data_test=y_data_test ,opt_method_const=:cg, hof=1)

#Show the result of the optimization
@show ("Fitness: (loss-fun): ", best[1].fitness)
@show ("R2-score-train: ", best[1].fitness_r2_train)
@show ("R2-score-test: ", best[1].fitness_r2_test)
@show ("Model in (x)-notation: ", string(best[1].compiled_function))

#@show string(best[1].compiled_function)
