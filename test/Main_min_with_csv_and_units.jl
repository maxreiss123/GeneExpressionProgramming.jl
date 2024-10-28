using VGeneExpressionProgramming
using DynamicExpressions
using OrderedCollections
using BenchmarkTools
using CSV
using DataFrames
using Random

Random.seed!(1)
#Example Call
#start with some GEP hyper-params:
epochs = 1000
pop_size = 1000
gene_count = 6
head_len = 5
cycles = 5
#Define utilized syms as Ordered Dict: Symbol:Arity
#Symbol representation: Here, each symbol belongs either to a terminal or non-terminal with a corresponding arity
utilized_syms = OrderedDict{Int8,Int8}(1 => 2, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6=>2, 7=>1, 8=>0, 9=>0, 10=>0, 11=>0)

#Create a connection between genes here (times, plus)  
connection_syms = Int8[1, 2, 3, 4]

#Define all the elements of the dynamic.jl
operators =  OperatorEnum(; binary_operators=[+, -, *, /,min,max], unary_operators=[exp])

callbacks = Dict{Int8,Function}(
        4 => (-),
        2 => (/),
        1 => (*),
        3 => (+),
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

#defining dimensions and things for employing the dims
target_dim = Float16[0, 1, 0, 0, 0, 0, 0] # def. target dim
std_phy_dim = zeros(Float16, 7) # A standard dimension

physical_operation_dict = OrderedDict{Int8,Function}(index => forward_function for (index, forward_function) in callbacks)
physical_dimension_dict = OrderedDict{Int8,Vector{Float16}}(index => std_phy_dim for (index, forward_function) in callbacks)

physical_dimension_dict[8]= Float16[0, 1, 0, 0, 0, 0, 0] #dim x1
physical_dimension_dict[9]= Float16[0, 1, 0, 0, 0, 0, 0] #dim x2

physical_dimension_dict[10]= std_phy_dim #dim for the constants
physical_dimension_dict[11]= std_phy_dim #dim for the constants

features_idx = Int8[8,9]
functions_idx = Int8[index for (index, arity) in utilized_syms if arity>0]
constants_idx = Int8[10,11]    
    
inverse_operations = Dict{Int8,Function}(
	1 => mul_unit_backward,
	2 => div_unit_backward,
	3 => equal_unit_backward,
	4 => equal_unit_backward,
	5 => equal_unit_backward,
	6 => equal_unit_backward,
	7 => zero_unit_backward
)
physical_operation_dict = OrderedDict{Int8, Function}(
	1 => mul_unit_forward,
	2 => div_unit_forward,
	3 => equal_unit_forward,
	4 => equal_unit_forward,
	5 => equal_unit_forward,
	6 => equal_unit_forward,
	7 => zero_unit_forward
)
#after defining the syms lets create the lib
#create a library
token_lib = TokenLib(physical_dimension_dict, #dictionary of all dims 
		     physical_operation_dict, #dictionary of all forward unit ops
                     utilized_syms) # dictionary containing the arity information

lib = create_lib(token_lib, #token_lib from above, marking the template 
	features_idx, #list of indices of the features 
	functions_idx, #list of indices for the functions 
	constants_idx; #list of indices for the constants 
	rounds=head_len-2, max_permutations=100000)
	total_len_lib = sum(length(entry) for entry in values(lib))
	@show ("Lib Entries:" , total_len_lib)

#reference object for the chromosomes later on
token_dto = TokenDto(token_lib, Int8[1,2], lib, inverse_operations, gene_count; head_len=head_len-2)

#define a call back for the correction
function corr_call_back!(genes::Vector{Int8}, start_indices::Vector{Int}, expression::Vector{Int8})
	return correct_genes!(genes, start_indices, expression, 
	convert.(Float16,target_dim), token_dto; cycles=cycles) 
end


# considers only the n-th row for the fitness evaluation
consider = 1

# Data file, here is expected to be a csv, where the columns are in the order x1,x2...xk, y 
data = Matrix(CSV.read("hydrogen.csv", DataFrame))
data = data[all.(x -> !any(isnan, x), eachrow(data)),:]
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

@show size(x_data)
# Prepare test data
x_data_test = Float64.(data_test[1:consider:end, 1:(num_cols-1)])
y_data_test = Float64.(data_test[1:consider:end, num_cols])


best=runGep(epochs, pop_size, gene_count, head_len, utilized_syms,operators, callbacks, nodes, x_data',y_data, connection_syms, gep_params;correction_callback=corr_call_back!,
    loss_fun_="mse",x_data_test=x_data_test', y_data_test=y_data_test ,opt_method_const=:cg, hof=1)

#Show the result of the optimization
@show ("Fitness: (loss-fun): ", best[1].fitness)
@show ("R2-score-train: ", best[1].fitness_r2_train)
@show ("R2-score-test: ", best[1].fitness_r2_test)
@show ("Model in (x)-notation: ", string(best[1].compiled_function))

#@show string(best[1].compiled_function)
