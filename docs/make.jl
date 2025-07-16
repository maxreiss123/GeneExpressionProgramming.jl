using Documenter
using DocumenterTools

# Push the parent directory to LOAD_PATH so we can load the package
push!(LOAD_PATH, "../")

# Try to load the actual package first
package_loaded = false
try
    @eval using GeneExpressionProgramming
    global package_loaded = true
    @info "Successfully loaded GeneExpressionProgramming package"
catch e
    @warn "Could not load GeneExpressionProgramming package: $e"
    @info "Will create mock module for documentation build"
end

# Create mock module if the real package couldn't be loaded
if !package_loaded
    @eval module GeneExpressionProgramming
        """
            GepRegressor(number_features::Int; kwargs...)
        
        Create a Gene Expression Programming regressor for symbolic regression.
        
        # Arguments
        - `number_features::Int`: Number of input features
        - `population_size::Int = 1000`: Size of the population
        - `gene_count::Int = 2`: Number of genes per chromosome
        - `head_len::Int = 7`: Head length of each gene
        - `function_set::Vector{Symbol} = [:+, :-, :*, :/]`: Available functions
        
        # Examples
        ```julia
        regressor = GepRegressor(3; population_size=500, gene_count=2)
        ```
        """
        struct GepRegressor
            number_features::Int
            population_size::Int
            gene_count::Int
            head_len::Int
            function_set::Vector{Symbol}
        end
        
        function GepRegressor(number_features::Int; 
                             population_size::Int = 1000,
                             gene_count::Int = 2,
                             head_len::Int = 7,
                             function_set::Vector{Symbol} = [:+, :-, :*, :/])
            return GepRegressor(number_features, population_size, gene_count, head_len, function_set)
        end
        
        """
            fit!(regressor, epochs::Int, population_size::Int, x_data, y_data; kwargs...)
        
        Train the GEP regressor on data.
        
        # Arguments
        - `regressor`: GepRegressor instance
        - `epochs::Int`: Number of generations to evolve
        - `population_size::Int`: Population size for evolution
        - `x_data`: Input features (features as rows, samples as columns)
        - `y_data`: Target values
        
        # Keyword Arguments
        - `loss_fun::String = "mse"`: Loss function ("mse", "mae", "rmse")
        - `x_test = nothing`: Test features for validation
        - `y_test = nothing`: Test targets for validation
        
        # Examples
        ```julia
        fit!(regressor, 1000, 1000, x_train', y_train; loss_fun="mse")
        ```
        """
        function fit!(regressor::GepRegressor, epochs::Int, population_size::Int, x_data, y_data; kwargs...)
            # Mock implementation
            println("Training GEP regressor for $epochs epochs...")
            return nothing
        end
        
        """
            GepTensorRegressor(number_features::Int, gene_count::Int, head_len::Int; kwargs...)
        
        Create a Gene Expression Programming regressor for tensor (vector/matrix) symbolic regression.
        
        # Arguments
        - `number_features::Int`: Number of input features
        - `gene_count::Int`: Number of genes per chromosome
        - `head_len::Int`: Head length of each gene
        
        # Keyword Arguments
        - `feature_names::Vector{String} = []`: Names for features (for interpretability)
        
        # Examples
        ```julia
        regressor = GepTensorRegressor(5, 2, 3; feature_names=["x1", "x2", "U1", "U2", "U3"])
        ```
        """
        struct GepTensorRegressor
            number_features::Int
            gene_count::Int
            head_len::Int
            feature_names::Vector{String}
        end
        
        function GepTensorRegressor(number_features::Int, gene_count::Int, head_len::Int; 
                                   feature_names::Vector{String} = String[])
            return GepTensorRegressor(number_features, gene_count, head_len, feature_names)
        end
        
        export GepRegressor, GepTensorRegressor, fit!
    end
end

# Configure Documenter
makedocs(;
    modules=[GeneExpressionProgramming],
    authors="Maximilian Reissmann and contributors",
    repo="https://github.com/maxreiss123/GeneExpressionProgramming.jl/blob/{commit}{path}#{line}",
    sitename="GeneExpressionProgramming.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://maxreiss123.github.io/GeneExpressionProgramming.jl",
        edit_link="master",
        assets=String[],
        sidebar_sitename=true,
        collapselevel=1,
        ansicolor=true,
    ),
    pages=[
        "Home" => "index.md",
        "User Guide" => [
            "Installation" => "installation.md",
            "Getting Started" => "getting-started.md",
            "Core Concepts" => "core-concepts.md",
        ],
        "Examples" => [
            "Basic Regression" => "examples/basic-regression.md",
            "Multi-Objective Optimization" => "examples/multi-objective.md",
            "Physical Dimensionality" => "examples/physical-dimensions.md",
            "Tensor Regression" => "examples/tensor-regression.md",
        ],
        "Reference" => [
            "API Reference" => "api-reference.md"
        ],
    ],
    clean=true,
    doctest=true,
    linkcheck=false,  # Set to true to check external links
    warnonly=[:missing_docs, :cross_references, :linkcheck],  # Use warnonly instead of strict
)

# Deploy documentation to GitHub Pages (only in CI)
deploydocs(;
    repo="github.com/maxreiss123/GeneExpressionProgramming.jl",
    target="build",
    branch="feature/modularError",
    devbranch="feature/modularError",
    push_preview=true,
)

