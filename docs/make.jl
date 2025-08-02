using Documenter
using DocumenterTools

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

