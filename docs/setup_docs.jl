#!/usr/bin/env julia

"""
Documentation Setup Script for GeneExpressionProgramming.jl

This script sets up the Documenter.jl documentation system for your package.
Run this script from your package root directory.

Usage:
    julia setup_docs.jl
"""

using Pkg

println("Setting up Documenter.jl documentation system...")
println()

# Check if we're in a package directory
if !isfile("Project.toml")
    println("Error: No Project.toml found. Please run this script from your package root directory.")
    exit(1)
end

# Read package information
project_toml = Pkg.TOML.parsefile("Project.toml")
package_name = project_toml["name"]
package_uuid = project_toml["uuid"]



# Create docs directory structure
println("Creating documentation directory structure...")
mkpath("docs/src/examples")
mkpath("docs/.github/workflows")

# Install Documenter.jl and dependencies
println("Installing documentation dependencies...")
Pkg.activate("docs")
Pkg.add([
    "Documenter",
    "DocumenterTools", 
    "Plots",
    "PlotlyJS"
])

# Add the main package as a dependency
println("Adding main package as documentation dependency...")
Pkg.develop(PackageSpec(path=".."))

println()
println("Documentation system setup complete!")


# Deactivate docs environment
Pkg.activate()

