# Installation Guide

This guide provides comprehensive instructions for installing GeneExpressionProgramming.jl and its dependencies across different platforms and environments.

## Prerequisites

Before installing GeneExpressionProgramming.jl, ensure you have the following prerequisites:

### Julia Requirements

- **Julia 1.6 or later**: The package requires Julia version 1.6 or higher. You can download Julia from the [official website](https://julialang.org/downloads/).
- **Package Manager**: Julia's built-in package manager (Pkg) is required for installation.

### System Requirements

The package has been tested on the following platforms:
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS (10.14+)
- Windows 10/11

### Memory and Performance Considerations

For optimal performance, we recommend:
- **Minimum RAM**: 4 GB (8 GB recommended for large datasets)
- **CPU**: Multi-core processor recommended for parallel operations
- **Storage**: At least 1 GB free space for package dependencies

## Installation Methods

### Method 1: Direct Installation from GitHub (Recommended)

The most straightforward way to install GeneExpressionProgramming.jl is directly from the GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git")
```

This method ensures you get the latest version with all recent updates and bug fixes.

### Method 2: Development Installation

If you plan to contribute to the package or need to modify the source code, you can install it in development mode:

```julia
using Pkg
Pkg.develop(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git")
```

This creates a local copy of the repository in your Julia development directory, allowing you to make changes and test them immediately.

### Method 3: Local Installation

If you have downloaded the source code locally:

```julia
using Pkg
Pkg.add(path="/path/to/GeneExpressionProgramming.jl")
```

Replace `/path/to/GeneExpressionProgramming.jl` with the actual path to your local copy.

## Dependency Installation

GeneExpressionProgramming.jl automatically installs its dependencies during the installation process. The main dependencies include:

### Core Dependencies

- **DynamicExpressions.jl**: For fast symbolic expression evaluation
- **Flux.jl**: For tensor operations and neural network backends
- **Random**: For random number generation and seeding
- **Statistics**: For statistical operations
- **LinearAlgebra**: For matrix operations

### Optional Dependencies

For enhanced functionality, you may want to install additional packages:

```julia
using Pkg
Pkg.add(["Plots", "CSV", "DataFrames", "Tensors"])
```

- **Plots.jl**: For visualization and plotting results
- **CSV.jl**: For reading CSV data files
- **DataFrames.jl**: For data manipulation and analysis
- **Tensors.jl**: For advanced tensor operations

## Verification

After installation, verify that the package works correctly:

```julia
using GeneExpressionProgramming

# Test basic functionality
println("GeneExpressionProgramming.jl installed successfully!")

# Create a simple regressor to test
regressor = GepRegressor(2)
println("Basic regressor created: ", typeof(regressor))
```

If the installation was successful, you should see output confirming the package is working.

## Troubleshooting

### Common Installation Issues

#### Issue 1: Package Not Found
```
ERROR: The following package names could not be resolved:
 * GeneExpressionProgramming (not found in project, manifest or registry)
```

**Solution**: Ensure you're using the correct URL and that you have internet connectivity:
```julia
Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git")
```

#### Issue 2: Dependency Conflicts
```
ERROR: Unsatisfiable requirements detected for package...
```

**Solution**: Update your Julia packages and try again:
```julia
using Pkg
Pkg.update()
Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git")
```

#### Issue 3: Compilation Errors
If you encounter compilation errors during installation:

1. **Update Julia**: Ensure you're using Julia 1.6 or later
2. **Clear package cache**: 
   ```julia
   using Pkg
   Pkg.gc()
   ```
3. **Reinstall dependencies**:
   ```julia
   Pkg.instantiate()
   ```

#### Issue 4: Permission Errors (Linux/macOS)
If you encounter permission errors:

1. **Check Julia installation**: Ensure Julia is properly installed with appropriate permissions
2. **Use local package directory**: Consider installing packages in a local directory
3. **Avoid sudo**: Never use `sudo` with Julia package operations


## Performance Optimization

### Julia Startup Optimization

To improve Julia startup time with GeneExpressionProgramming.jl:

1. **Precompile packages**:
   ```julia
   using Pkg
   Pkg.precompile()
   ```

2. **Use PackageCompiler.jl** for creating system images:
   ```julia
   using Pkg
   Pkg.add("PackageCompiler")
   using PackageCompiler
   create_sysimage(["GeneExpressionProgramming"]; sysimage_path="gep_sysimage.so")
   ```

### Memory Management

For large-scale problems:

1. **Increase Julia heap size**:
   ```bash
   julia --heap-size-hint=8G
   ```

2. **Monitor memory usage**:
   ```julia
   using Profile
   @profile your_gep_code()
   Profile.print()
   ```

## Environment Setup

### Jupyter Notebook Integration

To use GeneExpressionProgramming.jl in Jupyter notebooks:

```julia
using Pkg
Pkg.add("IJulia")
using IJulia
notebook()
```

### VS Code Integration

For development with VS Code:

1. Install the Julia extension for VS Code
2. Configure the Julia executable path
3. Use the integrated REPL for interactive development

### Docker Environment

For containerized environments, use the official Julia Docker image:

```dockerfile
FROM julia:1.8

RUN julia -e 'using Pkg; Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git")'

WORKDIR /app
COPY . .

CMD ["julia", "your_script.jl"]
```

## Next Steps

After successful installation, proceed to:

1. [Getting Started Guide](getting-started.md) - Learn basic usage patterns
2. [Core Concepts](core-concepts.md) - Understand the theoretical foundations
3. [Examples](../examples/) - Explore practical applications

For additional help, consult [GitHub repository](https://github.com/maxreiss123/GeneExpressionProgramming.jl/issues).

---

*Last updated: January 2025*

