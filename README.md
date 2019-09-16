TechnicalCompute.jl
===================

This package is a meta-package for a wide variety of commonly used Julia packages. It is
designed to make it easy to include one package that handles the diversity of technical
computing requirements for an undergraduate degree.

Topics (and the packages that support them)

* Discrete Mathematics
  * Graph algorithms: `LightGraphs.jl`
  * Combinatorics: `Combinatorics.jl`
* Engineering
  * Signals: `DSP.jl`, `FFTW.jl`
  * Image Processing: `Images.jl`
* Numerical and Scientific Computing
  * Sparse matrices: `SparseArrays`, `SuiteSparse`
  * Differential equations: `DifferentialEquations.jl`
  * Polynomials: `Polynomials.jl`
* Optimization and Operations Research
  * Modeling: `JuMP.jl` for Linear Programs and Nonlinear Problems
  * Root finding: `Roots.jl`
* Statistics
  * Multivariate: `MultivariateStats.jl`
  * Simple functions: `Statistics`
  
  
And this also includes a variety of useful utility packages in the base Julia library and beyond to make working with files and output easy. 
* `Random` 
* `DelimitedFiles`
* `IterTools`
* `Printf`
* `SpecialFunctions`
* `BenchmarkTools`
* `JSON.jl`
* `HDF5.jl`
* `Plots.jl` 
* `CSV.jl`

This takes a typical Julia install and provides all the functions needed for almost any undergraduate project. 

Examples
--------
* Morphing Carl Gustav Jacobi to David Young via an iterative method (from Numerical Methods by Greenbaum and Chartier)
