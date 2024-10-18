TechnicalCompute.jl
===================

This package is a meta-package for a wide variety of commonly used Julia packages. It is
designed to make it easy to include one package that handles the diversity of technical
computing requirements for an undergraduate degree.

Topics (and the packages that support them)

* Data and IO
  * DataFrames: `DataFrames.jl`, `Tables.jl`,
  * CSVs: `DelimitedFiles.jl`, `CSV.jl`
  * Structured Text: `JSON.jl`, `TOML.jl`, `YAML.jl`
  * Binary formats: `JLD2.jl`, `HDF5.jl`, `MAT.jl`, `BSON.jl`, `NIfTI.jl` 
  * Meshes: `MeshIO.jl`, `GraphIO.jl`
* Discrete Mathematics
  * Graph algorithms: `Graphs.jl`, `SimpleWeightedGraphs.jl`, `Metis.jl`
  * Combinatorics: `Combinatorics.jl`
* Engineering
  * Signals: `DSP.jl`, `FFTW.jl`
  * Image Processing: `Images.jl`, `ImageShow.jl`, `TestImages.jl` 
* Geometry and Graphics
  * Primitives: `GeometryBasics.jl`
  * Plotting and displays: `CairoMakie.jl`  
  * Metrics and distances: `NearestNeighbors.jl`, `Distances.jl` 
  * Colors: `Colors.jl`, `ColorVectorSpace.jl`, `ColorSchemes.jl`,
  * Triangulation and Meshing: `DelaunayTriangulation.jl`, `Meshes.jl` (only imported)
* Machine learning and AI
  * `Flux.jl`, `Clustering.jl` 
* Numerical and Scientific Computing
  * Sparse matrices: `SparseArrays`
  * Differential equations: `DifferentialEquations.jl`
  * Polynomials and Series: `Polynomials.jl`, `TaylorSeries.jl`, `FastTransforms.jl`, `ApproxFun.jl` (only imported)
  * Applied Math Functions: `SpecialFunctions.jl`
  * Matrices and Linear Algebra: `LinearAlgebra.jl`, `Arpack.jl`, `Krylov.jl`, `LinearMaps.jl`
  * High-accuracy: `DoubleFloats.jl`, `MultiFloats.jl`, `KahanSummation.jl`
* Optimization and Operations Research
  * Modeling: `JuMP.jl` for Linear Programs and Nonlinear Problems 
  * and associated solvers `Ipopt.jl`, `HiGHS.jl`, `GLPK.jl` (and `Clp.jl` where supported)
  * Root finding: `Roots.jl`
  * Convex Solvers: `Convex.jl`, `SCS.jl`
* Probability and Statistics
  * General: `StatsBase.jl` and `Statistics.jl` 
  * Distributions: `Distributions.jl`
  * Kernel density estimation: `KernelDensity.jl`
  * Splines: `Interpolations.jl`
  * Multivariate: `MultivariateStats.jl`, `NMF.jl`
  * Simple functions: `Statistics`
  * General Linear Models: `GLM.jl`
* Programming 
  * `DataStructures.jl`
  * `OrderedCollections.jl`
  * `BenchmarkTools.jl`
  * `Transducers.jl`
  * `ThreadsX.jl`
  * `IterTools.jl`  
  * `Observables.jl`
* Symbolic computing
  * Automatic differentiation: `ForwardDiff.jl`
  * `Symbolics.jl`
* Helpful tools
  * Random numbers: `Random.jl`, `StableRNGs.jl`
  * Output: `ProgressMeter.jl`, `Printf.jl`
  * Dates: `Dates.jl`
  * LaTeX: `LaTeXStrings.jl`
  * Arrays: `StaticArrays.jl`, `IndirectArrays.jl`, `OffsetArrays.jl`, `FillArrays.jl`, `AxisArrays.jl`, `TiledIteration.jl`, `MosaicViews.jl`
  * Units: `Measures.jl`, `Unitful.jl`

And this also includes a variety of useful utility packages in the base Julia library and beyond to make working with files and output easy. 


## To consider
- `Bessels.jl`
- `ApproxFun.jl`
- `NMF.jl`
- `TSne.jl`
- `UMap.jl`
- `GraphIO.jl`

This takes a typical Julia install and provides all the functions needed for almost any undergraduate project. 

Examples
--------
* Morphing Carl Gustav Jacobi to David Young via an iterative method (from Numerical Methods by Greenbaum and Chartier)
