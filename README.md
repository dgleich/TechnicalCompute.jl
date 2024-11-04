TechnicalCompute.jl
===================

This package is a meta-package for a wide variety of commonly used Julia packages. It is
designed to make it easy to include one package that handles the diversity of technical
computing requirements for an undergraduate degree.

Topics (and the packages that support them)
* Data and IO
  * DataFrames: [`DataFrames.jl`](https://github.com/JuliaData/DataFrames.jl), [`Tables.jl`](https://github.com/JuliaData/Tables.jl)
  * CSVs: [`DelimitedFiles.jl`](https://github.com/JuliaLang/julia/tree/master/stdlib/DelimitedFiles), [`CSV.jl`](https://github.com/JuliaData/CSV.jl)
  * Structured Text: [`JSON.jl`](https://github.com/JuliaIO/JSON.jl), [`TOML.jl`](https://github.com/JuliaLang/TOML.jl), [`YAML.jl`](https://github.com/JuliaData/YAML.jl), [`Serde.jl`](https://github.com/bhftbootcamp/Serde.jl), [`EzXML.jl`](https://github.com/JuliaIO/EzXML.jl)
  * Binary formats: [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl), [`HDF5.jl`](https://github.com/JuliaIO/HDF5.jl), [`MAT.jl`](https://github.com/JuliaIO/MAT.jl), [`BSON.jl`](https://github.com/JuliaIO/BSON.jl), [`NIfTI.jl`](https://github.com/JuliaIO/NIfTI.jl), [`CodecBzip2.jl`](https://github.com/bicycle1885/CodecBzip2.jl), [`CodecLz4.jl`](https://github.com/bicycle1885/CodecLz4.jl),  [`CodecXz.jl`](https://github.com/bicycle1885/CodecXz.jl),  [`CodecZlib.jl`](https://github.com/bicycle1885/CodecZlib.jl), [`CodecZstd.jl`](https://github.com/bicycle1885/CodecZstd.jl), [`ZipFile.jl`](https://github.com/fhs/ZipFile.jl), [`TranscodingStreams.jl`](https://github.com/JuliaIO/TranscodingStreams.jl), [`LibSndFile.jl`](https://github.com/JuliaAudio/LibSndFile.jl)   
  * Meshes and Graphs: [`MeshIO.jl`](https://github.com/JuliaIO/MeshIO.jl)
  * Datasets: [`MLDatasets.jl`](https://github.com/JuliaML/MLDatasets.jl), [`RDatasets.jl`](https://github.com/JuliaStats/RDatasets.jl)
* Discrete Mathematics
  * Graph algorithms: [`Graphs.jl`](https://github.com/JuliaGraphs/Graphs.jl), [`SimpleWeightedGraphs.jl`](https://github.com/JuliaGraphs/SimpleWeightedGraphs.jl), [`Metis.jl`](https://github.com/JuliaSparse/Metis.jl)
  * Combinatorics: [`Combinatorics.jl`](https://github.com/JuliaMath/Combinatorics.jl)
* Engineering
  * Signals: [`DSP.jl`](https://github.com/JuliaDSP/DSP.jl), [`FFTW.jl`](https://github.com/JuliaMath/FFTW.jl), [`SampledSignals.jl`](https://github.com/JuliaAudio/SampledSignals.jl)
  * Image Processing: [`Images.jl`](https://github.com/JuliaImages/Images.jl), [`ImageShow.jl`](https://github.com/JuliaImages/ImageShow.jl), [`TestImages.jl`](https://github.com/JuliaImages/TestImages.jl) 
* Geometry and Graphics
  * Primitives: [`GeometryBasics.jl`](https://github.com/JuliaGeometry/GeometryBasics.jl)
  * Plotting and displays: [`CairoMakie.jl`](https://github.com/MakieOrg/CairoMakie.jl)  
  * Metrics and distances: [`NearestNeighbors.jl`](https://github.com/KristofferC/NearestNeighbors.jl), [`Distances.jl`](https://github.com/JuliaStats/Distances.jl) 
  * Colors: [`Colors.jl`](https://github.com/JuliaGraphics/Colors.jl), [`ColorVectorSpace.jl`](https://github.com/JuliaGraphics/ColorVectorSpace.jl), [`ColorSchemes.jl`](https://github.com/JuliaGraphics/ColorSchemes.jl)
  * Triangulation and Meshing: [`DelaunayTriangulation.jl`](https://github.com/JuliaGeometry/DelaunayTriangulation.jl), [`Meshes.jl`](https://github.com/JuliaGeometry/Meshes.jl) (only imported)
* Machine learning and AI
  * [`Flux.jl`](https://github.com/FluxML/Flux.jl), [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl) 
* Numerical and Scientific Computing
  * Sparse matrices: [`SparseArrays`](https://github.com/JuliaLang/julia/tree/master/stdlib/SparseArrays)
  * Differential equations: [`DifferentialEquations.jl`](https://github.com/SciML/DifferentialEquations.jl)
  * Polynomials and Series: [`Polynomials.jl`](https://github.com/JuliaMath/Polynomials.jl), [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl), [`FastTransforms.jl`](https://github.com/JuliaApproximation/FastTransforms.jl), [`ApproxFun.jl`](https://github.com/JuliaApproximation/ApproxFun.jl) (only imported)
  * Applied Math Functions: [`SpecialFunctions.jl`](https://github.com/JuliaMath/SpecialFunctions.jl)
  * Matrices and Linear Algebra: [`LinearAlgebra.jl`](https://github.com/JuliaLang/julia/tree/master/stdlib/LinearAlgebra), [`Arpack.jl`](https://github.com/JuliaLinearAlgebra/Arpack.jl), [`Krylov.jl`](https://github.com/Jutho/Krylov.jl), [`LinearMaps.jl`](https://github.com/Jutho/LinearMaps.jl), [`MatrixMarket.jl`](https://github.com/JuliaSparse/MatrixMarket.jl), [`SuiteSparseMatrixCollection.jl`](https://github.com/JuliaSmoothOptimizers/SuiteSparseMatrixCollection.jl)
  * High-accuracy: [`DoubleFloats.jl`](https://github.com/JuliaMath/DoubleFloats.jl), [`MultiFloats.jl`](https://github.com/dzhang314/MultiFloats.jl), [`KahanSummation.jl`](https://github.com/JuliaMath/KahanSummation.jl)
  * Tensor methods: [`ITensors.jl`](https://github.com/ITensor/ITensors.jl), [`ITensorMPS.jl`](https://github.com/ITensor/ITensorMPS.jl) [`ITensorNetworks.jl`](https://github.com/ITensor/ITensorNetworks.jl)
* Optimization and Operations Research
  * Modeling: [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) for Linear Programs and Nonlinear Problems 
  * and associated solvers [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl), [`HiGHS.jl`](https://github.com/jump-dev/HiGHS.jl), [`GLPK.jl`](https://github.com/jump-dev/GLPK.jl), [`Tulip.jl`](https://github.com/ds4dm/Tulip.jl),  (and [`Clp.jl`](https://github.com/jump-dev/Clp.jl) where supported)
  * General solvers: [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), [`NonlinearSolve.jl`](https://github.com/SciML/NonlinearSolve.jl),[`LineSearches.jl`](https://github.com/JuliaNLSolvers/LineSearches.jl) (only imported)
  * Least squares: [`LsqFit.jl`](https://github.com/JuliaNLSolvers/LsqFit.jl)
  * Root finding: [`Roots.jl`](https://github.com/JuliaMath/Roots.jl)
  * Convex Solvers: [`Convex.jl`](https://github.com/jump-dev/Convex.jl), [`SCS.jl`](https://github.com/jump-dev/SCS.jl)
  * Test Problems: [`OptimTestProblems.jl`](https://github.com/JuliaNLSolvers/OptimTestProblems.jl)
* Probability and Statistics
  * General: [`StatsBase.jl`](https://github.com/JuliaStats/StatsBase.jl), [`Statistics.jl`](https://github.com/JuliaLang/julia/tree/master/stdlib/Statistics), [`OnlineStats.jl`](https://github.com/joshday/OnlineStats.jl)
  * Distributions: [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl)
  * Kernel density estimation: [`KernelDensity.jl`](https://github.com/JuliaStats/KernelDensity.jl)
  * Splines: [`Interpolations.jl`](https://github.com/JuliaMath/Interpolations.jl)
  * Multivariate: [`MultivariateStats.jl`](https://github.com/JuliaStats/MultivariateStats.jl), [`NMF.jl`](https://github.com/JuliaStats/NMF.jl)
  * Simple functions: [`Statistics`](https://github.com/JuliaLang/julia/tree/master/stdlib/Statistics)
  * General Linear Models: [`GLM.jl`](https://github.com/JuliaStats/GLM.jl)
* Programming 
  * [`DataStructures.jl`](https://github.com/JuliaCollections/DataStructures.jl)
  * [`OrderedCollections.jl`](https://github.com/JuliaCollections/OrderedCollections.jl)
  * [`BenchmarkTools.jl`](https://github.com/JuliaCI/BenchmarkTools.jl)
  * [`Transducers.jl`](https://github.com/JuliaFolds/Transducers.jl)
  * [`ThreadsX.jl`](https://github.com/tkf/ThreadsX.jl)
  * [`IterTools.jl`](https://github.com/JuliaCollections/IterTools.jl)  
  * [`Observables.jl`](https://github.com/JuliaGizmos/Observables.jl)
* Symbolic computing
  * Automatic differentiation: [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl)
  * [`Symbolics.jl`](https://github.com/JuliaSymbolics/Symbolics.jl)
* Helpful tools
  * Random numbers: [`Random.jl`](https://github.com/JuliaLang/julia/tree/master/stdlib/Random), [`StableRNGs.jl`](https://github.com/rfourquet/StableRNGs.jl)
  * Output: [`ProgressMeter.jl`](https://github.com/timholy/ProgressMeter.jl), [`Printf.jl`](https://github.com/JuliaLang/julia/tree/master/stdlib/Printf)
  * Dates: [`Dates.jl`](https://github.com/JuliaLang/julia/tree/master/stdlib/Dates)
  * LaTeX: [`LaTeXStrings.jl`](https://github.com/stevengj/LaTeXStrings.jl), [`Latexify.jl`](https://github.com/korsbo/Latexify.jl)
  * Parsing: [`ParserCombinator.jl`](https://github.com/JuliaParsing/ParserCombinator.jl) (only imported)
  * Arrays: [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl), [`IndirectArrays.jl`](https://github.com/JuliaArrays/IndirectArrays.jl), [`OffsetArrays.jl`](https://github.com/JuliaArrays/OffsetArrays.jl), [`FillArrays.jl`](https://github.com/JuliaArrays/FillArrays.jl), [`AxisArrays.jl`](https://github.com/JuliaArrays/AxisArrays.jl), [`TiledIteration.jl`](https://github.com/JuliaArrays/TiledIteration.jl), [`MosaicViews.jl`](https://github.com/JuliaArrays/MosaicViews.jl)
  * Units: [`Measures.jl`](https://github.com/JuliaGraphics/Measures.jl), [`Unitful.jl`](https://github.com/PainterQubits/Unitful.jl)

And this also includes a variety of useful utility packages in the base Julia library and beyond to make working with files and output easy. 

## Coming soon
- [`GraphIO.jl`](https://github.com/JuliaGraphs/GraphIO.jl)

## To consider
- `Bessels.jl`
- `TSne.jl`
- `UMap.jl`


This takes a typical Julia install and provides all the functions needed for almost any undergraduate project. 

Examples
--------
* Morphing Carl Gustav Jacobi to David Young via an iterative method (from Numerical Methods by Greenbaum and Chartier)
