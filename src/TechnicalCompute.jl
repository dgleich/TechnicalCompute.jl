module TechnicalCompute

using Reexport

#=
# Simple stuff 
# Stats
# Clustering
# Plotting
# Data / IO / Mre
# Images/Signals
# Files
# Tools
# Extra
# SparseArrays
# Linear Algebra
# applied math
# Graphs
# Optimization
# Differential Equations
=#

packages = [
# base ... 
"SparseArrays",
"Random",
"LinearAlgebra",
# Data structures
"DataStructures",
"OrderedCollections",
#"TaylorSeries",
#"SuiteSparse",
"GeometryBasics",
# Machine Learning / Stats
"Distributions",
"StatsBase",
"KernelDensity",
"Statistics",
"MultivariateStats",
"Flux", 
# Clustering
"Clustering",
"Distances",
"NearestNeighbors",
# Images
"Images",
"FFTW",
"TestImages",
# Text Data
"DataFrames", 
"Tables",
"JSON",
"CSV",
"TOML",
"YAML",
"DelimitedFiles",
# Files
"FileIO",
"JLD2",
"HDF5",
"MAT",
"BSON",
# Extra
"BenchmarkTools",
"StableRNGs",
"ProgressMeter",
"Printf",
"Measures",
"Colors",
"Dates",
# Iteration tools
"Transducers",
"ThreadsX",
"IterTools",
# Array tools 
"StaticArrays",
"IndirectArrays",
"OffsetArrays",
"KahanSummation",
"FillArrays",
"TiledIteration",
# Signals
"DSP", 
# Plot Tools
"CairoMakie",
"Observables",
"LaTeXStrings", 
# Linear Algebra
"Arpack",
"LinearMaps",
"Krylov",
# Meshes, MeshIO
"Meshes",
"MeshIO", 
# Applied math
"DoubleFloats",
"MultiFloats",  
"Polynomials",
"SpecialFunctions",
"Roots", 
"TaylorSeries",
# Graphs and combinatorics 
"Graphs",
"SimpleWeightedGraphs",
"Metis",
"Combinatorics",
# Optimization
"JuMP",
"Ipopt",
"GLPK",
"Clp",
# diff Equations
"DifferentialEquations"
]

for pkg in packages
  eval(Meta.parse("@reexport using $pkg"))
end 

end # module
