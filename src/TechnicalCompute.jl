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
#"SuiteSparse",
"GeometryBasics",
# Machine Learning / Stats
"Distributions",
"StatsBase",
"KernelDensity",
"Statistics",
"MultivariateStats",
"Flux", 
"MLDatasets", 
"ReinforcementLearning",
"NMF",
"RDatasets",
# Clustering
"Clustering",
"Distances",
"NearestNeighbors",
# Images
"Images",
"FFTW",
"TestImages",
"ImageShow", 
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
"NIfTI", # used in Makie demos... 
"GraphIO", 
# Extra
"BenchmarkTools",
"StableRNGs",
"ProgressMeter",
"Printf",
"Measures",
"Unitful", 
"Colors",
"ColorVectorSpace",
"ColorSchemes",
"Dates",
"HypertextLiteral", 
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
"AxisArrays", # this is reexported from Images anyway...
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
#"Meshes",
"MeshIO", 
"DelaunayTriangulation", # This is reexported above
# Applied math
"DoubleFloats",
"MultiFloats",  
"Polynomials",
"SpecialFunctions",
"Roots", 
"TaylorSeries",
#"ApproxFun",
"FastTransforms",
"Interpolations", 
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
"HiGHS",
"Convex",
"SCS",
# Symbolic
"ForwardDiff", 
"Symbolics",
# diff Equations
"DifferentialEquations"
]


# handle Clp failures
if Sys.ARCH == :aarch64 && Sys.isapple() 
  filter!(x -> x != "Clp", packages)
end 

for pkg in packages
  eval(Meta.parse("@reexport using $pkg"))
end 

# Meshes has too many overlapping names, so we'll just import it here
@reexport import Meshes 
# ApproxFun has too many overlapping names. 
@reexport import ApproxFun 

# these 
@reexport import UnicodePlots
@reexport import PGFPlotsX
@reexport import NaNMath


if get(()->"true", ENV, "JULIA_TECHNICALCOMPUTE_USE_GLMAKIE") == "true"
  @reexport import GLMakie 
end 

include("overrides.jl")

# this one is bizarre, since Meshes exports a type with the same name as
# DelaunayTriangulation, we need to get a function from the module to get 
# the module, 
# I found this here: https://stackoverflow.com/questions/38819327/given-a-function-object-how-do-i-find-its-name-and-module

# DelaunayTriangulation = typeof(triangulate).name.module
# export DelaunayTriangulation

end # module
