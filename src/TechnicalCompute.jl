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
"HiGHS",
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

const overrides = Set{Symbol}()

update!(p::ProgressMeter.AbstractProgress, val, color; options...) = ProgressMeter.update!(p, val, color; options...)
update!(p::Union{ProgressMeter.Progress,ProgressMeter.ProgressUnknown,ProgressMeter.ProgressThresh}, val; options...) = ProgressMeter.update!(p, val; options...)
update!(p::Union{ProgressMeter.Progress,ProgressMeter.ProgressUnknown,ProgressMeter.ProgressThresh}; options...) = ProgressMeter.update!(p; options...)
update!(h::DataStructures.MutableBinaryHeap, i, v) = DataStructures.update!(h, i, v)
#update!(points, bt::Delaunator.BasicTriangulation, cdata::Delaunator.TriangulationTemporaries; tol) = Delaunator.update!(points, bt, cdata; tol)
export update!
push!(overrides, :update!)

# This ignores images, which also exports it's function under tensor. 
⊗(k::Integer) = LinearMaps.⊗(k)
⊗(A,B,Cs...) = LinearMaps.⊗(A,B,Cs...)
⊗(x::T, y::T) where T<:Union{Float16, Float32, Float64} = DoubleFloats.⊗(x, y)
export ⊗
push!(overrides, :⊗)

@doc (@doc Meshes.partition)
partition(rng::AbstractRNG, object, method::Meshes.PartitionMethod) = Meshes.partition(rng, object, method)
partition(object, method::Meshes.PartitionMethod) = Meshes.partition(object, method)

@doc (@doc IterTools.partition)
partition(xs, n::Int64) = IterTools.partition(xs, n)
partition(xs::I, n::Int64, step::Int64) where I = IterTools.partition(xs, n, step)
export partition
push!(overrides, :partition)


end # module
