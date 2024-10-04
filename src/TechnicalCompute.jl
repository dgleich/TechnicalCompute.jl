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

# need to reexport DelaunayTriangulation first...
@reexport using DelaunayTriangulation
#DelaunayTriangulation = typeof(triangulate).name.module
#export DelaunayTriangulation

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
# "DelaunayTriangulation", # This is reexported above
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

# Name degree is not equal in packages Meshes and Polynomials and not in overrides
# Name degree is not equal in packages Meshes and Graphs and not in overrides
# Name degree is not equal in packages Polynomials and Graphs and not in overrides
@doc (@doc Meshes.degree)
degree(b::Meshes.BezierCurve) = Meshes.degree(b)
@doc (@doc Graphs.degree)
degree(g::Graphs.AbstractGraph, i) = Graphs.degree(g, i)
degree(g::Graphs.AbstractGraph) = Graphs.degree(g)
@doc (@doc Polynomials.degree)
degree(p::AbstractPolynomial) = Polynomials.degree(p)
degree(p::Polynomials.AbstractRationalFunction) = Polynomials.degree(p)
push!(overrides, :degree)

# width from CairoMakie == width from GeometryBasics
# Name width is not equal in packages GeometryBasics and Measures and not in overrides
# Name width is not equal in packages Measures and CairoMakie and not in overrides
@doc (@doc GeometryBasics.width)
width(prim::HyperRectangle) = GeometryBasics.width(prim)
@doc (@doc Measures.width)
width(x::BoundingBox) = Measures.width(x)
push!(overrides, :width)

# width from CairoMakie == width from GeometryBasics
# Name height is not equal in packages GeometryBasics and Measures and not in overrides
# Name width is not equal in packages Measures and CairoMakie and not in overrides
@doc (@doc GeometryBasics.height)
height(prim::HyperRectangle) = GeometryBasics.height(prim)
@doc (@doc Measures.height)
height(x::BoundingBox) = Measures.height(x)
push!(overrides, :height)

# this one is bizarre, since Meshes exports a type with the same name as
# DelaunayTriangulation, we need to get a function from the module to get 
# the module, 
# I found this here: https://stackoverflow.com/questions/38819327/given-a-function-object-how-do-i-find-its-name-and-module

# DelaunayTriangulation = typeof(triangulate).name.module
# export DelaunayTriangulation

# stuff to fix...
# Name Fixed is not equal in packages Images and CairoMakie and not in overrides
# Name MultiPoint is not equal in packages GeometryBasics and Meshes and not in overrides
# Name spectrogram is not equal in packages Flux and DSP and not in overrides
# Name Chain is not equal in packages Flux and Meshes and not in overrides
# Name Cylinder is not equal in packages GeometryBasics and Meshes and not in overrides
# Name fit is not equal in packages Distributions and Polynomials and not in overrides
# Name fit is not equal in packages StatsBase and Polynomials and not in overrides
# Name fit is not equal in packages MultivariateStats and Polynomials and not in overrides
# Name discretize is not equal in packages Meshes and DifferentialEquations and not in overrides
# Name rmsd is not equal in packages StatsBase and Distances and not in overrides
# Name Mesh is not equal in packages GeometryBasics and CairoMakie and not in overrides
# Name Mesh is not equal in packages GeometryBasics and Meshes and not in overrides
# Name Mesh is not equal in packages CairoMakie and Meshes and not in overrides
# Name reset! is not equal in packages DataStructures and DSP and not in overrides
# Name conv is not equal in packages Flux and DSP and not in overrides
# Name weights is not equal in packages StatsBase and Graphs and not in overrides
# Name integrate is not equal in packages Polynomials and TaylorSeries and not in overrides
# Name inverse is not equal in packages Meshes and TaylorSeries and not in overrides
# Name islinear is not equal in packages StatsBase and DifferentialEquations and not in overrides
# Name trim! is not equal in packages StatsBase and CairoMakie and not in overrides
# Name right is not equal in packages Transducers and CairoMakie and not in overrides
# Name ⊕ is not equal in packages LinearMaps and DoubleFloats and not in overrides
# Name Zeros is not equal in packages FillArrays and JuMP and not in overrides
# Name Vec2 is not equal in packages GeometryBasics and Measures and not in overrides
# Name Vec2 is not equal in packages Measures and CairoMakie and not in overrides
# Name entropy is not equal in packages Distributions and Images and not in overrides
# Name entropy is not equal in packages StatsBase and Images and not in overrides
# Name Bisection is not equal in packages Roots and DifferentialEquations and not in overrides
# Name solve! is not equal in packages Krylov and Roots and not in overrides
# Name solve! is not equal in packages Krylov and DifferentialEquations and not in overrides
# Name Box is not equal in packages CairoMakie and Meshes and not in overrides
# Name volume is not equal in packages GeometryBasics and CairoMakie and not in overrides
# Name volume is not equal in packages GeometryBasics and Meshes and not in overrides
# Name volume is not equal in packages CairoMakie and Meshes and not in overrides
# Name attributes is not equal in packages HDF5 and CairoMakie and not in overrides
# Name msd is not equal in packages StatsBase and Distances and not in overrides
# Name transform is not equal in packages MultivariateStats and DataFrames and not in overrides
# Name direction is not equal in packages GeometryBasics and Meshes and not in overrides
# Name issquare is not equal in packages DoubleFloats and DifferentialEquations and not in overrides
# Name nan is not equal in packages Images and DoubleFloats and not in overrides
# Name Polygon is not equal in packages GeometryBasics and Meshes and not in overrides
# Name radius is not equal in packages GeometryBasics and Meshes and not in overrides
# Name radius is not equal in packages GeometryBasics and Graphs and not in overrides
# Name radius is not equal in packages Meshes and Graphs and not in overrides
# Name connect is not equal in packages GeometryBasics and Meshes and not in overrides
# Name evaluate is not equal in packages MultivariateStats and Distances and not in overrides
# Name evaluate is not equal in packages MultivariateStats and Images and not in overrides
# Name evaluate is not equal in packages MultivariateStats and TaylorSeries and not in overrides
# Name evaluate is not equal in packages Distances and TaylorSeries and not in overrides
# Name evaluate is not equal in packages Images and TaylorSeries and not in overrides
# Name Length is not equal in packages Measures and StaticArrays and not in overrides
# Name properties is not equal in packages Images and IterTools and not in overrides
# Name Filters is not equal in packages HDF5 and DSP and not in overrides
# Name rotate! is not equal in packages LinearAlgebra and CairoMakie and not in overrides
# Name Categorical is not equal in packages Distributions and CairoMakie and not in overrides
# Name Pyramid is not equal in packages GeometryBasics and Meshes and not in overrides
# Name params is not equal in packages Distributions and BenchmarkTools and not in overrides
# Name Sphere is not equal in packages GeometryBasics and Meshes and not in overrides
# Name Sphere is not equal in packages CairoMakie and Meshes and not in overrides
# Name Triangle is not equal in packages GeometryBasics and Meshes and not in overrides
# Name boundingbox is not equal in packages CairoMakie and Meshes and not in overrides
# Name orientation is not equal in packages Images and Meshes and not in overrides
# Name vertices is not equal in packages Meshes and Graphs and not in overrides
# Name Polytope is not equal in packages GeometryBasics and Meshes and not in overrides
# Name center is not equal in packages Images and Meshes and not in overrides
# Name center is not equal in packages Images and Graphs and not in overrides
# Name center is not equal in packages Meshes and Graphs and not in overrides
# Name area is not equal in packages GeometryBasics and Meshes and not in overrides
# Name Axis is not equal in packages Images and CairoMakie and not in overrides
# Name metadata is not equal in packages DataFrames and FileIO and not in overrides
# Name metadata is not equal in packages DataFrames and Meshes and not in overrides
# Name metadata is not equal in packages FileIO and Meshes and not in overrides
# Name Grid is not equal in packages Meshes and Graphs and not in overrides
# Name faces is not equal in packages GeometryBasics and Meshes and not in overrides
# Name Point is not equal in packages GeometryBasics and Meshes and not in overrides
# Name Point is not equal in packages CairoMakie and Meshes and not in overrides
# Name mode is not equal in packages Distributions and JuMP and not in overrides
# Name mode is not equal in packages StatsBase and JuMP and not in overrides
# Name Tetrahedron is not equal in packages GeometryBasics and Meshes and not in overrides
# Name trim is not equal in packages StatsBase and BenchmarkTools and not in overrides
# Name intersects is not equal in packages GeometryBasics and Meshes and not in overrides
# Name Partition is not equal in packages Transducers and Meshes and not in overrides
# Name Partition is not equal in packages Transducers and Combinatorics and not in overrides
# Name Partition is not equal in packages Meshes and Combinatorics and not in overrides
# Name Circle is not equal in packages GeometryBasics and Meshes and not in overrides
# Name Circle is not equal in packages CairoMakie and Meshes and not in overrides
# Name density is not equal in packages CairoMakie and Graphs and not in overrides
# Name ⊗ is not equal in packages Images and LinearMaps and not in overrides
# Name ⊗ is not equal in packages Images and DoubleFloats and not in overrides
# Name ⊗ is not equal in packages LinearMaps and DoubleFloats and not in overrides
# Name stft is not equal in packages Flux and DSP and not in overrides
# Name Line is not equal in packages GeometryBasics and Meshes and not in overrides
# Name FunctionMap is not equal in packages LinearMaps and DifferentialEquations and not in overrides
# Name scale! is not equal in packages Distributions and CairoMakie and not in overrides
# Name imrotate is not equal in packages Flux and Images and not in overrides
# Name Vec3 is not equal in packages GeometryBasics and Measures and not in overrides
# Name Vec3 is not equal in packages Measures and CairoMakie and not in overrides
# Name order is not equal in packages DataFrames and Polynomials and not in overrides
# Name meanad is not equal in packages StatsBase and Distances and not in overrides
# Name groupby is not equal in packages DataFrames and IterTools and not in overrides
# Name complement is not equal in packages DataStructures and Images and not in overrides
# Name complement is not equal in packages DataStructures and Graphs and not in overrides
# Name complement is not equal in packages Images and Graphs and not in overrides
# Name Vec is not equal in packages GeometryBasics and Measures and not in overrides
# Name Vec is not equal in packages GeometryBasics and Meshes and not in overrides
# Name Vec is not equal in packages Measures and CairoMakie and not in overrides
# Name Vec is not equal in packages Measures and Meshes and not in overrides
# Name Vec is not equal in packages CairoMakie and Meshes and not in overrides
# Name top is not equal in packages DataStructures and CairoMakie and not in overrides
# Name top is not equal in packages DataStructures and Meshes and not in overrides
# Name top is not equal in packages CairoMakie and Meshes and not in overrides
# Name CartesianGrid is not equal in packages Meshes and DifferentialEquations and not in overrides
# Name hamming is not equal in packages Distances and DSP and not in overrides
# Name hamming is not equal in packages Images and DSP and not in overrides
# Name convexhull is not equal in packages Images and Meshes and not in overrides
# Name derivative is not equal in packages Polynomials and TaylorSeries and not in overrides
# Name MultiPolygon is not equal in packages GeometryBasics and Meshes and not in overrides
# Name shape is not equal in packages Distributions and JuMP and not in overrides
# Name Normal is not equal in packages GeometryBasics and Distributions and not in overrides
# Name Fill is not equal in packages Images and FillArrays and not in overrides
# Name centered is not equal in packages GeometryBasics and Images and not in overrides
# Name bottom is not equal in packages CairoMakie and Meshes and not in overrides

end # module
