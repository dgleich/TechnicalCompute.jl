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

import Preferences
const makie_load_glmakie = Preferences.@load_preference("makie_load_glmakie", true)
const makie_backend = Preferences.@load_preference("makie_backend", "CairoMakie")
const _show_banner = Preferences.@load_preference("show_banner", true)

function set_makie_backend(backend)
  if !(backend in ["CairoMakie", "GLMakie"])
    throw(ArgumentError("Backend must be either \"CairoMakie\" or \"GLMakie\", not \"$backend\""))
  end
  Preferences.@set_preferences!("makie_backend", backend)
  @info("Makie backend set to $(backend); restart your Julia session for this change to take effect!")
end

function set_makie_load_glmakie(value::Bool)
  Preferences.@set_preferences!("makie_load_glmakie", value)
  @info("TechnicalCompute $(value ? "_will_" : "will _not_") load GLMakie; restart your Julia session for this change to take effect!")
end

function set_show_banner(value::Bool)
  Preferences.@set_preferences!("show_banner", value)
  @info("TechnicalCompute $(value ? "_will_" : "will _not_") show the banner; restart your Julia session for this change to take effect!")
end

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
"OnlineStats", 
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
"Serde",
"EzXML", 
# Files
"FileIO",
"HTTP",
"JLD2",
"HDF5",
"MAT",
"BSON",
"NIfTI", # used in Makie demos... 
"CodecBzip2",
"CodecLz4",
"CodecXz",
"CodecZlib",
"CodecZstd",
"ZipFile",
"TranscodingStreams",
"LibSndFile",
# Graphs
# "GraphIO", # removed pending issue with precompiling circular dependencies. 
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
# Parser Tools
# "ParserCombinator", # too many overlaps, too specialized... 
# Signals
"DSP", 
"SampledSignals",
# Plot Tools
"CairoMakie",
"Observables",
"LaTeXStrings", 
"Latexify",
# Linear Algebra
"Arpack",
"LinearMaps",
"Krylov",
"MatrixMarket",
"SuiteSparseMatrixCollection",
# Tensors
"ITensors",
"ITensorMPS",
"ITensorNetworks",
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
"OptimTestProblems",
"Optim",
"NonlinearSolve",
"LsqFit", 
"Tulip",
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
# ParserCombinator has too many overlapping names.
@reexport import ParserCombinator
# LineSearches.jl seems to be on the way out, replaced by LineSearch included in NonlinearSolve.jl
@reexport import LineSearches
# ITensors is a big ecosystem with a lot of overlapping names
# @reexport import ITensors
const IT = ITensors
export IT 

# these are specialized packages... 
@reexport import UnicodePlots
@reexport import PGFPlotsX
@reexport import NaNMath

# Handle CairoMakie vs. GLMakie... 
if get(()->"true", ENV, "JULIA_TECHNICALCOMPUTE_USE_GLMAKIE") == "true" && makie_load_glmakie
  @reexport import GLMakie 
end 

# const logo="""
#                _
#    _       _ _(_)_  ___      |  "Batteries included" 
#   (_)     | (_) (_)|_ _|     |  
#    _ _   _| |_  __ _| | __   |  
#   | | | | | | |/ _` | |/ _\\  |  Version $(pkgversion(TechnicalCompute)) 
#   | | |_| | | | (_| | | (_   |  running on Julia $(VERSION)
#  _/ |\\__'_|_|_|\\__'_|_|\\__/  |  
# |__/                         |
# """
# const logo="""
#                _
#    _       _ _(_)_  ____ ____     |  "Batteries included" 
#   (_)     | (_) (_)|_ _ / __ \\    |  
#    _ _   _| |_  __ _| |/ /  \\_|   |  
#   | | | | | | |/ _` | | |    _    |  Version $(pkgversion(TechnicalCompute)) 
#   | | |_| | | | (_| | |\\ \\__/ |   |  running via Julia $(VERSION)
#  _/ |\\__'_|_|_|\\__'_|_| \\____/    |  
# |__/                              |
# """
const logo="""
                             \x1b[32m_\x1b[0m
 _____         _           \x1b[31m_\x1b[32m(_)\x1b[35m_\x1b[0m       _ _____                             _       
|_   _|       | |         \x1b[31m(_) \x1b[35m(_)\x1b[0m     | /  __ \\ ...batteries included...  | |      
  | | ___  ___| |__  _ __  _  ___ __ _| | /  \\/ ___  _ __ ___  _ __  _   _| |_ ___ 
  | |/ _ \\/ __| '_ \\| '_ \\| |/ __/ _` | | |    / _ \\| '_ ` _ \\| '_ \\| | | | __/ _ \\
  | |  __/ (__| | | | | | | | (_| (_| | | \\__/\\ (_) | | | | | | |_) | |_| | ||  __/
  |_|\\___|\\___|_| |_|_| |_|_|\\___\\__'_|_|\\____/\\___/|_| |_| |_| '__/ \\__'_|\\__\\___|
      \x1b[32m,\x1b[35m|           |\x1b[32m,\x1b[35m|           |\x1b[32m,\x1b[35m|           |\x1b[32m,\x1b[35m|           |\x1b[0m| |  Version $(pkgversion(TechnicalCompute))
      \x1b[32m'\x1b[35m|___________|\x1b[32m'\x1b[35m|___________|\x1b[32m'\x1b[35m|___________|\x1b[32m'\x1b[35m|___________|\x1b[0m|_|  on Julia $(VERSION) 
"""       
function __init__()
  if isinteractive() && get(ENV, "JULIA_TECHNICALCOMPUTE_SHOW_BANNER", "1") != "0" && _show_banner
    println(logo)
  end 
  # the default is for GLMakie to activate because
  # we loaded it last... 
  # so we need to load CairoMakie if that's what we want
  if makie_backend == "CairoMakie" 
    @assert(makie_backend == "CairoMakie")
    CairoMakie.activate!()
  end
end 

include("overrides.jl")
include("overrides-custom.jl")
include("compile.jl")

# this one is bizarre, since Meshes exports a type with the same name as
# DelaunayTriangulation, we need to get a function from the module to get 
# the module, 
# I found this here: https://stackoverflow.com/questions/38819327/given-a-function-object-how-do-i-find-its-name-and-module

# DelaunayTriangulation = typeof(triangulate).name.module
# export DelaunayTriangulation

end # module
