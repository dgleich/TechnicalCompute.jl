module TechnicalCompute

using Reexport

packages = [
"SparseArrays",
"Random",
"FFTW",
"Combinatorics",
"LinearAlgebra",
"DelimitedFiles",
"Printf",
"Statistics",
"StatsBase",
"DataStructures",
"KahanSummation",
"BenchmarkTools",
"SpecialFunctions",
"FileIO",
"IterTools",
"Polynomials",
"TaylorSeries",
"TestImages",
"StaticArrays",
"JSON",
"DSP",
"Roots",
"JuMP",
"Plots",
"MultivariateStats",
"DataFrames",
"CSV",
"DifferentialEquations",
"HDF5",
"Images",
"LightGraphs",
"SuiteSparse",
"Arpack",
"Flux",
"NearestNeighbors",
"GLPK",
"Clp",
"QuadGK",

]

# todo wrote code to auto re-export from the list.


#=

This will write Project.toml
for p in packages
  id = Base.identify_package(p)
  println("$(id.name) = \"$(id.uuid)\"")
end

=#


@reexport using SparseArrays
@reexport using Random
@reexport using FFTW # need
@reexport using Combinatorics
@reexport using LinearAlgebra
@reexport using DelimitedFiles
@reexport using Printf
@reexport using Statistics
@reexport using StatsBase
@reexport using DataStructures
@reexport using KahanSummation
@reexport using BenchmarkTools
@reexport using SpecialFunctions
@reexport using FileIO
@reexport using IterTools
@reexport using Polynomials
@reexport using TaylorSeries
@reexport using TestImages

@reexport using StaticArrays
@reexport using JSON
@reexport using DSP
@reexport using Roots
@reexport using JuMP
@reexport using Plots
@reexport using MultivariateStats
@reexport using DataFrames
@reexport using CSV
@reexport using DifferentialEquations
@reexport using HDF5
@reexport using Images
@reexport using LightGraphs
@reexport using SuiteSparse
@reexport using Arpack
@reexport using Flux
@reexport using NearestNeighbors
@reexport using QuadGK
@reexport using GLPK
@reexport using Clp
@reexport using Ipopt
end # module
