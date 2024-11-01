const overrides = Set{Symbol}()

## :@variables
# Showing duplicate methods for @variables in packages Module[JuMP, Symbolics]
# Methods for @variables in package JuMP
# var"@variables"(__source__::LineNumberNode, __module__::Module, model, block) @ JuMP ~/.julia/packages/JuMP/-----/src/macros/@variable.jl:354
# Methods for @variables in package Symbolics
# var"@variables"(__source__::LineNumberNode, __module__::Module, xs...) @ Symbolics ~/.julia/packages/Symbolics/-----/src/variable.jl:464

# Most demos for JuMP use @variable whereas most for Symbolics use @variables, so we'll go with that.
# This is just a copy-paste of @variables from Symbolics.jl ... need to figure out how to actaully call it... 
@doc (@doc Symbolics.var"@variables")
macro variables(expr...)
  #:(Symbolics.@variables $(esc.(expr...)))
  esc(Symbolics._parse_vars(:variables, Real, expr))
  #:(Symbolics.@variables expr...)
end
export @variables 
push!(overrides, Symbol("@variables"))

## :AdaMax
# Showing duplicate methods for AdaMax in packages Module[Flux, Optim]
# Methods for Flux.Optimise.AdaMax in package Core
# Flux.Optimise.AdaMax() @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:265
# Flux.Optimise.AdaMax(eta, beta, epsilon, state) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:260
# Flux.Optimise.AdaMax(eta::Float64, beta::Tuple{Float64, Float64}, epsilon::Float64, state::IdDict{Any, Any}) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:260
# Flux.Optimise.AdaMax(η::Real) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:265
# Flux.Optimise.AdaMax(η::Real, β::Tuple) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:265
# Flux.Optimise.AdaMax(η::Real, β::Tuple, state::IdDict) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:266
# Flux.Optimise.AdaMax(η::Real, β::Tuple, ϵ::Real) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:265
# Methods for Optim.AdaMax in package Core
# Optim.AdaMax(; alpha, beta_mean, beta_var, epsilon) @ Optim ~/.julia/packages/Optim/-----/src/multivariate/solvers/first_order/adamax.jl:21
# Optim.AdaMax(α::T, β₁::T, β₂::T, ϵ::T, manifold::Tm) where {T, Tm} @ Optim ~/.julia/packages/Optim/-----/src/multivariate/solvers/first_order/adamax.jl:15

# Flux is the ML package so it gets AdaMax.
AdaMax = Flux.AdaMax
export AdaMax
push!(overrides, :AdaMax)

## :Adam
# Showing duplicate methods for Adam in packages Module[Flux, Optim]
# Methods for Flux.Optimise.Adam in package Core
# Flux.Optimise.Adam() @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:173
# Flux.Optimise.Adam(eta, beta, epsilon, state) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:168
# Flux.Optimise.Adam(eta::Float64, beta::Tuple{Float64, Float64}, epsilon::Float64, state::IdDict{Any, Any}) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:168
# Flux.Optimise.Adam(η::Real) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:173
# Flux.Optimise.Adam(η::Real, β::Tuple) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:173
# Flux.Optimise.Adam(η::Real, β::Tuple, state::IdDict) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:174
# Flux.Optimise.Adam(η::Real, β::Tuple, ϵ::Real) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/optimisers.jl:173
# Methods for Optim.Adam in package Core
# Optim.Adam(; alpha, beta_mean, beta_var, epsilon) @ Optim ~/.julia/packages/Optim/-----/src/multivariate/solvers/first_order/adam.jl:21
# Optim.Adam(α::T, β₁::T, β₂::T, ϵ::T, manifold::Tm) where {T, Tm} @ Optim ~/.julia/packages/Optim/-----/src/multivariate/solvers/first_order/adam.jl:14

# Flux is the ML package so it gets Adam.
Adam = Flux.Adam
export Adam
push!(overrides, :Adam)

## :Axis
# Showing duplicate methods for Axis in packages Module[AxisArrays, CairoMakie, Images]
# Methods for AxisArrays.Axis in package Core
# Methods for Makie.Axis in package Core
# (::Type{T})(args...; kwargs...) where T<:Block @ Makie ~/.julia/packages/Makie/-----/src/makielayout/blocks.jl:236
# Makie.Axis(parent::Union{Nothing, Figure, Scene}, layoutobservables::LayoutObservables{GridLayout}, blockscene::Scene) @ Makie ~/.julia/packages/Makie/-----/src/makielayout/blocks.jl:50
@doc (@doc Makie.Axis) 
Axis = Makie.Axis
@doc (@doc Images.Axis)
ArrayAxis = Images.Axis
export Axis, ArrayAxis 
push!(overrides, :Axis)

## :BSpline
# Showing duplicate methods for BSpline in packages Module[DelaunayTriangulation, Interpolations]
# Methods for DelaunayTriangulation.BSpline in package Core
# DelaunayTriangulation.BSpline(control_points, knots, cache, lookup_table, orientation_markers) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/bspline.jl:36
# DelaunayTriangulation.BSpline(control_points::Vector{Tuple{Float64, Float64}}, knots::Vector{Int64}, cache::Vector{Tuple{Float64, Float64}}, lookup_table::Vector{Tuple{Float64, Float64}}, orientation_markers::Vector{Float64}) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/bspline.jl:36
# DelaunayTriangulation.BSpline(control_points::Vector{Tuple{Float64, Float64}}; degree, lookup_steps, kwargs...) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/bspline.jl:58
# Methods for Interpolations.BSpline in package Core
# Interpolations.BSpline() @ Interpolations ~/.julia/packages/Interpolations/-----/src/b-splines/b-splines.jl:25
# Interpolations.BSpline(::Type{D}) where D<:Degree @ Interpolations ~/.julia/packages/Interpolations/-----/src/b-splines/b-splines.jl:24
# Interpolations.BSpline(degree::D) where D<:Degree @ Interpolations ~/.julia/packages/Interpolations/-----/src/b-splines/b-splines.jl:21

@doc (@doc Interpolations.BSpline)
BSpline() = Interpolations.BSpline()
BSpline(::Type{D}) where D <: Interpolations.Degree = Interpolations.BSpline(Type{D})
BSpline(d::D) where D <: Interpolations.Degree = Interpolations.BSpline(D) 
@doc (@doc DelaunayTriangulation.BSpline)
BSpline(control_points::Vector{Tuple{Float64, Float64}}, knots::Vector{Int64}, cache::Vector{Tuple{Float64, Float64}}, lookup_table::Vector{Tuple{Float64, Float64}}, orientation_markers::Vector{Float64}) = DelaunayTriangulation.BSpline(control_points, knots, cache, lookup_table, orientation_markers) 
BSpline(control_points::Vector{Tuple{Float64, Float64}}; kwargs...) = DelaunayTriangulation.BSpline(control_points; kwargs...)
export BSpline
push!(overrides, :BSpline) 

## :Bisection
# Showing duplicate methods for Bisection in packages Module[DifferentialEquations, NonlinearSolve, Roots]
# Methods for SimpleNonlinearSolve.Bisection in package Core
# SimpleNonlinearSolve.Bisection(; exact_left, exact_right) @ SimpleNonlinearSolve ~/.julia/packages/SimpleNonlinearSolve/-----/src/bracketing/bisection.jl:17
# SimpleNonlinearSolve.Bisection(exact_left, exact_right) @ SimpleNonlinearSolve ~/.julia/packages/SimpleNonlinearSolve/-----/src/bracketing/bisection.jl:18
# SimpleNonlinearSolve.Bisection(exact_left::Bool, exact_right::Bool) @ SimpleNonlinearSolve ~/.julia/packages/SimpleNonlinearSolve/-----/src/bracketing/bisection.jl:18
# Methods for Roots.Bisection in package Core
# Roots.Bisection() @ Roots ~/.julia/packages/Roots/-----/src/Bracketing/bisection.jl:28
@doc (@doc Roots.Bisection)
Bisection() = Roots.Bisection()
@doc (@doc SimpleNonlinearSolve.Bisection)
Bisection(left,right) = SimpleNonlinearSolve.Bisection(left,right)
#Bisection(;exact_left,exact_right) = SimpleNonlinearSolve.Bisection(;exact_left,exact_right)
export Bisection 
push!(overrides, :Bisection)

## :Brent
# Showing duplicate methods for Brent in packages Module[DifferentialEquations, NonlinearSolve, Optim]
# Methods for SimpleNonlinearSolve.Brent in package Core
# SimpleNonlinearSolve.Brent() @ SimpleNonlinearSolve ~/.julia/packages/SimpleNonlinearSolve/-----/src/bracketing/brent.jl:6
# Methods for Optim.Brent in package Core
# Optim.Brent() @ Optim ~/.julia/packages/Optim/-----/src/univariate/solvers/brent.jl:19

@doc (@doc NonlinearSolve.Brent)
Brent = NonlinearSolve.Brent
export Brent
push!(overrides, :Brent)

## :Categorical
# Showing duplicate methods for Categorical in packages Module[CairoMakie, Distributions]
# Methods for Makie.Categorical in package Core
# Makie.Categorical(values) @ Makie ~/.julia/packages/Makie/-----/src/colorsampler.jl:229
# Methods for Distributions.Categorical{P} where P<:Real in package Core
# (Distributions.Categorical{P} where P<:Real)(k::Integer; check_args) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/categorical.jl:37
# (Distributions.Categorical{P} where P<:Real)(p::AbstractVector{P}; check_args) where P<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/categorical.jl:34
# (Distributions.Categorical{P} where P<:Real)(probabilities::Real...; check_args) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/categorical.jl:42
@doc (@doc Distributions.Categorical)
Categorical = Distributions.Categorical
@doc (@doc Makie.Categorical)
CategoricalColormap = Makie.Categorical
export Categorical, CategoricalColormap
push!(overrides, :Categorical)

## :ComplexVariable
# Showing duplicate methods for ComplexVariable in packages Module[Convex, JuMP]
# Methods for Convex.ComplexVariable in package Core
# Convex.ComplexVariable() @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:276
# Convex.ComplexVariable(m::Int64, args...) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:287
# Convex.ComplexVariable(m::Int64, n::Int64, args...) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:289
# Convex.ComplexVariable(set::Symbol, sets::Symbol...) @ Convex ~/.julia/packages/Convex/-----/src/deprecations.jl:123
# Convex.ComplexVariable(size::Tuple{Int64, Int64}) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:276
# Convex.ComplexVariable(size::Tuple{Int64, Int64}, set::Symbol, sets::Symbol...) @ Convex ~/.julia/packages/Convex/-----/src/deprecations.jl:102
# Methods for JuMP.ComplexVariable in package Core
# JuMP.ComplexVariable(info::VariableInfo{S, T, U, V}) where {S, T, U, V} @ JuMP ~/.julia/packages/JuMP/-----/src/variables.jl:2260

@doc (@doc JuMP.ComplexVariable)
ComplexVariable = JuMP.ComplexVariable 
export ComplexVariable
push!(overrides, :ComplexVariable)

## :EllipticalArc
# Showing duplicate methods for EllipticalArc in packages Module[CairoMakie, DelaunayTriangulation]
# Methods for Makie.EllipticalArc in package Core
# Makie.EllipticalArc(c, r1, r2, angle, a1, a2) @ Makie ~/.julia/packages/Makie/-----/src/bezier.jl:75
# Makie.EllipticalArc(c::Point{2, Float64}, r1::Float64, r2::Float64, angle::Float64, a1::Float64, a2::Float64) @ Makie ~/.julia/packages/Makie/-----/src/bezier.jl:75
# Makie.EllipticalArc(cx, cy, r1, r2, angle, a1, a2) @ Makie ~/.julia/packages/Makie/-----/src/bezier.jl:83
# Makie.EllipticalArc(x1, y1, x2, y2, rx, ry, ϕ, largearc::Bool, sweepflag::Bool) @ Makie ~/.julia/packages/Makie/-----/src/bezier.jl:546
# Methods for DelaunayTriangulation.EllipticalArc in package Core
# DelaunayTriangulation.EllipticalArc(center, horz_radius, vert_radius, rotation_scales, start_angle, sector_angle, first, last) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/ellipticalarc.jl:26
# DelaunayTriangulation.EllipticalArc(center::Tuple{Float64, Float64}, horz_radius::Float64, vert_radius::Float64, rotation_scales::Tuple{Float64, Float64}, start_angle::Float64, sector_angle::Float64, first::Tuple{Float64, Float64}, last::Tuple{Float64, Float64}) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/ellipticalarc.jl:26
# DelaunayTriangulation.EllipticalArc(p, q, c, α, β, θ°; positive) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/ellipticalarc.jl:47
@doc (@doc Makie.EllipticalArc)
EllipticalArc(c::Point{2, Float64}, r1::Float64, r2::Float64, angle::Float64, a1::Float64, a2::Float64)= Makie.EllipticalArc(c, r1, r2, angle, a1, a2)
EllipticalArc(x1, y1, x2, y2, rx, ry, ϕ, largearc::Bool, sweepflag::Bool) = Makie.EllipticalArc(x1, y1, x2, y2, rx, ry, ϕ, largearc, sweepflag)
@doc (@doc DelaunayTriangulation.EllipticalArc)
EllipticalArc(p, q, c, α, β, θ°; kwargs...) = DelaunayTriangulation.EllipticalArc(p, q, c, α, β, θ°; kwargs...)
export EllipticalArc
push!(overrides, :EllipticalArc)

## :Fill
# Showing duplicate methods for Fill in packages Module[FillArrays, Images]
# Methods for FillArrays.Fill in package Core
# FillArrays.Fill(x, sz::Union{Infinities.Infinity, Integer}...) @ InfiniteArrays ~/.julia/packages/InfiniteArrays/-----/src/infarrays.jl:59
# FillArrays.Fill(x::T, sz::NTuple{N, Any}) where {T, N} @ FillArrays ~/.julia/packages/FillArrays/-----/src/FillArrays.jl:138
# FillArrays.Fill(x::T, sz::Vararg{Integer, N}) where {T, N} @ FillArrays ~/.julia/packages/FillArrays/-----/src/FillArrays.jl:136
# Methods for ImageFiltering.Fill in package Core
# ImageFiltering.Fill(value, kernel) @ ImageFiltering ~/.julia/packages/ImageFiltering/-----/src/border.jl:547
# ImageFiltering.Fill(value, lo::AbstractVector, hi::AbstractVector) @ ImageFiltering ~/.julia/packages/ImageFiltering/-----/src/border.jl:527
# ImageFiltering.Fill(value::T) where T @ ImageFiltering ~/.julia/packages/ImageFiltering/-----/src/border.jl:485
# ImageFiltering.Fill(value::T, ::Tuple{}) where T @ ImageFiltering ~/.julia/packages/ImageFiltering/-----/src/border.jl:528
# ImageFiltering.Fill(value::T, both::NTuple{N, Int64}) where {T, N} @ ImageFiltering ~/.julia/packages/ImageFiltering/-----/src/border.jl:525
# ImageFiltering.Fill(value::T, inds::NTuple{N, AbstractUnitRange}) where {T, N} @ ImageFiltering ~/.julia/packages/ImageFiltering/-----/src/border.jl:529
# ImageFiltering.Fill(value::T, lo::NTuple{N, Int64}, hi::NTuple{N, Int64}) where {T, N} @ ImageFiltering ~/.julia/packages/ImageFiltering/-----/src/border.jl:509
@doc (@doc ImageFiltering.Fill)
FillValue = ImageFiltering.Fill
@doc (@doc FillArrays.Fill)
FillArray = FillArrays.Fill 
export FillValue, FillArray
push!(overrides, :Fill)

## :Filters
# Showing duplicate methods for Filters in packages Module[DSP, HDF5]
# Methods for DSP.Filters in package Core
# Methods for HDF5.Filters in package Core
@doc (@doc DSP.Filters)
Filters = DSP.Filters
export Filters 
push!(overrides, :Filters)

## :Fixed
# Showing duplicate methods for Fixed in packages Module[CairoMakie, Images]
# Methods for GridLayoutBase.Fixed in package Core
# GridLayoutBase.Fixed(x) @ GridLayoutBase ~/.julia/packages/GridLayoutBase/-----/src/types.jl:160
# GridLayoutBase.Fixed(x::Float32) @ GridLayoutBase ~/.julia/packages/GridLayoutBase/-----/src/types.jl:160
# Methods for FixedPointNumbers.Fixed in package Core
# (::Type{<:FixedPoint})(x::AbstractChar) @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/-----/src/FixedPointNumbers.jl:60
# (::Type{T})(p::DomainSets.Point{<:Number}) where T<:Number @ DomainSets ~/.julia/packages/DomainSets/-----/src/domains/point.jl:13
# (::Type{T})(x::AbstractChar) where T<:Union{AbstractChar, Number} @ Base char.jl:50
# (::Type{T})(x::AbstractGray) where T<:Real @ ColorTypes ~/.julia/packages/ColorTypes/-----/src/conversions.jl:115
# (::Type{T})(x::Base.TwicePrecision) where T<:Number @ Base twiceprecision.jl:265
# (::Type{T})(x::T) where T<:Number @ Core boot.jl:900
# (::Type{X})(x::Base.TwicePrecision) where X<:FixedPoint @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/-----/src/FixedPointNumbers.jl:64
# (::Type{X})(x::Complex) where X<:FixedPoint @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/-----/src/FixedPointNumbers.jl:63
# (::Type{X})(x::Number) where X<:FixedPoint @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/-----/src/FixedPointNumbers.jl:58
# (::Type{X})(x::X) where X<:FixedPoint @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/-----/src/FixedPointNumbers.jl:57
@doc (@doc FixedPointNumbers.Fixed)
Fixed = FixedPointNumbers.Fixed
@doc (@doc Makie.Fixed)
FixedSize = Makie.Fixed
export Fixed, FixedSize 
push!(overrides, :Fixed)

## :Flat
# Showing duplicate methods for Flat in packages Module[Interpolations, Optim]
# Methods for Interpolations.Flat in package Core
# (::Type{BC})() where BC<:BoundaryCondition @ Interpolations ~/.julia/packages/Interpolations/-----/src/Interpolations.jl:111
# Interpolations.Flat(::Type{GT}) where GT<:GridType @ Interpolations ~/.julia/packages/Interpolations/-----/src/Interpolations.jl:114
# Interpolations.Flat(gt::GT) where GT<:Union{Nothing, GridType} @ Interpolations ~/.julia/packages/Interpolations/-----/src/Interpolations.jl:94
# Methods for Optim.Flat in package Core
# Optim.Flat() @ Optim ~/.julia/packages/Optim/-----/src/Manifolds.jl:56

@doc (@doc Interpolations.Flat)
Flat(x::Interpolations.GridType) = Interpolations.Flat(x)
Flat(x::Nothing) = Interpolations.Flat(x)
Float(::Type{GT}) where GT <: Interpolations.GridType = Interpolations.Flat(Type{GT})
@doc (@doc Optim.Flat)
Flat() = Optim.Flat()
export Flat
push!(overrides, :Flat)

## :FunctionMap
# Showing duplicate methods for FunctionMap in packages Module[DifferentialEquations, LinearMaps]
# Methods for OrdinaryDiffEqFunctionMap.FunctionMap in package Core
# OrdinaryDiffEqFunctionMap.FunctionMap(; scale_by_time) @ OrdinaryDiffEqFunctionMap ~/.julia/packages/OrdinaryDiffEqFunctionMap/-----/src/algorithms.jl:2
# Methods for LinearMaps.FunctionMap in package Core
@doc (@doc LinearMaps.FunctionMap)
FunctionMap = LinearMaps.FunctionMap
export FunctionMap
push!(overrides, :FunctionMap)

## :Graph
# Showing duplicate methods for Graph in packages Module[DelaunayTriangulation, Graphs]
# Methods for DelaunayTriangulation.Graph in package Core
# DelaunayTriangulation.Graph(vertices::Set{I}, edges::Set{Tuple{I, I}}, neighbours::Dict{I, Set{I}}) where I @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/triangulation/graph.jl:22
# Methods for SimpleGraph in package Core
# SimpleGraph() @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/simplegraph.jl:46
# SimpleGraph(::Type{T}) where T<:Integer @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/simplegraph.jl:62
# SimpleGraph(adjmx::AbstractMatrix) @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/simplegraph.jl:87
# SimpleGraph(edge_list::Array{Graphs.SimpleGraphs.SimpleEdge{T}, 1}) where T<:Integer @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/simplegraph.jl:219
# SimpleGraph(g::AbstractGraph{T}) where T @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/simplegraph.jl:285
# SimpleGraph(g::MetaGraphs.MetaGraph) @ MetaGraphs ~/.julia/packages/MetaGraphs/-----/src/metagraph.jl:50
# SimpleGraph(g::SimpleDiGraph) @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/simplegraph.jl:150
# SimpleGraph(g::SimpleGraph) @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/simplegraph.jl:123
# SimpleGraph(g::SimpleWeightedGraph) @ SimpleWeightedGraphs ~/.julia/packages/SimpleWeightedGraphs/-----/src/simpleweightedgraph.jl:151
# SimpleGraph(n::T) where T<:Integer @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/simplegraph.jl:43
# SimpleGraph(ne, fadjlist::Array{Vector{T}, 1}) where T @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/simplegraph.jl:18
# SimpleGraph(nv::T, ne::Integer; rng, seed) where T<:Integer @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/generators/randgraphs.jl:46
# SimpleGraph(nvg::Integer, neg::Integer, edgestream::Channel) @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/generators/randgraphs.jl:1365
# SimpleGraph(nvg::Integer, neg::Integer, sbm::StochasticBlockModel; rng, seed) @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/generators/randgraphs.jl:1382

Graph = Graphs.Graph 
export Graph
push!(overrides, :Graph)

## :GroupBy
# Showing duplicate methods for GroupBy in packages Module[OnlineStats, Transducers]
# Methods for OnlineStatsBase.GroupBy in package Core
# OnlineStatsBase.GroupBy(T::Type, stat::O) where O<:OnlineStat @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:439
# OnlineStatsBase.GroupBy(value::OrderedDict{T, O}, init::O, n::Int64) where {T, S, O<:OnlineStat{S}} @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:435
# Methods for Transducers.GroupBy in package Core
# Transducers.GroupBy(key, rf) @ Transducers ~/.julia/packages/Transducers/-----/src/groupby.jl:111
# Transducers.GroupBy(key, xf::Transducer) @ Transducers ~/.julia/packages/Transducers/-----/src/groupby.jl:106
# Transducers.GroupBy(key, xf::Transducer, step) @ Transducers ~/.julia/packages/Transducers/-----/src/groupby.jl:106
# Transducers.GroupBy(key, xf::Transducer, step, init) @ Transducers ~/.julia/packages/Transducers/-----/src/groupby.jl:106
# Transducers.GroupBy(key::K, rf::R, init::T) where {K, R, T} @ Transducers ~/.julia/packages/Transducers/-----/src/groupby.jl:101

@doc (@doc OnlineStats.GroupBy)
GroupBy(T::Type, stat::OnlineStat) = OnlineStats.GroupBy(T, stat)
GroupBy(value::OrderedDict{T, O}, init::OnlineStat, n::Int64) where {T, S, O<:OnlineStat{S}} = OnlineStats.GroupBy(value, init, n)
@doc (@doc Transducers.GroupBy)
GroupBy(key, xf::Transducer) = Transducers.GroupBy(key, xf)
GroupBy(key, xf::Transducer, step) = Transducers.GroupBy(key, xf, step)
GroupBy(key, xf::Transducer, step, init) = Transducers.GroupBy(key, xf, step, init)
export GroupBy
push!(overrides, :GroupBy)

## :Hist
# Showing duplicate methods for Hist in packages Module[CairoMakie, OnlineStats]
# Methods for Plot{Makie.hist} in package Core
# (Plot{Func})(user_args::Tuple, user_attributes::Dict) where Func @ Makie ~/.julia/packages/Makie/-----/src/interfaces.jl:260
# Methods for OnlineStats.Hist in package Core
# OnlineStats.Hist(edges::R, T::Type; left, closed) where R<:(AbstractVector) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/histograms.jl:64
# OnlineStats.Hist(edges::R; ...) where R<:(AbstractVector) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/histograms.jl:64

# for Makie, we just use the hist function.
Hist = OnlineStats.Hist
export Hist
push!(overrides, :Hist) 

## :Length
# Showing duplicate methods for Length in packages Module[Measures, StaticArrays]
# Methods for Measures.Length in package Core
# Measures.Length(unit::Symbol, x::T) where T @ Measures ~/.julia/packages/Measures/-----/src/length.jl:7
# Methods for StaticArrays.Length in package Core
# StaticArrays.Length(::Size{S}) where S @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/traits.jl:40
# StaticArrays.Length(::Type{A}) where A<:AbstractArray @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/traits.jl:38
# StaticArrays.Length(L::Int64) @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/traits.jl:39
# StaticArrays.Length(a::AbstractArray) @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/traits.jl:37
# StaticArrays.Length(x::StaticArrays.Args) @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/convert.jl:9

# Neither of these seems to be the right one to get Length. 
push!(overrides, :Length)

## :Line
# Showing duplicate methods for Line in packages Module[GeometryBasics, Interpolations]
# Methods for GeometryBasics.Line in package Core
# (::Type{<:GeometryBasics.Ngon{Dim, T, N1, P} where {Dim, T, P}})(p0::P, points::Vararg{P, N2}) where {Dim, T, P<:AbstractPoint{Dim, T}, N1, N2} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/basic_types.jl:65
# Methods for Interpolations.Line in package Core
# (::Type{BC})() where BC<:BoundaryCondition @ Interpolations ~/.julia/packages/Interpolations/-----/src/Interpolations.jl:111
# Interpolations.Line(::Type{GT}) where GT<:GridType @ Interpolations ~/.julia/packages/Interpolations/-----/src/Interpolations.jl:114
# Interpolations.Line(gt::GT) where GT<:Union{Nothing, GridType} @ Interpolations ~/.julia/packages/Interpolations/-----/src/Interpolations.jl:96

# you can call Line for GeometryBasics by calling Polytope with two arguments. 
# I'm not sure this is the right call, but let's go with it. 
@doc (@doc Interpolations.Line)
Line = Interpolations.Line 
export Line 
push!(overrides, :Line)

## :Mesh
# Showing duplicate methods for Mesh in packages Module[CairoMakie, GeometryBasics]
# Methods for MakieCore.Mesh in package Core
# (Plot{Func})(user_args::Tuple, user_attributes::Dict) where Func @ Makie ~/.julia/packages/Makie/-----/src/interfaces.jl:260
# Methods for GeometryBasics.Mesh in package Core
# GeometryBasics.Mesh(elements::AbstractVector{<:Polytope{Dim, T}}) where {Dim, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/basic_types.jl:408
# GeometryBasics.Mesh(points::AbstractVector{<:AbstractPoint}, faces::AbstractVector{<:AbstractFace}) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/basic_types.jl:412
# GeometryBasics.Mesh(points::AbstractVector{<:AbstractPoint}, faces::AbstractVector{<:Integer}) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/basic_types.jl:417
# GeometryBasics.Mesh(points::AbstractVector{<:AbstractPoint}, faces::AbstractVector{<:Integer}, facetype) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/basic_types.jl:417
# GeometryBasics.Mesh(points::AbstractVector{<:AbstractPoint}, faces::AbstractVector{<:Integer}, facetype, skip) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/basic_types.jl:417
# GeometryBasics.Mesh(simplices::V) where {Dim, T<:Number, Element<:Polytope{Dim, T}, V<:AbstractVector{Element}} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/basic_types.jl:375
Mesh = GeometryBasics.Mesh 
export Mesh
push!(overrides, :Mesh)

## :Moments
# Showing duplicate methods for Moments in packages Module[Images, OnlineStats]
# Methods for HistogramThresholding.Moments in package Core
# HistogramThresholding.Moments() @ HistogramThresholding ~/.julia/packages/HistogramThresholding/-----/src/algorithms/moments.jl:92
# Methods for OnlineStatsBase.Moments in package Core
# OnlineStatsBase.Moments(; weight) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:505
# OnlineStatsBase.Moments(m::Vector{Float64}, weight::W, n::Int64) where W @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:501

# Stats gets Moments... Images needs Images.Moments. 
Moments = OnlineStats.Moments
export Moments
push!(overrides, :Moments)

## :Normal
# Showing duplicate methods for Normal in packages Module[Distributions, GeometryBasics]
# Methods for Distributions.Normal in package Core
# Distributions.Normal() @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:44
# Distributions.Normal(mu::Num, sigma::Num) @ Symbolics ~/.julia/packages/Symbolics/-----/src/wrapper-types.jl:158
# Distributions.Normal(mu::Num, sigma::Real) @ Symbolics ~/.julia/packages/Symbolics/-----/src/wrapper-types.jl:158
# Distributions.Normal(mu::Num, sigma::SymbolicUtils.Symbolic{<:Real}) @ Symbolics ~/.julia/packages/Symbolics/-----/src/wrapper-types.jl:158
# Distributions.Normal(mu::Real, sigma::Num) @ Symbolics ~/.julia/packages/Symbolics/-----/src/wrapper-types.jl:158
# Distributions.Normal(mu::Real, sigma::SymbolicUtils.Symbolic{<:Real}) @ Symbolics ~/.julia/packages/Symbolics/-----/src/wrapper-types.jl:158
# Distributions.Normal(mu::SymbolicUtils.Symbolic{<:Real}, sigma::Num) @ Symbolics ~/.julia/packages/Symbolics/-----/src/wrapper-types.jl:158
# Distributions.Normal(mu::SymbolicUtils.Symbolic{<:Real}, sigma::Real) @ Symbolics ~/.julia/packages/Symbolics/-----/src/wrapper-types.jl:158
# Distributions.Normal(mu::SymbolicUtils.Symbolic{<:Real}, sigma::SymbolicUtils.Symbolic{<:Real}) @ Symbolics ~/.julia/packages/Symbolics/-----/src/wrapper-types.jl:158
# Distributions.Normal(μ::Integer, σ::Integer; check_args) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:43
# Distributions.Normal(μ::Real) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:44
# Distributions.Normal(μ::Real, σ::Real; check_args) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:42
# Distributions.Normal(μ::T, σ::T; check_args) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:36
# Methods for GeometryBasics.Normal in package Core
# GeometryBasics.Normal() @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/interfaces.jl:100
# GeometryBasics.Normal(::Type{T}) where T @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/interfaces.jl:99
@doc (@doc Distributions.Normal)
Normal = Distributions.Normal
push!(overrides, :Normal)
NormalVector = GeometryBasics.Normal
export Normal, NormalVector 

## :Partition
# Showing duplicate methods for Partition in packages Module[Combinatorics, OnlineStats, Transducers]
# Methods for Combinatorics.Partition in package Core
# Combinatorics.Partition(x) @ Combinatorics ~/.julia/packages/Combinatorics/-----/src/youngdiagrams.jl:6
# Combinatorics.Partition(x::Vector{Int64}) @ Combinatorics ~/.julia/packages/Combinatorics/-----/src/youngdiagrams.jl:6
# Methods for OnlineStats.Partition in package Core
# OnlineStats.Partition(init::Function) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/partition.jl:22
# OnlineStats.Partition(init::Function, b::Int64) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/partition.jl:22
# OnlineStats.Partition(o::OnlineStat) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/partition.jl:23
# OnlineStats.Partition(o::OnlineStat, b::Int64) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/partition.jl:23
# OnlineStats.Partition(parts::Array{Pair{Tuple{Int64, Int64}, O}, 1}, b::Int64, init::I, n::Int64) where {T, I, O<:OnlineStat{T}} @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/partition.jl:17
# Methods for Transducers.Partition in package Core
# Transducers.Partition(size, step, flush) @ Transducers ~/.julia/packages/Transducers/-----/src/library.jl:820
# Transducers.Partition(size, step; flush) @ Transducers ~/.julia/packages/Transducers/-----/src/library.jl:827
# Transducers.Partition(size; step, flush) @ Transducers ~/.julia/packages/Transducers/-----/src/library.jl:828
Partition = Transducers.Partition
export Partition
push!(overrides, :Partition)

## :Series
# Showing duplicate methods for Series in packages Module[CairoMakie, OnlineStats]
# Methods for Plot{Makie.series} in package Core
# (Plot{Func})(user_args::Tuple, user_attributes::Dict) where Func @ Makie ~/.julia/packages/Makie/-----/src/interfaces.jl:260
# Methods for OnlineStatsBase.Series in package Core
# OnlineStatsBase.Series(; t...) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:617
# OnlineStatsBase.Series(stats::T) where T @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:614
# OnlineStatsBase.Series(t::OnlineStat...) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:616

# makie Series is just a series plot, via `series`
Series = OnlineStats.Series
export Series
push!(overrides, :Series)

## :Sphere
# Showing duplicate methods for Sphere in packages Module[CairoMakie, GeometryBasics, Optim]
# Methods for GeometryBasics.Sphere in package Core
# (HyperSphere{N})(p::Point{N, T}, number) where {N, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/spheres.jl:26
# Methods for Optim.Sphere in package Core
# Optim.Sphere() @ Optim ~/.julia/packages/Optim/-----/src/Manifolds.jl:67

Sphere = GeometryBasics.Sphere
export Sphere
push!(overrides, :Sphere)

## :Trace
# Showing duplicate methods for Trace in packages Module[OnlineStats, ReinforcementLearning]
# Methods for OnlineStats.Trace in package Core
# OnlineStats.Trace(o::OnlineStat) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/trace.jl:25
# OnlineStats.Trace(o::OnlineStat, b::Int64) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/trace.jl:25
# OnlineStats.Trace(parts::Array{Pair{Tuple{Int64, Int64}, O}, 1}, b::Int64, n::Int64) where {T, O<:OnlineStat{T}} @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/trace.jl:21
# Methods for ReinforcementLearningTrajectories.Trace in package Core
# ReinforcementLearningTrajectories.Trace(x::T) where T<:AbstractArray @ ReinforcementLearningTrajectories ~/.julia/packages/ReinforcementLearningTrajectories/-----/src/traces.jl:36

@doc (@doc OnlineStats.Trace)
Trace(o::OnlineStat) = OnlineStats.Trace(o)
Trace(o::OnlineStat, b::Int64) = OnlineStats.Trace(o, b)
Trace(parts::Array{Pair{Tuple{Int64, Int64}, O}, 1}, b::Int64, n::Int64) where {T, O<:OnlineStat{T}} = OnlineStats.Trace(parts, b, n)
@doc (@doc ReinforcementLearning.Trace)
Trace(x::T) where T<:AbstractArray = ReinforcementLearning.Trace(x)
export Trace
push!(overrides, :Trace)
## :Variable
# Showing duplicate methods for Variable in packages Module[Convex, Symbolics]
# Methods for Convex.Variable in package Core
# Convex.Variable() @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:228
# Convex.Variable(m::Int64, args...) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:261
# Convex.Variable(m::Int64, n::Int64, args...) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:259
# Convex.Variable(m::Int64, set::Symbol, sets::Symbol...) @ Convex ~/.julia/packages/Convex/-----/src/deprecations.jl:94
# Convex.Variable(set::Symbol, sets::Symbol...) @ Convex ~/.julia/packages/Convex/-----/src/deprecations.jl:98
# Convex.Variable(sign::Convex.Sign) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:263
# Convex.Variable(sign::Convex.Sign, set::Symbol, sets::Symbol...) @ Convex ~/.julia/packages/Convex/-----/src/deprecations.jl:86
# Convex.Variable(sign::Convex.Sign, vartype::Convex.VarType) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:263
# Convex.Variable(size::Tuple{Int64, Int64}) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:228
# Convex.Variable(size::Tuple{Int64, Int64}, set::Symbol, sets::Symbol...) @ Convex ~/.julia/packages/Convex/-----/src/deprecations.jl:90
# Convex.Variable(size::Tuple{Int64, Int64}, sign::Convex.Sign) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:228
# Convex.Variable(size::Tuple{Int64, Int64}, sign::Convex.Sign, set::Symbol, sets::Symbol...) @ Convex ~/.julia/packages/Convex/-----/src/deprecations.jl:61
# Convex.Variable(size::Tuple{Int64, Int64}, sign::Convex.Sign, vartype::Convex.VarType) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:228
# Convex.Variable(size::Tuple{Int64, Int64}, vartype::Convex.VarType) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:255
# Convex.Variable(vartype::Convex.VarType) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:267
# Methods for Symbolics.Variable in package Core
# Symbolics.Variable(s, i...) @ Symbolics ~/.julia/packages/Symbolics/-----/src/variable.jl:749

# we aren't using Variable because it's 
# deprecated in Symbolics.
# and I'm not sure I want to give it to Convex. 
#@doc (@doc Symbolics.Variable)
#Variable = Symbolics.Variable 
#export Variable
push!(overrides, :Variable)

## :Vec
# Showing duplicate methods for Vec in packages Module[CairoMakie, GeometryBasics, Measures]
# Methods for GeometryBasics.Vec in package Core
# (::Type{SA})(gen::Base.Generator) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/SArray.jl:57
# (::Type{SA})(sa::StaticArray) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/convert.jl:178
# (::Type{SA})(x...) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/convert.jl:173
# (::Type{SV})(x::StaticArray{Tuple{N}, T, 1} where {N, T}) where SV<:Vec @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:76
# (T::Type{<:StaticArray})(a::AbstractArray) @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/convert.jl:182
# GeometryBasics.Vec(x::NTuple{S, T} where T) where S @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:52
# GeometryBasics.Vec(x::T) where {S, T<:NTuple{S, Any}} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:53
# Methods for NTuple{N, Measure} where N in package Core
# (::Type{T})(itr) where T<:Tuple @ Base tuple.jl:455
# (::Type{T})(nt::NamedTuple) where T<:Tuple @ Base namedtuple.jl:198
# (::Type{T})(x::Tuple) where T<:Tuple @ Base tuple.jl:450

# We are just ignoring the Measures vectors... 
@doc (@doc GeometryBasics.Vec)
Vec = GeometryBasics.Vec
push!(overrides, :Vec)
export Vec

## :Vec2
# Showing duplicate methods for Vec2 in packages Module[CairoMakie, GeometryBasics, Measures]
# Methods for GeometryBasics.Vec2 in package Core
# (::Type{SA})(x...) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/convert.jl:173
# (GeometryBasics.Vec{S})(x::AbstractVector{T}) where {S, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:36
# (GeometryBasics.Vec{S})(x::T) where {S, T<:Tuple} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:57
# (GeometryBasics.Vec{S})(x::T) where {S, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:50
# Methods for Tuple{Measure, Measure} in package Core
# (::Type{T})(itr) where T<:Tuple @ Base tuple.jl:455
# (::Type{T})(nt::NamedTuple) where T<:Tuple @ Base namedtuple.jl:198
# (::Type{T})(x::Tuple) where T<:Tuple @ Base tuple.jl:450
# NTuple{N, T}(v::SIMD.Vec{N}) where {T, N} @ SIMD ~/.julia/packages/SIMD/-----/src/simdvec.jl:68

@doc (@doc GeometryBasics.Vec2)
Vec2 = GeometryBasics.Vec2
export Vec2
push!(overrides, :Vec2)

## :Vec3
# Showing duplicate methods for Vec3 in packages Module[CairoMakie, GeometryBasics, Measures]
# Methods for GeometryBasics.Vec3 in package Core
# (::Type{SA})(x...) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/-----/src/convert.jl:173
# (GeometryBasics.Vec{S})(x::AbstractVector{T}) where {S, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:36
# (GeometryBasics.Vec{S})(x::T) where {S, T<:Tuple} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:57
# (GeometryBasics.Vec{S})(x::T) where {S, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:50
# Methods for Tuple{Measure, Measure, Measure} in package Core
# (::Type{T})(itr) where T<:Tuple @ Base tuple.jl:455
# (::Type{T})(nt::NamedTuple) where T<:Tuple @ Base namedtuple.jl:198
# (::Type{T})(x::Tuple) where T<:Tuple @ Base tuple.jl:450
# NTuple{N, T}(v::SIMD.Vec{N}) where {T, N} @ SIMD ~/.julia/packages/SIMD/-----/src/simdvec.jl:68

@doc (@doc GeometryBasics.Vec3)
Vec3 = GeometryBasics.Vec3
export Vec3
push!(overrides, :Vec3)

## :Zeros
# Showing duplicate methods for Zeros in packages Module[FillArrays, JuMP]
# Methods for FillArrays.Zeros in package Core
# FillArrays.Zeros(::Type{T}, m...) where T @ FillArrays ~/.julia/packages/FillArrays/-----/src/FillArrays.jl:317
# FillArrays.Zeros(A::AbstractArray) @ FillArrays ~/.julia/packages/FillArrays/-----/src/FillArrays.jl:316
# FillArrays.Zeros(n::Integer) @ FillArrays ~/.julia/packages/FillArrays/-----/src/FillArrays.jl:311
# FillArrays.Zeros(sz::SZ) where {N, SZ<:NTuple{N, Any}} @ FillArrays ~/.julia/packages/FillArrays/-----/src/FillArrays.jl:309
# FillArrays.Zeros(sz::Vararg{Any, N}) where N @ FillArrays ~/.julia/packages/FillArrays/-----/src/FillArrays.jl:308
# Methods for JuMP.Zeros in package Core
# JuMP.Zeros() @ JuMP ~/.julia/packages/JuMP/-----/src/macros/@constraint.jl:704
@doc (@doc FillArrays.Zeros)
Zeros = FillArrays.Zeros
export Zeros
push!(overrides, :Zeros)

## :attributes
# Showing duplicate methods for attributes in packages Module[CairoMakie, EzXML, HDF5]
# Methods for attributes in package MakieCore
# attributes(x::AbstractPlot) @ MakieCore ~/.julia/packages/MakieCore/-----/src/attributes.jl:35
# attributes(x::Attributes) @ MakieCore ~/.julia/packages/MakieCore/-----/src/attributes.jl:34
# Methods for attributes in package EzXML
# attributes(node::EzXML.Node) @ EzXML ~/.julia/packages/EzXML/-----/src/node.jl:1459
# Methods for attributes in package HDF5
# attributes(p::Union{HDF5.Dataset, HDF5.Datatype, HDF5.File, HDF5.Group}) @ HDF5 ~/.julia/packages/HDF5/-----/src/attributes.jl:374
@doc (@doc HDF5.attributes)
attributes(p::Union{HDF5.Dataset, HDF5.Datatype, HDF5.File, HDF5.Group}) = HDF5.attributes(p)
@doc (@doc EzXML.attributes)
attributes(node::EzXML.Node) = EzXML.attributes(node)
@doc (@doc Makie.attributes)
attributes(x::Attributes) = Makie.attributes(x)
attributes(x::AbstractPlot) = Makie.attributes(x)
export attributes
push!(overrides, :attributes)

## :center
# Showing duplicate methods for center in packages Module[Graphs, Images]
# Methods for center in package Graphs
# center(eccentricities::Vector) @ Graphs ~/.julia/packages/Graphs/-----/src/distance.jl:193
# center(g::AbstractGraph) @ Graphs ~/.julia/packages/Graphs/-----/src/distance.jl:198
# center(g::AbstractGraph, distmx::AbstractMatrix) @ Graphs ~/.julia/packages/Graphs/-----/src/distance.jl:198
# Methods for center in package ImageTransformations
# center(img::AbstractArray{T, N}) where {T, N} @ ImageTransformations ~/.julia/packages/ImageTransformations/-----/src/ImageTransformations.jl:80

@doc (@doc Images.center)
center(img::AbstractArray{T, N}) where {T, N} = Images.center(img)
@doc (@doc Graphs.center)
center(g::AbstractGraph, distmx::AbstractMatrix) = Graphs.center(g, distmx)
center(g::AbstractGraph) = Graphs.center(g)
center(eccentricities::Vector) = Graphs.center(eccentricities)
export center
push!(overrides, :center)

## :centered
# Showing duplicate methods for centered in packages Module[GeometryBasics, Images]
# Methods for centered in package GeometryBasics
# centered(::Type{T}) where T<:HyperSphere @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/spheres.jl:40
# centered(R::Type{HyperRectangle{N, T}}) where {N, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/rectangles.jl:535
# centered(R::Type{HyperRectangle{N}}) where N @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/rectangles.jl:536
# centered(R::Type{HyperRectangle}) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/rectangles.jl:537
# centered(S::Type{HyperSphere{N, T}}) where {N, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/spheres.jl:39
# Methods for centered in package OffsetArrays
# centered(A::AbstractArray) @ OffsetArrays ~/.julia/packages/OffsetArrays/-----/src/OffsetArrays.jl:823
# centered(A::AbstractArray, cp::NTuple{N, Int64} where N) @ OffsetArrays ~/.julia/packages/OffsetArrays/-----/src/OffsetArrays.jl:823
# centered(A::AbstractArray, i::CartesianIndex) @ OffsetArrays ~/.julia/packages/OffsetArrays/-----/src/OffsetArrays.jl:825
# centered(A::AbstractArray, r::RoundingMode) @ OffsetArrays deprecated.jl:103
# centered(A::ImageMorphology.StructuringElements.MorphologySEArray) @ ImageMorphology.StructuringElements ~/.julia/packages/ImageMorphology/-----/src/StructuringElements/StructuringElements.jl:20
# centered(a::AxisArray) @ ImageAxes ~/.julia/packages/ImageAxes/-----/src/offsetarrays.jl:6
# centered(a::ImageMeta) @ ImageMetadata ~/.julia/packages/ImageMetadata/-----/src/ImageMetadata.jl:272
# centered(ax::AxisArrays.Axis{name}) where name @ ImageAxes ~/.julia/packages/ImageAxes/-----/src/offsetarrays.jl:5

# Remove override for GeometryBasics.jl - 2024-10-25 - This doesn't seem to be used
# and calling this with HypersSphere actually breaks. 
#@doc (@doc GeometryBasics.centered)
#centered(R::Type{T}) where T <: Union{HyperRectangle,HyperSphere} = GeometryBasics.centered(R) 
#centered(::Type{T}) where T<:HyperSphere = GeometryBasics.centered(T)
#@doc (@doc OffsetArrays.centered)
centered = OffsetArrays.centered 
# centered(ax::Images.Axis) = OffsetArrays.centered(ax)
# centered(a::ImageMeta) = OffsetArrays.centered(a)
# centered(a::AxisArray) = OffsetArrays.centered(a)
# centered(A::ImageMorphology.StructuringElements.MorphologySEArray) = OffsetArrays.centered(A)
# centered(A::AbstractArray) = OffsetArrays.centered(A)
# centered(A::AbstractArray, r) = OffsetArrays.centered(A, r)
export centered
push!(overrides, :centered)

## :complement
# Showing duplicate methods for complement in packages Module[ColorVectorSpace, DataStructures, Graphs, Images]
# Methods for complement in package ColorVectorSpace
# complement(x::TransparentColor) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/-----/src/ColorVectorSpace.jl:238
# complement(x::Union{Number, Colorant}) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/-----/src/ColorVectorSpace.jl:237
# Methods for complement in package DataStructures
# complement(s::DataStructures.IntSet) @ DataStructures ~/.julia/packages/DataStructures/-----/src/int_set.jl:193
# Methods for complement in package Graphs
# complement(g::SimpleDiGraph) @ Graphs ~/.julia/packages/Graphs/-----/src/operators.jl:49
# complement(g::SimpleGraph) @ Graphs ~/.julia/packages/Graphs/-----/src/operators.jl:36
@doc (@doc DataStructures.complement)
complement(s::DataStructures.IntSet) = DataStructures.complement(s)
@doc (@doc Images.complement)
complement(x::TransparentColor) = Images.complement(x)
complement(x::Union{Number, Colorant}) = Images.complement(x)
@doc (@doc Graphs.complement)
complement(g::SimpleDiGraph) = Graphs.complement(g)
complement(g::SimpleGraph) = Graphs.complement(g)
export complement # Export the combined complement function
push!(overrides, :complement)

## :constant
# Showing duplicate methods for constant in packages Module[Convex, JuMP]
# Methods for constant in package Convex
# constant(x) @ Convex ~/.julia/packages/Convex/-----/src/constant.jl:102
# constant(x::Convex.ComplexConstant) @ Convex ~/.julia/packages/Convex/-----/src/constant.jl:101
# constant(x::Convex.Constant) @ Convex ~/.julia/packages/Convex/-----/src/constant.jl:100
# Methods for constant in package JuMP
# constant(aff::GenericAffExpr) @ JuMP ~/.julia/packages/JuMP/-----/src/aff_expr.jl:432
# constant(quad::GenericQuadExpr) @ JuMP ~/.julia/packages/JuMP/-----/src/quad_expr.jl:370

@doc (@doc JuMP.constant)
constant(aff::GenericAffExpr) = JuMP.constant(aff)
constant(quad::GenericQuadExpr) = JuMP.constant(quad)
@doc (@doc Convex.constant)
constant(x::Convex.ComplexConstant) = Convex.constant(x) 
constant(x::Convex.Constant) = Convex.constant(x) 
export constant
push!(overrides, :constant)

## :conv
# Showing duplicate methods for conv in packages Module[Convex, DSP, Flux]
# Methods for conv in package Convex
# conv(x::Convex.AbstractExpr, y::Union{Number, AbstractArray}) @ Convex ~/.julia/packages/Convex/-----/src/reformulations/conv.jl:60
# conv(x::Union{Number, AbstractArray}, y::Convex.AbstractExpr) @ Convex ~/.julia/packages/Convex/-----/src/reformulations/conv.jl:50
# Methods for conv in package DSP
# conv(::Ones{T, 1, <:Tuple{var"#s2"} where var"#s2"<:InfiniteArrays.OneToInf}, ::Ones{V, 1, <:Tuple{var"#s4"} where var"#s4"<:InfiniteArrays.OneToInf}) where {T<:Integer, V<:Integer} @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:18
# conv(::Ones{T, 1, <:Tuple{var"#s3"} where var"#s3"<:InfiniteArrays.OneToInf}, ::Ones{V, 1, <:Tuple{var"#s1"} where var"#s1"<:InfiniteArrays.OneToInf}) where {T, V} @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:22
# conv(::Ones{T, 1, <:Tuple{var"#s3"} where var"#s3"<:InfiniteArrays.OneToInf}, a::AbstractVector{V}) where {T, V} @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:25
# conv(::Ones{T, 1, <:Tuple{var"#s3"} where var"#s3"<:InfiniteArrays.OneToInf}, a::Vector{V}) where {T, V} @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:30
# conv(::Trues{1, <:Tuple{var"#s3"} where var"#s3"<:InfiniteArrays.OneToInf}, ::Trues{1, <:Tuple{var"#s1"} where var"#s1"<:InfiniteArrays.OneToInf}) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:20
# conv(A::AbstractArray{<:Number, M}, B::AbstractArray{<:Number, N}) where {M, N} @ DSP ~/.julia/packages/DSP/-----/src/dspbase.jl:722
# conv(a::AbstractVector{V}, ::Ones{T, 1, <:Tuple{var"#s3"} where var"#s3"<:InfiniteArrays.OneToInf}) where {T, V} @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:35
# conv(a::Vector{V}, ::Ones{T, 1, <:Tuple{var"#s3"} where var"#s3"<:InfiniteArrays.OneToInf}) where {T, V} @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:40
# conv(b1::SampleBuf{T, 1}, b2::SampleBuf{T, 1}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:322
# conv(b1::SampleBuf{T, 1}, b2::StridedVector{T}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:344
# conv(b1::SampleBuf{T, 2}, b2::StridedMatrix{T}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:350
# conv(b1::SampleBuf{T, N1}, b2::SampleBuf{T, N2}) where {T, N1, N2} @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:329
# conv(b1::SpectrumBuf{T, 1}, b2::SpectrumBuf{T, 1}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:322
# conv(b1::SpectrumBuf{T, 1}, b2::StridedVector{T}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:344
# conv(b1::SpectrumBuf{T, 2}, b2::StridedMatrix{T}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:350
# conv(b1::SpectrumBuf{T, N1}, b2::SpectrumBuf{T, N2}) where {T, N1, N2} @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:329
# conv(b1::StridedMatrix{T}, b2::SampleBuf{T, 2}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:362
# conv(b1::StridedMatrix{T}, b2::SpectrumBuf{T, 2}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:362
# conv(b1::StridedVector{T}, b2::SampleBuf{T, 1}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:348
# conv(b1::StridedVector{T}, b2::SpectrumBuf{T, 1}) where T @ SampledSignals ~/.julia/packages/SampledSignals/-----/src/SampleBuf.jl:348
# conv(r1::FillArrays.AbstractFill{<:Any, 1, <:Tuple{var"#s2"} where var"#s2"<:InfiniteArrays.OneToInf}, r2::FillArrays.AbstractFill{<:Any, 1, <:Tuple{var"#s6"} where var"#s6"<:InfiniteArrays.OneToInf}) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:65
# conv(r1::FillArrays.AbstractFill{<:Any, 1, <:Tuple{var"#s4"} where var"#s4"<:InfiniteArrays.OneToInf}, r2::Ones{<:Any, 1, <:Tuple{var"#s1"} where var"#s1"<:InfiniteArrays.OneToInf}) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:69
# conv(r1::Ones{<:Any, 1, <:Tuple{var"#s4"} where var"#s4"<:InfiniteArrays.OneToInf}, r2::FillArrays.AbstractFill{<:Any, 1, <:Tuple{var"#s1"} where var"#s1"<:InfiniteArrays.OneToInf}) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:73
# conv(r1::Union{InfiniteArrays.AbstractInfUnitRange{T}, InfiniteArrays.InfStepRange{T}} where T, r2::FillArrays.AbstractFill{<:Any, 1, <:Tuple{var"#s2"} where var"#s2"<:InfiniteArrays.OneToInf}) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:55
# conv(r1::Union{InfiniteArrays.AbstractInfUnitRange{T}, InfiniteArrays.InfStepRange{T}} where T, r2::Ones{<:Any, 1, <:Tuple{var"#s2"} where var"#s2"<:InfiniteArrays.OneToInf}) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:60
# conv(r1::Union{InfiniteArrays.AbstractInfUnitRange{T}, InfiniteArrays.InfStepRange{T}} where T, r2::Union{InfiniteArrays.AbstractInfUnitRange{T}, InfiniteArrays.InfStepRange{T}} where T) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:63
# conv(r2::FillArrays.AbstractFill{<:Any, 1, <:Tuple{var"#s2"} where var"#s2"<:InfiniteArrays.OneToInf}, r1::Union{InfiniteArrays.AbstractInfUnitRange{T}, InfiniteArrays.InfStepRange{T}} where T) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:57
# conv(r2::Ones{<:Any, 1, <:Tuple{var"#s2"} where var"#s2"<:InfiniteArrays.OneToInf}, r1::Union{InfiniteArrays.AbstractInfUnitRange{T}, InfiniteArrays.InfStepRange{T}} where T) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:61
# conv(r::Union{InfiniteArrays.AbstractInfUnitRange{T}, InfiniteArrays.InfStepRange{T}} where T, x::AbstractVector) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:46
# conv(u::AbstractArray{<:Integer, N}, v::AbstractArray{<:Integer, N}) where N @ DSP ~/.julia/packages/DSP/-----/src/dspbase.jl:706
# conv(u::AbstractArray{<:Number, N}, v::AbstractArray{<:Number, N}) where N @ DSP ~/.julia/packages/DSP/-----/src/dspbase.jl:709
# conv(u::AbstractArray{<:Number, N}, v::AbstractArray{<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N}) where N @ DSP ~/.julia/packages/DSP/-----/src/dspbase.jl:712
# conv(u::AbstractArray{<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N}, v::AbstractArray{<:Number, N}) where N @ DSP ~/.julia/packages/DSP/-----/src/dspbase.jl:717
# conv(u::AbstractArray{<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N}, v::AbstractArray{<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N}) where N @ DSP ~/.julia/packages/DSP/-----/src/dspbase.jl:700
# conv(u::AbstractArray{T, N}, v::AbstractArray{T, N}) where {T<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N} @ DSP ~/.julia/packages/DSP/-----/src/dspbase.jl:689
# conv(u::AbstractVector{T}, v::AbstractVector{T}, A::AbstractMatrix{T}) where T @ DSP ~/.julia/packages/DSP/-----/src/dspbase.jl:739
# conv(x::AbstractVector, r::Union{InfiniteArrays.AbstractInfUnitRange{T}, InfiniteArrays.InfStepRange{T}} where T) @ InfiniteArraysDSPExt ~/.julia/packages/InfiniteArrays/aJMPs/ext/InfiniteArraysDSPExt.jl:50
# Methods for conv in package NNlib
# conv(a::AbstractArray{<:Real}, b::AbstractArray{Flux.NilNumber.Nil}, dims::DenseConvDims) @ Flux ~/.julia/packages/Flux/-----/src/outputsize.jl:152
# conv(a::AbstractArray{Flux.NilNumber.Nil}, b::AbstractArray{<:Real}, dims::DenseConvDims) @ Flux ~/.julia/packages/Flux/-----/src/outputsize.jl:156
# conv(a::AbstractArray{Flux.NilNumber.Nil}, b::AbstractArray{Flux.NilNumber.Nil}, dims::DenseConvDims) @ Flux ~/.julia/packages/Flux/-----/src/outputsize.jl:148
# conv(x, w::AbstractArray{T, N}; stride, pad, dilation, flipped, groups) where {T, N} @ NNlib ~/.julia/packages/NNlib/-----/src/conv.jl:50
# conv(x::AbstractArray{xT, N}, w::AbstractArray{wT, N}, cdims::ConvDims; kwargs...) where {xT, wT, N} @ NNlib ~/.julia/packages/NNlib/-----/src/conv.jl:83

conv = DSP.conv # the ML folks will have to specialize in their modules... 
export conv
push!(overrides, :conv)


## :crossentropy
# Showing duplicate methods for crossentropy in packages Module[Flux, StatsBase]
# Methods for crossentropy in package Flux.Losses
# crossentropy(ŷ, y; dims, agg, eps) @ Flux.Losses ~/.julia/packages/Flux/-----/src/losses/functions.jl:231
# Methods for crossentropy in package StatsBase
# crossentropy(p::AbstractArray{<:Real}, q::AbstractArray{<:Real}) @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:799
# crossentropy(p::AbstractArray{<:Real}, q::AbstractArray{<:Real}, b::Real) @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:818

crossentropy = Flux.crossentropy
export crossentropy
push!(overrides, :crossentropy)

## :curvature
# Showing duplicate methods for curvature in packages Module[Convex, DelaunayTriangulation]
# Methods for curvature in package Convex
# curvature(::Convex.AbsAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/AbsAtom.jl:19
# curvature(::Convex.AdditionAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/AdditionAtom.jl:49
# curvature(::Convex.ConjugateAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/ConjugateAtom.jl:19
# curvature(::Convex.DiagAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/DiagAtom.jl:28
# curvature(::Convex.DiagMatrixAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/DiagMatrixAtom.jl:29
# curvature(::Convex.DotSortAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/DotSortAtom.jl:43
# curvature(::Convex.EigMaxAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/EigMaxAtom.jl:24
# curvature(::Convex.EigMinAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/EigMinAtom.jl:24
# curvature(::Convex.EntropyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/EntropyAtom.jl:27
# curvature(::Convex.EuclideanNormAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/EuclideanNormAtom.jl:19
# curvature(::Convex.ExpAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/ExpAtom.jl:26
# curvature(::Convex.GeoMeanAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/GeoMeanAtom.jl:33
# curvature(::Convex.HcatAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/HcatAtom.jl:30
# curvature(::Convex.HuberAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/HuberAtom.jl:27
# curvature(::Convex.ImaginaryAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/ImaginaryAtom.jl:19
# curvature(::Convex.IndexAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/IndexAtom.jl:28
# curvature(::Convex.LogAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/LogAtom.jl:26
# curvature(::Convex.LogDetAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/LogDetAtom.jl:19
# curvature(::Convex.LogSumExpAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/LogSumExpAtom.jl:35
# curvature(::Convex.MatrixFracAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MatrixFracAtom.jl:34
# curvature(::Convex.MaxAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MaxAtom.jl:47
# curvature(::Convex.MaximumAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MaximumAtom.jl:26
# curvature(::Convex.MinAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MinAtom.jl:47
# curvature(::Convex.MinimumAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MinimumAtom.jl:26
# curvature(::Convex.NegateAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/NegateAtom.jl:17
# curvature(::Convex.NuclearNormAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/NuclearNormAtom.jl:19
# curvature(::Convex.OperatorNormAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/OperatorNormAtom.jl:19
# curvature(::Convex.QolElemAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QolElemAtom.jl:28
# curvature(::Convex.QuadOverLinAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QuadOverLinAtom.jl:33
# curvature(::Convex.QuantumEntropyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QuantumEntropyAtom.jl:28
# curvature(::Convex.QuantumRelativeEntropy1Atom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QuantumRelativeEntropyAtom.jl:39
# curvature(::Convex.QuantumRelativeEntropy2Atom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QuantumRelativeEntropyAtom.jl:104
# curvature(::Convex.RationalNormAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/RationalNormAtom.jl:36
# curvature(::Convex.RealAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/RealAtom.jl:24
# curvature(::Convex.RelativeEntropyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/RelativeEntropyAtom.jl:26
# curvature(::Convex.ReshapeAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/ReshapeAtom.jl:28
# curvature(::Convex.RootDetAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/RootDetAtom.jl:14
# curvature(::Convex.SumAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/SumAtom.jl:19
# curvature(::Convex.SumLargestAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/SumLargestAtom.jl:35
# curvature(::Convex.SumLargestEigsAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/SumLargestEigsAtom.jl:26
# curvature(::Convex.TraceLogmAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/TraceLogmAtom.jl:40
# curvature(::Convex.VcatAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/VcatAtom.jl:30
# curvature(atom::Convex.TraceMpowerAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/TraceMpowerAtom.jl:51
# curvature(p::Problem) @ Convex ~/.julia/packages/Convex/-----/src/problems.jl:117
# curvature(x::Convex.BroadcastMultiplyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/BroadcastMultiplyAtom.jl:45
# curvature(x::Convex.MultiplyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MultiplyAtom.jl:39
# Methods for curvature in package DelaunayTriangulation
# curvature(::LineSegment, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/linesegment.jl:51
# curvature(c::CircularArc, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/circulararc.jl:121
# curvature(c::DelaunayTriangulation.AbstractParametricCurve, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/abstract.jl:370
# curvature(e::DelaunayTriangulation.EllipticalArc, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/ellipticalarc.jl:101

@doc (@doc Convex.curvature) 
curvature(x::Convex.AbstractExpr) = Convex.curvature(x)
@doc (@doc DelaunayTriangulation.curvature) 
curvature(c::DelaunayTriangulation.AbstractParametricCurve,t) = DelaunayTriangulation.curvature(c, t)

export curvature
push!(overrides, :curvature)

## :degree
# Showing duplicate methods for degree in packages Module[Graphs, Polynomials]
# Methods for degree in package Graphs
# degree(g::AbstractGraph) @ Graphs ~/.julia/packages/Graphs/-----/src/core.jl:137
# degree(g::AbstractGraph, v::Integer) @ Graphs ~/.julia/packages/Graphs/-----/src/core.jl:130
# degree(g::AbstractGraph, vs) @ Graphs ~/.julia/packages/Graphs/-----/src/core.jl:137
# Methods for degree in package Polynomials
# degree(p::AbstractPolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:702
# degree(p::P) where P<:FactoredPolynomial @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/factored_polynomial.jl:232
# degree(p::Polynomials.AbstractDenseUnivariatePolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/abstract-polynomial.jl:123
# degree(p::Polynomials.AbstractLaurentUnivariatePolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/abstract-polynomial.jl:124
# degree(p::Polynomials.ImmutableDensePolynomial{B, T, X, 0}) where {B, T, X} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomial-container-types/immutable-dense-polynomial.jl:262
# degree(p::Polynomials.ImmutableDensePolynomial{B, T, X, N}) where {B, T, X, N} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomial-container-types/immutable-dense-polynomial.jl:263
# degree(p::Polynomials.MutableDenseLaurentPolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomial-container-types/mutable-dense-laurent-polynomial.jl:106
# degree(p::Polynomials.MutableDensePolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomial-container-types/mutable-dense-polynomial.jl:103
# degree(p::Polynomials.MutableDenseViewPolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomial-container-types/mutable-dense-view-polynomial.jl:64
# degree(p::Polynomials.MutableSparsePolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomial-container-types/mutable-sparse-polynomial.jl:93
# degree(p::Polynomials.MutableSparseVectorPolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomial-container-types/mutable-sparse-vector-polynomial.jl:96
# degree(pq::Polynomials.AbstractRationalFunction) @ Polynomials ~/.julia/packages/Polynomials/-----/src/rational-functions/common.jl:222
@doc (@doc Graphs.degree)
degree(g::Graphs.AbstractGraph, i) = Graphs.degree(g, i)
degree(g::Graphs.AbstractGraph) = Graphs.degree(g)
@doc (@doc Polynomials.degree)
degree(p::AbstractPolynomial) = Polynomials.degree(p)
degree(p::Polynomials.AbstractRationalFunction) = Polynomials.degree(p)
# @doc (@doc Meshes.degree)
# degree(b::Meshes.BezierCurve) = Meshes.degree(b)
export degree 
push!(overrides, :degree)

## :density
# Showing duplicate methods for density in packages Module[CairoMakie, Graphs]
# Methods for density in package Makie
# density() @ Makie ~/.julia/packages/MakieCore/-----/src/recipes.jl:432
# density(args...; kw...) @ Makie ~/.julia/packages/MakieCore/-----/src/recipes.jl:447
# Methods for density in package Graphs
# density(::Type{IsDirected{var"##232"}}, g::var"##232") where var"##232" @ Graphs ~/.julia/packages/Graphs/-----/src/core.jl:392
# density(::Type{SimpleTraits.Not{IsDirected{var"##233"}}}, g::var"##233") where var"##233" @ Graphs ~/.julia/packages/Graphs/-----/src/core.jl:393
# density(g::var"##232") where var"##232" @ Graphs ~/.julia/packages/SimpleTraits/-----/src/SimpleTraits.jl:331
@doc (@doc Makie.density)
density() = Makie.density()
density(args...; kw...) = Makie.density(args...; kw...)
@doc (@doc Graphs.density)
density(g::AbstractGraph) = Graphs.density(g)
export density
push!(overrides, :density)

## :derivative
# Showing duplicate methods for derivative in packages Module[Polynomials, TaylorSeries]
# Methods for derivative in package Polynomials
# derivative(p::AbstractUnivariatePolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/abstract-polynomial.jl:231
# derivative(p::AbstractUnivariatePolynomial, n::Int64) @ Polynomials ~/.julia/packages/Polynomials/-----/src/abstract-polynomial.jl:231
# derivative(p::LaurentPolynomial{T, X}) where {T, X} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/laurent-polynomial.jl:155
# derivative(p::P) where P<:FactoredPolynomial @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/factored_polynomial.jl:372
# derivative(p::P) where {B<:ChebyshevTBasis, T, X, P<:MutableDensePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/chebyshev.jl:182
# derivative(p::P) where {B<:StandardBasis, T, X, P<:AbstractDenseUnivariatePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:126
# derivative(p::P) where {B<:StandardBasis, T, X, P<:AbstractLaurentUnivariatePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:140
# derivative(p::P, n::Int64) where P<:FactoredPolynomial @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/factored_polynomial.jl:372
# derivative(p::Polynomials.ImmutableDensePolynomial{B, T, X, 0}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/immutable-polynomial.jl:142
# derivative(p::Polynomials.ImmutableDensePolynomial{B, T, X, N}) where {B<:StandardBasis, T, X, N} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/immutable-polynomial.jl:143
# derivative(p::Polynomials.MutableSparsePolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/sparse-polynomial.jl:89
# derivative(p::Polynomials.MutableSparseVectorPolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/sparse-vector-polynomial.jl:55
# derivative(pq::P) where P<:AbstractRationalFunction @ Polynomials ~/.julia/packages/Polynomials/-----/src/rational-functions/common.jl:368
# derivative(pq::P, n::Int64) where P<:AbstractRationalFunction @ Polynomials ~/.julia/packages/Polynomials/-----/src/rational-functions/common.jl:368
# Methods for differentiate in package TaylorSeries
# differentiate(a::HomogeneousPolynomial, r::Int64) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:136
# differentiate(a::HomogeneousPolynomial, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:159
# differentiate(a::Taylor1) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:18
# differentiate(a::Taylor1{T}, n::Int64) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:79
# differentiate(a::TaylorN) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:168
# differentiate(a::TaylorN, ntup::NTuple{N, Int64}) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:187
# differentiate(a::TaylorN, r) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:168
# differentiate(a::TaylorN, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:177
# differentiate(n::Int64, a::Taylor1{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:99
# differentiate(ntup::NTuple{N, Int64}, a::TaylorN) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:211
@doc (@doc Polynomials.derivative)
derivative(p::AbstractPolynomial) = Polynomials.derivative(p)
derivative(p::AbstractPolynomial, n::Integer) = Polynomials.derivative(p, n)
derivative(pq::Polynomials.AbstractRationalFunction) = Polynomials.derivative(pq)
derivative(pq::Polynomials.AbstractRationalFunction, n::Integer) = Polynomials.derivative(pq, n)
@doc (@doc TaylorSeries.differentiate)
derivative(a::AbstractSeries) = TaylorSeries.differentiate(a)
derivative(a::AbstractSeries, r) = TaylorSeries.differentiate(a, r)
export derivative
push!(overrides, :derivative)

## :differentiate
# Showing duplicate methods for differentiate in packages Module[DelaunayTriangulation, TaylorSeries]
# Methods for differentiate in package DelaunayTriangulation
# differentiate(L::DelaunayTriangulation.PiecewiseLinear, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/piecewiselinear.jl:45
# differentiate(L::LineSegment, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/linesegment.jl:43
# differentiate(b::BezierCurve, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/beziercurve.jl:99
# differentiate(b::DelaunayTriangulation.BSpline, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/bspline.jl:127
# differentiate(c::CatmullRomSpline, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/catmullromspline.jl:422
# differentiate(c::CircularArc, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/circulararc.jl:88
# differentiate(c::DelaunayTriangulation.CatmullRomSplineSegment, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/catmullromspline.jl:99
# differentiate(e::DelaunayTriangulation.EllipticalArc, t) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/mesh_refinement/curves/ellipticalarc.jl:83
# Methods for differentiate in package TaylorSeries
# differentiate(a::HomogeneousPolynomial, r::Int64) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:136
# differentiate(a::HomogeneousPolynomial, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:159
# differentiate(a::Taylor1) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:18
# differentiate(a::Taylor1{T}, n::Int64) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:79
# differentiate(a::TaylorN) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:168
# differentiate(a::TaylorN, ntup::NTuple{N, Int64}) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:187
# differentiate(a::TaylorN, r) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:168
# differentiate(a::TaylorN, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:177
# differentiate(n::Int64, a::Taylor1{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:99
# differentiate(ntup::NTuple{N, Int64}, a::TaylorN) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:211

@doc (@doc DelaunayTriangulation.differentiate)
differentiate(a::DelaunayTriangulation.AbstractParametricCurve, t) = DelaunayTriangulation.differentiate(a, t)
@doc (@doc TaylorSeries.differentiate)
differentiate(s::TaylorSeries.AbstractSeries) = TaylorSeries.differentiate(s)
differentiate(s::TaylorSeries.AbstractSeries, r) = TaylorSeries.differentiate(s, r)
differentiate(r, s) = TaylorSeries.differentiate(r, s)
export differentiate
push!(overrides, :differentiate)

## :entropy
# Showing duplicate methods for entropy in packages Module[Convex, Distributions, Images, StatsBase]
# Methods for entropy in package Convex
# entropy(x::Convex.AbstractExpr) @ Convex ~/.julia/packages/Convex/-----/src/supported_operations.jl:602
# Methods for entropy in package StatsBase
# entropy(d::AbstractMvNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvnormal.jl:95
# entropy(d::Arcsine) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/arcsine.jl:72
# entropy(d::Bernoulli) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/bernoulli.jl:77
# entropy(d::BernoulliLogit) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/bernoullilogit.jl:64
# entropy(d::Beta) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/beta.jl:107
# entropy(d::BetaBinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/betabinomial.jl:109
# entropy(d::Binomial; approx) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/binomial.jl:90
# entropy(d::Cauchy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/cauchy.jl:68
# entropy(d::Chernoff) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chernoff.jl:213
# entropy(d::Chisq) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chisq.jl:68
# entropy(d::Chi{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chi.jl:70
# entropy(d::Dirac{T}) where T @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/dirac.jl:38
# entropy(d::Dirichlet) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/dirichlet.jl:109
# entropy(d::DiscreteNonParametric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/discretenonparametric.jl:203
# entropy(d::DiscreteNonParametric, b::Real) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/discretenonparametric.jl:204
# entropy(d::DiscreteUniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/discreteuniform.jl:65
# entropy(d::Distributions.Censored) @ Distributions ~/.julia/packages/Distributions/-----/src/censored.jl:281
# entropy(d::Distributions.Censored{D, S, T, Nothing, T} where {D<:(UnivariateDistribution), S<:ValueSupport, T<:Real}) @ Distributions ~/.julia/packages/Distributions/-----/src/censored.jl:263
# entropy(d::Distributions.Censored{D, S, T, T, Nothing} where {D<:(UnivariateDistribution), S<:ValueSupport, T<:Real}) @ Distributions ~/.julia/packages/Distributions/-----/src/censored.jl:245
# entropy(d::Distributions.GenericMvTDist) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvtdist.jl:107
# entropy(d::Distributions.Normal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:76
# entropy(d::Distributions.ProductDistribution{1, 0, <:Tuple}) @ Distributions ~/.julia/packages/Distributions/-----/src/product.jl:103
# entropy(d::Distributions.ProductDistribution{N, 0} where N) @ Distributions ~/.julia/packages/Distributions/-----/src/product.jl:98
# entropy(d::Erlang) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/erlang.jl:74
# entropy(d::Exponential{T}) where T @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/exponential.jl:64
# entropy(d::FDist) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/fdist.jl:88
# entropy(d::Frechet) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/frechet.jl:103
# entropy(d::Gamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gamma.jl:74
# entropy(d::GeneralizedExtremeValue) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/generalizedextremevalue.jl:158
# entropy(d::Geometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/geometric.jl:68
# entropy(d::Gumbel) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gumbel.jl:82
# entropy(d::Hypergeometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/hypergeometric.jl:78
# entropy(d::InverseGamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/inversegamma.jl:84
# entropy(d::Kumaraswamy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/kumaraswamy.jl:80
# entropy(d::Laplace) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/laplace.jl:72
# entropy(d::Levy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/levy.jl:63
# entropy(d::Lindley) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/lindley.jl:74
# entropy(d::LocationScale{T, Continuous, D} where {T<:Real, D<:Distribution{Univariate, Continuous}}) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/locationscale.jl:123
# entropy(d::LocationScale{T, Discrete, D} where {T<:Real, D<:Distribution{Univariate, Discrete}}) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/locationscale.jl:124
# entropy(d::LogNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/lognormal.jl:87
# entropy(d::LogUniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/loguniform.jl:50
# entropy(d::Logistic) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/logistic.jl:73
# entropy(d::Multinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/multinomial.jl:120
# entropy(d::MultivariateDistribution, b::Real) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariates.jl:77
# entropy(d::MvLogNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvlognormal.jl:232
# entropy(d::NormalCanon) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normalcanon.jl:57
# entropy(d::PGeneralizedGaussian) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pgeneralizedgaussian.jl:94
# entropy(d::Pareto) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pareto.jl:80
# entropy(d::PoissonBinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/poissonbinomial.jl:110
# entropy(d::Poisson{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/poisson.jl:68
# entropy(d::Product) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/product.jl:52
# entropy(d::Rayleigh{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/rayleigh.jl:66
# entropy(d::Semicircle) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/semicircle.jl:43
# entropy(d::SymTriangularDist) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/symtriangular.jl:66
# entropy(d::TDist{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/tdist.jl:70
# entropy(d::TriangularDist{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/triangular.jl:89
# entropy(d::Truncated{var"#s689", Continuous, T} where {var"#s689"<:(Distributions.Normal), T<:Real}) @ Distributions ~/.julia/packages/Distributions/-----/src/truncated/normal.jl:108
# entropy(d::Uniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/uniform.jl:69
# entropy(d::UnivariateDistribution, b::Real) @ Distributions ~/.julia/packages/Distributions/-----/src/univariates.jl:224
# entropy(d::VonMises) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/vonmises.jl:61
# entropy(d::WalleniusNoncentralHypergeometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/noncentralhypergeometric.jl:266
# entropy(d::Weibull) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/weibull.jl:88
# entropy(d::Wishart) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/wishart.jl:128
# entropy(o::NBClassifier) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/nbclassifier.jl:153
# entropy(p) @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:735
# entropy(p, b::Real) @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:743
# Methods for entropy in package ImageQualityIndexes
# entropy(img::AbstractArray; kind, nbins) @ ImageQualityIndexes ~/.julia/packages/ImageQualityIndexes/-----/src/entropy.jl:53
# entropy(logᵦ::Log, img::AbstractArray{Bool}) where Log<:Function @ ImageQualityIndexes ~/.julia/packages/ImageQualityIndexes/-----/src/entropy.jl:62
# entropy(logᵦ::Log, img; nbins) where Log<:Function @ ImageQualityIndexes ~/.julia/packages/ImageQualityIndexes/-----/src/entropy.jl:54

@doc (@doc Distributions.entropy)
entropy(d::Distributions.Distribution) = Distributions.entropy(d)
entropy(o::NBClassifier) = StatsBase.entropy(o)
@doc (@doc Images.entropy)
entropy(logᵦ::Function, img::AbstractArray{Bool}) = Images.entropy(logᵦ, img)
entropy(logᵦ::Function, img; nbins) = Images.entropy(logᵦ, img; nbins)
entropy(img::AbstractArray; kind, nbins) = Images.entropy(img; kind, nbins)
@doc (@doc Convex.entropy)
entropy(x::Convex.AbstractExpr) = Convex.entropy(x)
export entropy  
push!(overrides, :entropy)

## :evaluate
# Showing duplicate methods for evaluate in packages Module[Convex, Distances, Images, MultivariateStats, TaylorSeries]
# Methods for evaluate in package Convex
# evaluate(atom::Convex.QuantumEntropyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QuantumEntropyAtom.jl:30
# evaluate(atom::Convex.QuantumRelativeEntropy1Atom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QuantumRelativeEntropyAtom.jl:41
# evaluate(atom::Convex.QuantumRelativeEntropy2Atom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QuantumRelativeEntropyAtom.jl:106
# evaluate(atom::Convex.TraceLogmAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/TraceLogmAtom.jl:42
# evaluate(atom::Convex.TraceMpowerAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/TraceMpowerAtom.jl:58
# evaluate(c::Convex.ComplexConstant) @ Convex ~/.julia/packages/Convex/-----/src/constant.jl:80
# evaluate(e::Convex.RelativeEntropyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/RelativeEntropyAtom.jl:28
# evaluate(m::Convex.MatrixFracAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MatrixFracAtom.jl:36
# evaluate(q::Convex.GeoMeanAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/GeoMeanAtom.jl:37
# evaluate(q::Convex.QolElemAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QolElemAtom.jl:30
# evaluate(q::Convex.QuadOverLinAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/QuadOverLinAtom.jl:35
# evaluate(x) @ Convex ~/.julia/packages/Convex/-----/src/expressions.jl:77
# evaluate(x::Convex.AbsAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/AbsAtom.jl:21
# evaluate(x::Convex.AbstractVariable) @ Convex ~/.julia/packages/Convex/-----/src/variable.jl:102
# evaluate(x::Convex.AdditionAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/AdditionAtom.jl:51
# evaluate(x::Convex.BroadcastMultiplyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/BroadcastMultiplyAtom.jl:53
# evaluate(x::Convex.ConjugateAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/ConjugateAtom.jl:21
# evaluate(x::Convex.Constant) @ Convex ~/.julia/packages/Convex/-----/src/constant.jl:130
# evaluate(x::Convex.DiagAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/DiagAtom.jl:30
# evaluate(x::Convex.DiagMatrixAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/DiagMatrixAtom.jl:31
# evaluate(x::Convex.DotSortAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/DotSortAtom.jl:45
# evaluate(x::Convex.EigMaxAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/EigMaxAtom.jl:26
# evaluate(x::Convex.EigMinAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/EigMinAtom.jl:26
# evaluate(x::Convex.EntropyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/EntropyAtom.jl:29
# evaluate(x::Convex.EuclideanNormAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/EuclideanNormAtom.jl:21
# evaluate(x::Convex.ExpAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/ExpAtom.jl:28
# evaluate(x::Convex.HcatAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/HcatAtom.jl:32
# evaluate(x::Convex.HuberAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/HuberAtom.jl:29
# evaluate(x::Convex.ImaginaryAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/ImaginaryAtom.jl:21
# evaluate(x::Convex.IndexAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/IndexAtom.jl:30
# evaluate(x::Convex.LogAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/LogAtom.jl:28
# evaluate(x::Convex.LogDetAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/LogDetAtom.jl:21
# evaluate(x::Convex.LogSumExpAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/LogSumExpAtom.jl:37
# evaluate(x::Convex.MaxAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MaxAtom.jl:49
# evaluate(x::Convex.MaximumAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MaximumAtom.jl:28
# evaluate(x::Convex.MinAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MinAtom.jl:49
# evaluate(x::Convex.MinimumAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MinimumAtom.jl:28
# evaluate(x::Convex.MultiplyAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/MultiplyAtom.jl:47
# evaluate(x::Convex.NegateAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/NegateAtom.jl:19
# evaluate(x::Convex.NuclearNormAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/NuclearNormAtom.jl:21
# evaluate(x::Convex.OperatorNormAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/OperatorNormAtom.jl:22
# evaluate(x::Convex.RationalNormAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/RationalNormAtom.jl:38
# evaluate(x::Convex.RealAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/RealAtom.jl:26
# evaluate(x::Convex.ReshapeAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/ReshapeAtom.jl:30
# evaluate(x::Convex.RootDetAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/RootDetAtom.jl:16
# evaluate(x::Convex.SumAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/SumAtom.jl:21
# evaluate(x::Convex.SumLargestAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/SumLargestAtom.jl:37
# evaluate(x::Convex.SumLargestEigsAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/SumLargestEigsAtom.jl:28
# evaluate(x::Convex.VcatAtom) @ Convex ~/.julia/packages/Convex/-----/src/atoms/VcatAtom.jl:32
# Methods for evaluate in package Distances
# evaluate(::Euclidean, p::Meshes.Point, l::Meshes.Line) @ Meshes ~/.julia/packages/Meshes/-----/src/distances.jl:13
# evaluate(d::Euclidean, l₁::Meshes.Line, l₂::Meshes.Line) @ Meshes ~/.julia/packages/Meshes/-----/src/distances.jl:26
# evaluate(d::Haversine, p₁::Meshes.Point, p₂::Meshes.Point) @ Meshes ~/.julia/packages/Meshes/-----/src/distances.jl:56
# evaluate(d::PreMetric, g::Meshes.Geometry, p::Meshes.Point) @ Meshes ~/.julia/packages/Meshes/-----/src/distances.jl:6
# evaluate(d::PreMetric, p₁::Meshes.Point, p₂::Meshes.Point) @ Meshes ~/.julia/packages/Meshes/-----/src/distances.jl:43
# evaluate(d::SphericalAngle, p₁::Meshes.Point, p₂::Meshes.Point) @ Meshes ~/.julia/packages/Meshes/-----/src/distances.jl:69
# evaluate(dist::PreMetric, a, b) @ Distances ~/.julia/packages/Distances/-----/src/generic.jl:24
# Methods for evaluate in package MultivariateStats
# evaluate(f::LinearDiscriminant, X::AbstractMatrix) @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/lda.jl:71
# evaluate(f::LinearDiscriminant, x::AbstractVector) @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/lda.jl:64
# Methods for evaluate in package TaylorSeries
# evaluate(A::AbstractArray{TaylorN{T}, N}, δx::Vector{S}) where {T<:Number, S<:Number, N} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:444
# evaluate(A::AbstractArray{TaylorN{T}}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:454
# evaluate(A::Array{TaylorN{T}}, δx::Vector{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:449
# evaluate(a::AbstractArray{Taylor1{T}}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:47
# evaluate(a::HomogeneousPolynomial, dx::IntervalArithmetic.IntervalBox{N, T}) where {T<:Real, N} @ TaylorSeriesIAExt ~/.julia/packages/TaylorSeries/XsXwM/ext/TaylorSeriesIAExt.jl:193
# evaluate(a::HomogeneousPolynomial, v) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:167
# evaluate(a::HomogeneousPolynomial, v, vals::Vararg{Number, N}) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:164
# evaluate(a::HomogeneousPolynomial, vals::NTuple{N, var"#s481"} where var"#s481"<:Number) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:156
# evaluate(a::HomogeneousPolynomial{T}) where T @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:169
# evaluate(a::HomogeneousPolynomial{T}, vals::AbstractVector{S}) where {T<:Number, S<:Union{Real, Complex, Taylor1}} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:161
# evaluate(a::Taylor1, dx::IntervalArithmetic.Interval{S}) where S<:Real @ TaylorSeriesIAExt ~/.julia/packages/TaylorSeries/XsXwM/ext/TaylorSeriesIAExt.jl:162
# evaluate(a::Taylor1{Taylor1{T}}, x::Taylor1{T}) where T<:Union{Real, Complex, Taylor1} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:69
# evaluate(a::Taylor1{TaylorN{T}}, dx::Array{TaylorN{T}, 1}) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:107
# evaluate(a::Taylor1{TaylorN{T}}, ind::Int64, dx::T) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:114
# evaluate(a::Taylor1{TaylorN{T}}, ind::Int64, dx::TaylorN{T}) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:125
# evaluate(a::Taylor1{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:33
# evaluate(a::Taylor1{T}, dx::S) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:25
# evaluate(a::Taylor1{T}, dx::T) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:17
# evaluate(a::Taylor1{T}, dx::Taylor1{TaylorN{T}}) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:94
# evaluate(a::Taylor1{T}, dx::TaylorN{T}) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:87
# evaluate(a::Taylor1{T}, x::Taylor1{S}) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:56
# evaluate(a::Taylor1{T}, x::Taylor1{Taylor1{T}}) where T<:Union{Real, Complex, Taylor1} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:76
# evaluate(a::Taylor1{T}, x::Taylor1{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:59
# evaluate(a::TaylorN, dx::IntervalArithmetic.IntervalBox{N, T}) where {T<:Real, N} @ TaylorSeriesIAExt ~/.julia/packages/TaylorSeries/XsXwM/ext/TaylorSeriesIAExt.jl:182
# evaluate(a::TaylorN, vals::NTuple{N, var"#s478"} where var"#s478"<:AbstractSeries; sorting) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:263
# evaluate(a::TaylorN, vals::NTuple{N, var"#s478"} where var"#s478"<:Number; sorting) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:257
# evaluate(a::TaylorN{Taylor1{T}}, vals::AbstractVector{S}; sorting) where {T, S} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:277
# evaluate(a::TaylorN{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:311
# evaluate(a::TaylorN{T}, ind::Int64, val::S) where {T<:Number, S<:Union{Real, Complex, Taylor1}} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:288
# evaluate(a::TaylorN{T}, ind::Int64, val::TaylorN) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:301
# evaluate(a::TaylorN{T}, s::Symbol, val::S) where {T<:Number, S<:Union{Real, Complex, Taylor1}} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:281
# evaluate(a::TaylorN{T}, s::Symbol, val::TaylorN) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:295
# evaluate(a::TaylorN{T}, vals::AbstractVector{<:AbstractSeries}; sorting) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:273
# evaluate(a::TaylorN{T}, vals::AbstractVector{<:Number}; sorting) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:269
# evaluate(a::TaylorN{T}, x::Pair{Symbol, S}) where {T, S} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:308
# evaluate(p::Taylor1{T}, x::AbstractArray{S}) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:83
# evaluate(x::AbstractArray{Taylor1{T}}, δt::S) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/evaluate.jl:44
@doc (@doc MultivariateStats.evaluate)
evaluate(f::MultivariateStats.LinearDiscriminant, X::AbstractMatrix) = MultivariateStats.evaluate(f, X)
evaluate(f::MultivariateStats.LinearDiscriminant, x::AbstractVector) = MultivariateStats.evaluate(f, x)
@doc (@doc Distances.evaluate)
evalaute(d::Distances.PreMetric, a, b) = Distances.evaluate(d, a, b)
@doc (@doc TaylorSeries.evaluate)
evaluate(A::AbstractArray{TaylorN{T}}) where T = TaylorSeries.evaluate(A)
evaluate(A::AbstractArray{TaylorN{T}, N}, δx::Vector{S}) where {T, S, N} = TaylorSeries.evaluate(A, δx)
evaluate(a::TaylorSeries.AbstractSeries) = TaylorSeries.evaluate(a)
evaluate(a::TaylorSeries.AbstractSeries, x) = TaylorSeries.evaluate(a, x)
evaluate(a::TaylorSeries.AbstractSeries, x, y) = TaylorSeries.evaluate(a, x, y)
@doc (@doc Convex.evaluate) 
evaluate(a::Convex.AbstractExpr) = Convex.evaluate(a) 
export evaluate
push!(overrides, :evaluate)

## :fit
# Showing duplicate methods for fit in packages Module[Distributions, MultivariateStats, Polynomials, StatsBase]
# Methods for fit in package StatsAPI
# fit(::Type{<:Beta}, x::AbstractArray{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/beta.jl:241
# fit(::Type{<:Cauchy}, x::AbstractArray{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/cauchy.jl:110
# fit(::Type{<:Distributions.Categorical{P} where P<:Real}, data::Tuple{Int64, AbstractArray}) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/categorical.jl:182
# fit(::Type{<:Distributions.Categorical{P} where P<:Real}, data::Tuple{Int64, AbstractArray}, w::AbstractArray{Float64}) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/categorical.jl:183
# fit(::Type{<:Rician}, x::AbstractArray{T}; tol, maxiters) where T @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/rician.jl:141
# fit(::Type{CCA}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}; outdim, method, xmean, ymean) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/cca.jl:309
# fit(::Type{FactorAnalysis}, X::AbstractMatrix{T}; method, maxoutdim, mean, tol, η, maxiter) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/fa.jl:257
# fit(::Type{Histogram{T}}, v::AbstractVector, edg::AbstractVector; closed) where T @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:298
# fit(::Type{Histogram{T}}, v::AbstractVector, wv::AbstractWeights, edg::AbstractVector; closed) where T @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:302
# fit(::Type{Histogram{T}}, v::AbstractVector, wv::AbstractWeights; closed, nbins) where T @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:304
# fit(::Type{Histogram{T}}, v::AbstractVector; closed, nbins) where T @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:300
# fit(::Type{Histogram{T}}, vs::NTuple{N, AbstractVector}, edges::NTuple{N, AbstractVector}; closed) where {T, N} @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:353
# fit(::Type{Histogram{T}}, vs::NTuple{N, AbstractVector}, wv::AbstractWeights; closed, nbins) where {T, N} @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:362
# fit(::Type{Histogram{T}}, vs::NTuple{N, AbstractVector}, wv::AbstractWeights{W, T} where T<:Real, edges::NTuple{N, AbstractVector}; closed) where {T, N, W} @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:359
# fit(::Type{Histogram{T}}, vs::NTuple{N, AbstractVector}; closed, nbins) where {T, N} @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:356
# fit(::Type{Histogram}, args...; kwargs...) @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:413
# fit(::Type{Histogram}, v::AbstractVector, wv::AbstractWeights{W, T} where T<:Real, args...; kwargs...) where W @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:307
# fit(::Type{Histogram}, vs::NTuple{N, AbstractVector}, wv::AbstractWeights{W, T} where T<:Real, args...; kwargs...) where {N, W} @ StatsBase ~/.julia/packages/StatsBase/-----/src/hist.jl:414
# fit(::Type{ICA}, X::AbstractMatrix{T}, k::Int64; alg, fun, do_whiten, maxiter, tol, mean, winit) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/ica.jl:212
# fit(::Type{KernelPCA}, X::AbstractMatrix{T}; kernel, maxoutdim, remove_zero_eig, atol, solver, inverse, β, tol, maxiter) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/kpca.jl:146
# fit(::Type{LinearDiscriminant}, Xp::DenseMatrix{T}, Xn::DenseMatrix{T}; covestimator) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/lda.jl:142
# fit(::Type{MDS}, X::AbstractMatrix{T}; maxoutdim, distances) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/cmds.jl:232
# fit(::Type{MetricMDS}, X::AbstractMatrix{T}; maxoutdim, metric, tol, maxiter, initial, weights, distances) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/mmds.jl:125
# fit(::Type{MulticlassLDA}, X::AbstractMatrix{T}, y::AbstractVector; method, outdim, regcoef, covestimator_within, covestimator_between) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/lda.jl:331
# fit(::Type{MultivariateStats.KernelCenter}, K::AbstractMatrix{<:Real}) @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/kpca.jl:12
# fit(::Type{PCA}, X::AbstractMatrix{T}; method, maxoutdim, pratio, mean) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/pca.jl:281
# fit(::Type{PPCA}, X::AbstractMatrix{T}; method, maxoutdim, mean, tol, maxiter) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/ppca.jl:306
# fit(::Type{SubspaceLDA}, X::AbstractMatrix{T}, y::AbstractVector; normalize) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/lda.jl:463
# fit(::Type{T}, data::Tuple{Int64, AbstractArray}) where T<:Binomial @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/binomial.jl:204
# fit(::Type{T}, data::Tuple{Int64, AbstractArray}, w::AbstractArray{<:Real}) where T<:Binomial @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/binomial.jl:205
# fit(::Type{UnitRangeTransform}, X::AbstractMatrix{<:Real}; dims, unit) @ StatsBase ~/.julia/packages/StatsBase/-----/src/transformations.jl:263
# fit(::Type{UnitRangeTransform}, X::AbstractVector{<:Real}; dims, unit) @ StatsBase ~/.julia/packages/StatsBase/-----/src/transformations.jl:287
# fit(::Type{Whitening}, X::AbstractMatrix{T}; dims, mean, regcoef) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/whiten.jl:124
# fit(::Type{ZScoreTransform}, X::AbstractMatrix{<:Real}; dims, center, scale) @ StatsBase ~/.julia/packages/StatsBase/-----/src/transformations.jl:109
# fit(::Type{ZScoreTransform}, X::AbstractVector{<:Real}; dims, center, scale) @ StatsBase ~/.julia/packages/StatsBase/-----/src/transformations.jl:130
# fit(dt::Type{D}, args...) where D<:Distribution @ Distributions ~/.julia/packages/Distributions/-----/src/genericfit.jl:47
# fit(dt::Type{D}, x) where D<:Distribution @ Distributions ~/.julia/packages/Distributions/-----/src/genericfit.jl:46
# Methods for fit in package Polynomials
# fit(::Type{ArnoldiFit}, x::AbstractVector{T}, y::AbstractVector{T}, deg::Int64; var, kwargs...) where T @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:764
# fit(::Type{ArnoldiFit}, x::AbstractVector{T}, y::AbstractVector{T}; ...) where T @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:764
# fit(::Type{PQ}, xs::AbstractVector{S}, ys::AbstractVector{T}, m, n; var) where {T, S, PQ<:RationalFunction} @ Polynomials.RationalFunctionFit ~/.julia/packages/Polynomials/-----/src/rational-functions/fit.jl:71
# fit(::Type{P}, x::AbstractVector{T}, y::AbstractVector{T}, deg, cs::Dict; kwargs...) where {T, P<:AbstractUnivariatePolynomial} @ Polynomials ~/.julia/packages/Polynomials/-----/src/abstract-polynomial.jl:243
# fit(::Type{RationalFunction}, r::Polynomial, m::Integer, n::Integer; var) @ Polynomials.RationalFunctionFit ~/.julia/packages/Polynomials/-----/src/rational-functions/fit.jl:114
# fit(P::Type{<:AbstractPolynomial}, x, y, deg::Integer; weights, var) @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:119
# fit(P::Type{<:AbstractPolynomial}, x, y; ...) @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:119
# fit(P::Type{<:AbstractPolynomial}, x::AbstractVector{T}, y::AbstractVector{T}, deg::Integer; weights, var) where T @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:110
# fit(P::Type{<:AbstractPolynomial}, x::AbstractVector{T}, y::AbstractVector{T}; ...) where T @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:110
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}, J, cs::Dict; weights, var) where T @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:602
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}, J, cs; weights, var) where T @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:591
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}, J; ...) where T @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:591
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}, deg::Integer; weights, var) where T @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:554
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}; ...) where T @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:554
# fit(x::AbstractVector, y::AbstractVector, deg::Integer; weights, var) @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:134
# fit(x::AbstractVector, y::AbstractVector; ...) @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:134
@doc (@doc Distributions.fit)
fit(::Type{Distributions.UnivariateDistribution}, data::Tuple{Int64, AbstractArray}) = Distributions.fit(Distributions.UnivariateDistribution, data)
@doc (@doc MultivariateStats.fit)
fit(::Type{T}, args...;kwargs...) where T <: Union{
  AbstractHistogram,
  AbstractDataTransform,
  StatisticalModel
} = MultivariateStats.fit(T, args...; kwargs...)
@doc (@doc Polynomials.fit)
fit(::Type{T}, args...;kwargs...) where T <: Union{
  AbstractPolynomial,
  Polynomials.AbstractRationalFunction,
} = MultivariateStats.fit(T, args...; kwargs...)
# This skips the x, y form for the fit method for Polynomials, but that shouldn't exist...
export fit
push!(overrides, :fit)

## :geomean
# Showing duplicate methods for geomean in packages Module[Convex, StatsBase]
# Methods for geomean in package Convex
# geomean(args::Union{Convex.AbstractExpr, Number, AbstractArray}...) @ Convex ~/.julia/packages/Convex/-----/src/supported_operations.jl:704
# Methods for geomean in package StatsBase
# geomean(a) @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:16

@doc (@doc StatsBase.geomean)
geomean(a) = StatsBase.geomean(a)
@doc (@doc Convex.geomean)
geomean(a::Convex.AbstractExpr, b) = Convex.geomean(a, b)
geomean(a::Convex.AbstractExpr, b::Convex.AbstractExpr) = Convex.geomean(a, b) # need this to fix ambiguities 
geomean(a, b::Convex.AbstractExpr) = Convex.geomean(a, b)
geomean(a::Convex.AbstractExpr, b::Union{Number, AbstractArray, Convex.AbstractExpr}) = Convex.geomean(a, b)
geomean(a::Union{Number, AbstractArray, Convex.AbstractExpr}, b::Convex.AbstractExpr) = Convex.geomean(a, b)
geomean(args::Union{Convex.AbstractExpr, Number, AbstractArray}...) = Convex.geomean(args...)
export geomean
push!(overrides, :geomean)

## :get_weight
# Showing duplicate methods for get_weight in packages Module[DelaunayTriangulation, SimpleWeightedGraphs]
# Methods for get_weight in package DelaunayTriangulation
# get_weight(tri::Triangulation, i) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/triangulation/methods/weights.jl:48
# get_weight(tri::Triangulation, i::Integer) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/triangulation/methods/weights.jl:52
# get_weight(weights, i) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/triangulation/methods/weights.jl:17
# get_weight(weights, i::Integer) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/triangulation/methods/weights.jl:16
# get_weight(weights::DelaunayTriangulation.ZeroWeight{T}, i) where T @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/triangulation/methods/weights.jl:27
# get_weight(weights::DelaunayTriangulation.ZeroWeight{T}, i::Integer) where T @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/triangulation/methods/weights.jl:28
# Methods for get_weight in package SimpleWeightedGraphs
# get_weight(g::AbstractSimpleWeightedGraph, u::Integer, v::Integer) @ SimpleWeightedGraphs ~/.julia/packages/SimpleWeightedGraphs/-----/src/abstractsimpleweightedgraph.jl:34
@doc (@doc DelaunayTriangulation.get_weight)
get_weight(tri::DelaunayTriangulation.Triangulation, i::Integer) = DelaunayTriangulation.get_weight(tri, i)
get_weight(tri::DelaunayTriangulation.ZeroWeight, i::Integer) = DelaunayTriangulation.get_weight(tri, i)
@doc (@doc SimpleWeightedGraphs.get_weight)
get_weight(g::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, u::Integer, v::Integer) = SimpleWeightedGraphs.get_weight(g, u, v)
export get_weight
push!(overrides, :get_weight)


## :gradient
# Showing duplicate methods for gradient in packages Module[Flux, Interpolations]
# Methods for gradient in package Zygote
# gradient(f, args...) @ Zygote ~/.julia/packages/Zygote/-----/src/compiler/interface.jl:146
# Methods for gradient in package Interpolations
# gradient(ch::CubicHermite, x::Float64) @ Interpolations ~/.julia/packages/Interpolations/-----/src/hermite/cubic.jl:81
# gradient(etp::AbstractExtrapolation{T, N}, x::Vararg{Number, N}) where {T, N} @ Interpolations ~/.julia/packages/Interpolations/-----/src/extrapolation/extrapolation.jl:66
# gradient(etp::Interpolations.FilledExtrapolation{T, N, ITP}, x::Vararg{Number, N}) where {T, N, ITP} @ Interpolations ~/.julia/packages/Interpolations/-----/src/extrapolation/filled.jl:57
# gradient(itp::AbstractInterpolation, x::Union{Number, CartesianIndex, AbstractVector}...) @ Interpolations ~/.julia/packages/Interpolations/-----/src/Interpolations.jl:381
# gradient(itp::Interpolations.BSplineInterpolation{T, N, TCoefs, IT, Axs} where {TCoefs<:AbstractArray, IT<:Union{NoInterp, Tuple{Vararg{Union{NoInterp, Interpolations.BSpline}}}, Interpolations.BSpline}, Axs<:NTuple{N, AbstractUnitRange}}, x::Vararg{Number, N}) where {T, N} @ Interpolations ~/.julia/packages/Interpolations/-----/src/b-splines/indexing.jl:25
# gradient(itp::Interpolations.GriddedInterpolation{T, N}, x::Vararg{Number, N}) where {T, N} @ Interpolations ~/.julia/packages/Interpolations/-----/src/gridded/indexing.jl:19
# gradient(itp::Interpolations.MonotonicInterpolation, x::Number) @ Interpolations ~/.julia/packages/Interpolations/-----/src/monotonic/monotonic.jl:213
# gradient(sitp::ScaledInterpolation{T, N}, xs::Vararg{Number, N}) where {T, N} @ Interpolations ~/.julia/packages/Interpolations/-----/src/scaling/scaling.jl:115

@doc (@doc Flux.gradient)
gradient(f, args...) = Flux.gradient(f, args...)
@doc (@doc Interpolations.gradient)
gradient(itp::Interpolations.AbstractInterpolation, x::Union{Number, CartesianIndex, AbstractVector}...) = Interpolations.gradient(itp, x...)
gradient(itp::Interpolations.AbstractExtrapolation, x::Vararg{Number, N}) where N = Interpolations.gradient(itp, x)
gradient(ch::Interpolations.CubicHermite, x::Float64) = Interpolations.gradient(ch, x)
export gradient
push!(overrides, :gradient)

## :groupby
# Showing duplicate methods for groupby in packages Module[DataFrames, IterTools, RDatasets]
# Methods for groupby in package DataAPI
# groupby(df::AbstractDataFrame, cols; sort, skipmissing) @ DataFrames ~/.julia/packages/DataFrames/-----/src/groupeddataframe/groupeddataframe.jl:218
# Methods for groupby in package IterTools
# groupby(keyfunc::F, xs::I) where {F<:Union{Function, Type}, I} @ IterTools ~/.julia/packages/IterTools/-----/src/IterTools.jl:396
@doc (@doc DataFrames.groupby)
groupby(df::DataFrames.AbstractDataFrame, cols; sort, skipmissing) = DataFrames.groupby(df, cols; sort, skipmissing)
@doc (@doc IterTools.groupby)
groupby(keyfunc::F, xs::I) where {F <: Union{Function, Type}, I} = IterTools.groupby(keyfunc, xs)
export groupby
push!(overrides, :groupby)

## :hamming
# Showing duplicate methods for hamming in packages Module[DSP, Distances, Images]
# Methods for hamming in package DSP.Windows
# hamming(dims::Tuple; padding, zerophase) @ DSP.Windows ~/.julia/packages/DSP/-----/src/windows.jl:645
# hamming(n::Integer; padding, zerophase) @ DSP.Windows ~/.julia/packages/DSP/-----/src/windows.jl:200
# Methods for Hamming() in package Distances
# (dist::Hamming)(a, b) @ Distances ~/.julia/packages/Distances/-----/src/metrics.jl:328
@doc (@doc Distances.hamming)
hamming(a, b) = Distances.hamming(a, b)
@doc (@doc DSP.Windows.hamming)
hamming(dims::Tuple; padding, zerophase) = DSP.Windows.hamming(dims; padding, zerophase)
hamming(n::Integer; padding, zerophase) = DSP.Windows.hamming(n; padding, zerophase)
export hamming
push!(overrides, :hamming)

## :has_vertex
# Showing duplicate methods for has_vertex in packages Module[DelaunayTriangulation, Graphs]
# Methods for has_vertex in package DelaunayTriangulation
# has_vertex(G::DelaunayTriangulation.Graph, u) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/triangulation/graph.jl:119
# has_vertex(tri::Triangulation, u) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/triangulation/methods/graph.jl:50
# Methods for has_vertex in package Graphs
# has_vertex(g::AbstractSimpleWeightedGraph, v::Integer) @ SimpleWeightedGraphs ~/.julia/packages/SimpleWeightedGraphs/-----/src/abstractsimpleweightedgraph.jl:40
# has_vertex(g::Graphs.SimpleGraphs.AbstractSimpleGraph, v::Integer) @ Graphs.SimpleGraphs ~/.julia/packages/Graphs/-----/src/SimpleGraphs/SimpleGraphs.jl:179
# has_vertex(g::Graphs.Test.GenericDiGraph, v) @ Graphs.Test ~/.julia/packages/Graphs/-----/src/Test/Test.jl:76
# has_vertex(g::Graphs.Test.GenericGraph, v) @ Graphs.Test ~/.julia/packages/Graphs/-----/src/Test/Test.jl:75
# has_vertex(g::MetaGraphs.AbstractMetaGraph, x...) @ MetaGraphs ~/.julia/packages/MetaGraphs/-----/src/MetaGraphs.jl:71
# has_vertex(g::VertexSafeGraphs.VSafeGraph, v) @ VertexSafeGraphs ~/.julia/packages/VertexSafeGraphs/-----/src/VertexSafeGraphs.jl:32
# has_vertex(x, v) @ Graphs ~/.julia/packages/Graphs/-----/src/interface.jl:247

@doc (@doc DelaunayTriangulation.has_vertex)
has_vertex(tri::DelaunayTriangulation.Triangulation, u) = DelaunayTriangulation.has_vertex(tri, u)
has_vertex(G::DelaunayTriangulation.Graph, u) = DelaunayTriangulation.has_vertex(G, u)
@doc (@doc Graphs.has_vertex)
has_vertex(g::Graphs.AbstractGraph, v) = Graphs.has_vertex(g, v)
export has_vertex
push!(overrides, :has_vertex)

## :height
# Showing duplicate methods for height in packages Module[CairoMakie, GeometryBasics, Measures]
# Methods for height in package GeometryBasics
# height(c::Cylinder{N, T}) where {N, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/cylinders.jl:26
# height(prim::HyperRectangle) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/rectangles.jl:187
# Methods for height in package Measures
# height(x::BoundingBox) @ Measures ~/.julia/packages/Measures/-----/src/boundingbox.jl:44
@doc (@doc GeometryBasics.height)
height(prim::HyperRectangle) = GeometryBasics.height(prim)
height(prim::Cylinder) = GeometryBasics.height(prim)
@doc (@doc Measures.height)
height(x::BoundingBox) = Measures.height(x)
export height
push!(overrides, :height)

## :hinge_loss
# Showing duplicate methods for hinge_loss in packages Module[Convex, Flux]
# Methods for hinge_loss in package Convex
# hinge_loss(x::Convex.AbstractExpr) @ Convex ~/.julia/packages/Convex/-----/src/supported_operations.jl:842
# Methods for hinge_loss in package Flux.Losses
# hinge_loss(ŷ, y; agg) @ Flux.Losses ~/.julia/packages/Flux/-----/src/losses/functions.jl:449
## :imrotate
# Showing duplicate methods for imrotate in packages Module[Flux, Images]
# Methods for imrotate in package NNlib
# imrotate(arr::AbstractArray{T, 4}, θ; method, rotation_center) where T @ NNlib ~/.julia/packages/NNlib/-----/src/rotation.jl:165
# Methods for imrotate in package ImageTransformations
# imrotate(img::AbstractArray, θ::Real, fillvalue::Union{Number, Colorant, Interpolations.Flat, Periodic, Reflect}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, fillvalue::Union{Number, Colorant, Interpolations.Flat, Periodic, Reflect}, method::Union{Interpolations.InterpolationType, Interpolations.Degree}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, inds, fillvalue::Union{Number, Colorant, Interpolations.Flat, Periodic, Reflect}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, inds, fillvalue::Union{Number, Colorant, Interpolations.Flat, Periodic, Reflect}, method::Union{Interpolations.InterpolationType, Interpolations.Degree}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, inds, method::Union{Interpolations.InterpolationType, Interpolations.Degree}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, inds, method::Union{Interpolations.InterpolationType, Interpolations.Degree}, fillvalue::Union{Number, Colorant, Interpolations.Flat, Periodic, Reflect}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, method::Union{Interpolations.InterpolationType, Interpolations.Degree}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, method::Union{Interpolations.InterpolationType, Interpolations.Degree}, fillvalue::Union{Number, Colorant, Interpolations.Flat, Periodic, Reflect}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray{T}, θ::Real, inds::Union{Nothing, Tuple}; kwargs...) where T @ ImageTransformations ~/.julia/packages/ImageTransformations/-----/src/warp.jl:245
# imrotate(img::AbstractArray{T}, θ::Real; ...) where T @ ImageTransformations ~/.julia/packages/ImageTransformations/-----/src/warp.jl:245
imrotate = ImageTransformations.imrotate
export imrotate
push!(overrides, :imrotate)

## :integrate
# Showing duplicate methods for integrate in packages Module[Polynomials, TaylorSeries]
# Methods for integrate in package Polynomials
# integrate(P::AbstractPolynomial) @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:228
# integrate(p::AbstractPolynomial, a, b) @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:248
# integrate(p::P) where P<:FactoredPolynomial @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/factored_polynomial.jl:365
# integrate(p::P) where {B<:ChebyshevTBasis, T, X, P<:MutableDensePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/chebyshev.jl:206
# integrate(p::P) where {B<:StandardBasis, T, X, P<:AbstractDenseUnivariatePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:144
# integrate(p::P, C) where P<:AbstractPolynomial @ Polynomials ~/.julia/packages/Polynomials/-----/src/common.jl:236
# integrate(p::Polynomials.AbstractLaurentUnivariatePolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/standard-basis.jl:162
# integrate(p::Polynomials.ImmutableDensePolynomial{B, T, X, 0}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/immutable-polynomial.jl:151
# integrate(p::Polynomials.ImmutableDensePolynomial{B, T, X, N}) where {B<:StandardBasis, T, X, N} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/immutable-polynomial.jl:153
# integrate(p::Polynomials.MutableSparsePolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/sparse-polynomial.jl:104
# integrate(p::Polynomials.MutableSparseVectorPolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/-----/src/polynomials/standard-basis/sparse-vector-polynomial.jl:73
# integrate(pq::P) where P<:AbstractRationalFunction @ Polynomials ~/.julia/packages/Polynomials/-----/src/rational-functions/common.jl:378
# Methods for integrate in package TaylorSeries
# integrate(a::HomogeneousPolynomial, r::Int64) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:362
# integrate(a::HomogeneousPolynomial, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:389
# integrate(a::Taylor1{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:125
# integrate(a::Taylor1{T}, x::S) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:113
# integrate(a::TaylorN, r::Int64) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:399
# integrate(a::TaylorN, r::Int64, x0) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:419
# integrate(a::TaylorN, r::Int64, x0::TaylorN) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:410
# integrate(a::TaylorN, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:422
# integrate(a::TaylorN, s::Symbol, x0) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:424
# integrate(a::TaylorN, s::Symbol, x0::TaylorN) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/calculus.jl:423
@doc (@doc Polynomials.integrate)
integrate(p::AbstractPolynomial) = Polynomials.integrate(p)
integrate(p::AbstractPolynomial, a, b) = Polynomials.integrate(p, a, b)
integrate(p::AbstractPolynomial, C) = Polynomials.integrate(p, C)
integrate(pq::Polynomials.AbstractRationalFunction) = Polynomials.integrate(pq)
@doc (@doc TaylorSeries.integrate)
integrate(a::AbstractSeries) = TaylorSeries.integrate(a)
integrate(a::AbstractSeries, r) = TaylorSeries.integrate(a, r)
integrate(a::AbstractSeries, r, x0) = TaylorSeries.integrate(a, r, x0)

export integrate
push!(overrides, :integrate)

## :islinear
# Showing duplicate methods for islinear in packages Module[DifferentialEquations, NonlinearSolve, StatsBase]
# Methods for islinear in package SciMLOperators
# islinear(::AffineOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/matrix.jl:536
# islinear(::IdentityOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/basic.jl:33
# islinear(::MatrixOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/matrix.jl:102
# islinear(::NullOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/basic.jl:126
# islinear(::SciMLBase.AbstractDiffEqFunction) @ SciMLBase ~/.julia/packages/SciMLBase/-----/src/scimlfunctions.jl:4468
# islinear(::SciMLOperators.AbstractSciMLOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/interface.jl:309
# islinear(::SciMLOperators.AbstractSciMLScalarOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/scalar.jl:32
# islinear(::SciMLOperators.BatchedDiagonalOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/batch.jl:98
# islinear(::Union{Number, Factorization, UniformScaling, AbstractMatrix}) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/interface.jl:311
# islinear(L) @ SciMLBase ~/.julia/packages/SciMLBase/-----/src/operators/operators.jl:7
# islinear(L::FunctionOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/func.jl:579
# islinear(L::InvertibleOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/matrix.jl:343
# islinear(L::SciMLOperators.AddedOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/basic.jl:418
# islinear(L::SciMLOperators.AdjointOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/left.jl:77
# islinear(L::SciMLOperators.ComposedOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/basic.jl:583
# islinear(L::SciMLOperators.InvertedOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/basic.jl:766
# islinear(L::SciMLOperators.ScaledOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/basic.jl:250
# islinear(L::SciMLOperators.TransposedOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/left.jl:78
# islinear(L::TensorProductOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/tensor.jl:117
# islinear(f::ODEFunction) @ SciMLBase ~/.julia/packages/SciMLBase/-----/src/scimlfunctions.jl:4469
# islinear(f::SplitFunction) @ SciMLBase ~/.julia/packages/SciMLBase/-----/src/scimlfunctions.jl:4470
# islinear(o::SciMLBase.AbstractDiffEqLinearOperator) @ SciMLBase ~/.julia/packages/SciMLBase/-----/src/operators/operators.jl:4
# Methods for islinear in package StatsAPI
islinear = DifferentialEquations.islinear
export islinear
push!(overrides, :islinear)

## :issquare
# Showing duplicate methods for issquare in packages Module[DifferentialEquations, DoubleFloats, NonlinearSolve]
# Methods for issquare in package SciMLOperators
# issquare(::AbstractVector) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/interface.jl:360
# issquare(::Union{Number, UniformScaling, SciMLOperators.AbstractSciMLScalarOperator}) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/interface.jl:361
# issquare(A...) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/interface.jl:366
# issquare(L) @ SciMLOperators ~/.julia/packages/SciMLOperators/-----/src/interface.jl:359
# issquare(x::MatrixOperator, args...; kwargs...) @ SciMLOperators ~/.julia/packages/MacroTools/-----/src/examples/forward.jl:17
# Methods for issquare in package DoubleFloats
# issquare(m::AbstractMatrix{T}) where T<:Number @ DoubleFloats ~/.julia/packages/DoubleFloats/-----/src/math/linearalgebra/support.jl:1
# issquare(m::Array{DoubleFloat{T}, 2}) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/-----/src/math/linearalgebra/support.jl:6
issquare = DifferentialEquations.issquare
export issquare
push!(overrides, :issquare)

## :kldivergence
# Showing duplicate methods for kldivergence in packages Module[Distributions, Flux, StatsBase]
# Methods for kldivergence in package StatsBase
# kldivergence(p::AbstractArray{<:Real}, q::AbstractArray{<:Real}) @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:830
# kldivergence(p::AbstractArray{<:Real}, q::AbstractArray{<:Real}, b::Real) @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:855
# kldivergence(p::AbstractMvNormal, q::AbstractMvNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvnormal.jl:105
# kldivergence(p::Beta, q::Beta) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/beta.jl:114
# kldivergence(p::Binomial, q::Binomial; kwargs...) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/binomial.jl:108
# kldivergence(p::Chi, q::Chi) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chi.jl:82
# kldivergence(p::Chisq, q::Chisq) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chisq.jl:73
# kldivergence(p::Distributions.Normal, q::Distributions.Normal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:78
# kldivergence(p::Distribution{V}, q::Distribution{V}; kwargs...) where V<:VariateForm @ Distributions ~/.julia/packages/Distributions/-----/src/functionals.jl:29
# kldivergence(p::Exponential, q::Exponential) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/exponential.jl:66
# kldivergence(p::Gamma, q::Gamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gamma.jl:87
# kldivergence(p::Geometric, q::Geometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/geometric.jl:70
# kldivergence(p::InverseGamma, q::InverseGamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/inversegamma.jl:89
# kldivergence(p::Laplace, q::Laplace) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/laplace.jl:74
# kldivergence(p::Lindley, q::Lindley) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/lindley.jl:62
# kldivergence(p::LogNormal, q::LogNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/lognormal.jl:92
# kldivergence(p::LogUniform, q::LogUniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/loguniform.jl:72
# kldivergence(p::LogitNormal, q::LogitNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/logitnormal.jl:106
# kldivergence(p::MvLogitNormal, q::MvLogitNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvlogitnormal.jl:87
# kldivergence(p::NegativeBinomial, q::NegativeBinomial; kwargs...) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/negativebinomial.jl:83
# kldivergence(p::NormalCanon, q::NormalCanon) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normalcanon.jl:62
# kldivergence(p::Poisson, q::Poisson) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/poisson.jl:88
# Methods for kldivergence in package Flux.Losses
# kldivergence(ŷ, y; dims, agg, eps) @ Flux.Losses ~/.julia/packages/Flux/-----/src/losses/functions.jl:389

@doc (@doc Distributions.kldivergence)
kldivergence(p::T, q::T) where {T <: Distributions.Distribution} = StatsBase.kldivergence(p, q)
@doc (@doc Flux.kldivergence)
kldivergence(ŷ, y; kwargs...) = Flux.kldivergence(ŷ, y; kwargs...)
export kldivergence
push!(overrides, :kldivergence)

## :logsumexp
# Showing duplicate methods for logsumexp in packages Module[Convex, Flux]
# Methods for logsumexp in package Convex
# logsumexp(x::Convex.AbstractExpr; dims) @ Convex ~/.julia/packages/Convex/-----/src/supported_operations.jl:1177
# Methods for logsumexp in package NNlib
# logsumexp(x::AbstractArray; dims) @ NNlib ~/.julia/packages/NNlib/-----/src/softmax.jl:142
# logsumexp(x::Number, args...) @ NNlib ~/.julia/packages/NNlib/-----/src/softmax.jl:158

@doc (@doc NNlib.logsumexp)
logsumexp(x::AbstractArray; dims) = NNlib.logsumexp(x; dims)
logsumexp(x::Number, args...) = NNlib.logsumexp(x, args...)
@doc (@doc Convex.logsumexp)
logsumexp(x::Convex.AbstractExpr; dims) = Convex.logsumexp(x; dims)
export logsumexp
push!(overrides, :logsumexp)


## :mae
# Showing duplicate methods for mae in packages Module[Flux, Images]
# Methods for mae in package Flux.Losses
# mae(ŷ, y; agg) @ Flux.Losses ~/.julia/packages/Flux/-----/src/losses/functions.jl:21
# Methods for mae in package ImageDistances
# mae(a::AbstractArray{T} where T<:Union{Number, Colorant}, b::AbstractArray{T} where T<:Union{Number, Colorant}) @ ImageDistances ~/.julia/packages/ImageDistances/-----/src/metrics.jl:91

mae = Flux.mae
export mae
push!(overrides, :mae)


## :maximize
# Showing duplicate methods for maximize in packages Module[Convex, Optim]
# Methods for maximize in package Convex
# maximize(objective::Convex.AbstractExpr, constraints::Constraint...; numeric_type) @ Convex ~/.julia/packages/Convex/-----/src/problems.jl:231
# maximize(objective::Convex.AbstractExpr, constraints; numeric_type) @ Convex ~/.julia/packages/Convex/-----/src/problems.jl:239
# maximize(objective::Convex.AbstractExpr; ...) @ Convex ~/.julia/packages/Convex/-----/src/problems.jl:239
# maximize(objective::Union{Number, AbstractArray}, constraints::Constraint...; numeric_type) @ Convex ~/.julia/packages/Convex/-----/src/problems.jl:247
# maximize(objective::Union{Number, AbstractArray}, constraints; numeric_type) @ Convex ~/.julia/packages/Convex/-----/src/problems.jl:255
# maximize(objective::Union{Number, AbstractArray}; ...) @ Convex ~/.julia/packages/Convex/-----/src/problems.jl:255
# Methods for maximize in package Optim
# maximize(f, g, h, x0::AbstractArray, method::Optim.AbstractOptimizer, options; kwargs...) @ Optim ~/.julia/packages/Optim/-----/src/maximize.jl:38
# maximize(f, g, h, x0::AbstractArray, method::Optim.AbstractOptimizer; ...) @ Optim ~/.julia/packages/Optim/-----/src/maximize.jl:38
# maximize(f, g, x0::AbstractArray, method::Optim.AbstractOptimizer, options; kwargs...) @ Optim ~/.julia/packages/Optim/-----/src/maximize.jl:32
# maximize(f, g, x0::AbstractArray, method::Optim.AbstractOptimizer; ...) @ Optim ~/.julia/packages/Optim/-----/src/maximize.jl:32
# maximize(f, lb::Real, ub::Real, method::Optim.AbstractOptimizer; kwargs...) @ Optim ~/.julia/packages/Optim/-----/src/maximize.jl:11
# maximize(f, lb::Real, ub::Real; kwargs...) @ Optim ~/.julia/packages/Optim/-----/src/maximize.jl:16
# maximize(f, x0::AbstractArray, method::Optim.AbstractOptimizer, options; kwargs...) @ Optim ~/.julia/packages/Optim/-----/src/maximize.jl:28
# maximize(f, x0::AbstractArray, method::Optim.AbstractOptimizer; ...) @ Optim ~/.julia/packages/Optim/-----/src/maximize.jl:28
# maximize(f, x0::AbstractArray; kwargs...) @ Optim ~/.julia/packages/Optim/-----/src/maximize.jl:24

@doc (@doc Convex.maximize)
maximize(objective::Convex.AbstractExpr, args::Constraint...; kwargs...) = Convex.maximize(objective, args...; kwargs...)
maximize(objective::Convex.AbstractExpr, constraints::AbstractArray{Constraint}; kwargs...) = Convex.maximize(objective, constraints; kwargs...)
maximize(objective::Union{Number, AbstractArray}, constraints::Constraint...; kwargs...) = Convex.maximize(objective, constraints...; kwargs...)
maximize(objective::Union{Number, AbstractArray}, constraints::AbstractArray{Constraint}; kwargs...) = Convex.maximize(objective, constraints...; kwargs...)
@doc (@doc Optim.maximize)
maximize(f, x0::AbstractArray; kwargs...) = Optim.maximize(f, x0; kwargs...)
maximize(f, x0::AbstractArray, method::Optim.AbstractOptimizer; kwargs...) = Optim.maximize(f, x0, method; kwargs...)
maximize(f, x0::AbstractArray, method::Optim.AbstractOptimizer, options; kwargs...) = Optim.maximize(f, x0, method, options; kwargs...)
maximize(f, g, x0::AbstractArray, method::Optim.AbstractOptimizer; kwargs...) = Optim.maximize(f, g, x0, method; kwargs...)
maximize(f, g, x0::AbstractArray, method::Optim.AbstractOptimizer, options; kwargs...) = Optim.maximize(f, g, x0, method, options; kwargs...)
maximize(f, g, h, x0::AbstractArray, method::Optim.AbstractOptimizer; kwargs...) = Optim.maximize(f, g, h, x0, method; kwargs...)
maximize(f, g, h, x0::AbstractArray, method::Optim.AbstractOptimizer, options; kwargs...) = Optim.maximize(f, g, h, x0, method, options; kwargs...)
maximize(f, lb::Real, ub::Real; kwargs...) = Optim.maximize(f, lb, ub; kwargs...)
maximize(f, lb::Real, ub::Real, method::Optim.AbstractOptimizer; kwargs...) = Optim.maximize(f, lb, ub, method; kwargs...)
export maximize
push!(overrides, :maximize)




## :meanad
# Showing duplicate methods for meanad in packages Module[Distances, StatsBase]
# Methods for MeanAbsDeviation() in package Distances
# (::MeanAbsDeviation)(a, b) @ Distances ~/.julia/packages/Distances/-----/src/metrics.jl:590
# Methods for meanad in package StatsBase
# meanad(a::AbstractArray{T}, b::AbstractArray{T}) where T<:Number @ StatsBase ~/.julia/packages/StatsBase/-----/src/deviation.jl:140
meanad = Distances.MeanAbsDeviation()
export meanad
push!(overrides, :meanad)

## :metadata
# Showing duplicate methods for metadata in packages Module[DataFrames, FileIO, RDatasets, SampledSignals]
# Methods for metadata in package DataAPI
# metadata(df::DataFrame, key::AbstractString, default; style) @ DataFrames ~/.julia/packages/DataFrames/-----/src/other/metadata.jl:102
# metadata(df::DataFrame, key::AbstractString; ...) @ DataFrames ~/.julia/packages/DataFrames/-----/src/other/metadata.jl:102
# metadata(x::T; style) where T @ DataAPI ~/.julia/packages/DataAPI/-----/src/DataAPI.jl:371
# metadata(x::Union{DataFrameRow, SubDataFrame}, key::AbstractString, default; style) @ DataFrames ~/.julia/packages/DataFrames/-----/src/other/metadata.jl:119
# metadata(x::Union{DataFrameRow, SubDataFrame}, key::AbstractString; ...) @ DataFrames ~/.julia/packages/DataFrames/-----/src/other/metadata.jl:119
# metadata(x::Union{DataFrames.DataFrameColumns, DataFrames.DataFrameRows}, key::AbstractString, default; style) @ DataFrames ~/.julia/packages/DataFrames/-----/src/other/metadata.jl:115
# metadata(x::Union{DataFrames.DataFrameColumns, DataFrames.DataFrameRows}, key::AbstractString; ...) @ DataFrames ~/.julia/packages/DataFrames/-----/src/other/metadata.jl:115
# Methods for metadata in package FileIO
# metadata(file, args...; options...) @ FileIO ~/.julia/packages/FileIO/-----/src/loadsave.jl:109
# metadata(file::Formatted, args...; options...) @ FileIO ~/.julia/packages/FileIO/-----/src/loadsave.jl:116
# Methods for metadata in package SampledSignals
@doc (@doc DataFrames.metadata)
#metadata(x; style) = DataFrames.metadata(x; style)
metadata(df::Union{DataFrame, DataFrames.DataFrameColumns, DataFrames.DataFrameRows, DataFrameRow, SubDataFrame}, key::AbstractString; kwargs...) = DataFrames.metadata(df, key; kwargs...)
metadata(df::Union{DataFrame, DataFrames.DataFrameColumns, DataFrames.DataFrameRows, DataFrameRow, SubDataFrame}, key::AbstractString, default; style) = DataFrames.metadata(df, key, default; style)
@doc (@doc FileIO.metadata)
metadata(file::FileIO.Formatted, args...; options...) = FileIO.metadata(file, args...; options...)
export metadata
push!(overrides, :metadata)

## :mode
# Showing duplicate methods for mode in packages Module[Distributions, JuMP, StatsBase]
# Methods for mode in package StatsBase
# mode(::Exponential{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/exponential.jl:58
# mode(a) @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:111
# mode(a::AbstractArray{T}, r::UnitRange{T}) where T<:Integer @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:56
# mode(a::AbstractVector, wv::AbstractWeights{T, T1, V} where {T1<:Real, V<:AbstractVector{T1}}) where T<:Real @ StatsBase ~/.julia/packages/StatsBase/-----/src/scalarstats.jl:164
# mode(d::AbstractMvNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvnormal.jl:85
# mode(d::Arcsine) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/arcsine.jl:65
# mode(d::Bernoulli) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/bernoulli.jl:67
# mode(d::BernoulliLogit) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/bernoullilogit.jl:54
# mode(d::Beta; check_args) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/beta.jl:67
# mode(d::BetaBinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/betabinomial.jl:111
# mode(d::BetaPrime{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/betaprime.jl:68
# mode(d::Binomial{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/binomial.jl:70
# mode(d::Biweight) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/biweight.jl:28
# mode(d::Cauchy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/cauchy.jl:62
# mode(d::Chernoff) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chernoff.jl:209
# mode(d::Chi; check_args) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chi.jl:73
# mode(d::Chisq{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chisq.jl:58
# mode(d::Cosine) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/cosine.jl:51
# mode(d::Dirac) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/dirac.jl:36
# mode(d::Dirichlet) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/dirichlet.jl:134
# mode(d::DiscreteNonParametric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/discretenonparametric.jl:206
# mode(d::DiscreteUniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/discreteuniform.jl:67
# mode(d::Distributions.AffineDistribution) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/locationscale.jl:111
# mode(d::Distributions.DirichletCanon) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/dirichlet.jl:135
# mode(d::Distributions.GenericMvTDist) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvtdist.jl:92
# mode(d::Distributions.Normal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:69
# mode(d::Distributions.ReshapedDistribution) @ Distributions ~/.julia/packages/Distributions/-----/src/reshaped.jl:44
# mode(d::Epanechnikov) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/epanechnikov.jl:40
# mode(d::Erlang; check_args) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/erlang.jl:65
# mode(d::FDist{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/fdist.jl:58
# mode(d::FisherNoncentralHypergeometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/noncentralhypergeometric.jl:74
# mode(d::Frechet) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/frechet.jl:67
# mode(d::Gamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gamma.jl:69
# mode(d::GeneralizedExtremeValue) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/generalizedextremevalue.jl:105
# mode(d::Geometric{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/geometric.jl:60
# mode(d::Gumbel) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gumbel.jl:74
# mode(d::Hypergeometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/hypergeometric.jl:56
# mode(d::InverseGamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/inversegamma.jl:67
# mode(d::InverseGaussian) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/inversegaussian.jl:70
# mode(d::InverseWishart) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/inversewishart.jl:91
# mode(d::Kolmogorov) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/kolmogorov.jl:26
# mode(d::Kumaraswamy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/kumaraswamy.jl:134
# mode(d::LKJ; check_args) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/lkj.jl:78
# mode(d::LKJCholesky) @ Distributions ~/.julia/packages/Distributions/-----/src/cholesky/lkjcholesky.jl:112
# mode(d::Laplace) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/laplace.jl:65
# mode(d::Levy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/levy.jl:61
# mode(d::Lindley) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/lindley.jl:57
# mode(d::LogNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/lognormal.jl:64
# mode(d::LogUniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/loguniform.jl:47
# mode(d::Logistic) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/logistic.jl:66
# mode(d::MatrixNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/matrixnormal.jl:91
# mode(d::MatrixTDist) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/matrixtdist.jl:113
# mode(d::MvLogNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvlognormal.jl:224
# mode(d::NegativeBinomial{T}) where T @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/negativebinomial.jl:81
# mode(d::NormalCanon) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normalcanon.jl:49
# mode(d::PGeneralizedGaussian) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pgeneralizedgaussian.jl:87
# mode(d::Pareto) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pareto.jl:63
# mode(d::Poisson) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/poisson.jl:55
# mode(d::PoissonBinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/poissonbinomial.jl:112
# mode(d::Rayleigh) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/rayleigh.jl:58
# mode(d::Rician) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/rician.jl:72
# mode(d::Semicircle) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/semicircle.jl:42
# mode(d::SkewNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/skewnormal.jl:62
# mode(d::SkewedExponentialPower) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/skewedexponentialpower.jl:80
# mode(d::SymTriangularDist) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/symtriangular.jl:60
# mode(d::TDist{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/tdist.jl:53
# mode(d::TriangularDist) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/triangular.jl:64
# mode(d::Triweight) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/triweight.jl:38
# mode(d::Truncated{var"#s689", Continuous, T, TL, TU} where {var"#s689"<:(Distributions.Normal), TL<:Union{Nothing, T}, TU<:Union{Nothing, T}}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/truncated/normal.jl:3
# mode(d::Uniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/uniform.jl:61
# mode(d::VonMises) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/vonmises.jl:57
# mode(d::WalleniusNoncentralHypergeometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/noncentralhypergeometric.jl:264
# mode(d::Weibull{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/weibull.jl:65
# mode(d::Wishart) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/wishart.jl:111
# Methods for mode in package JuMP
# mode(model::GenericModel) @ JuMP ~/.julia/packages/JuMP/-----/src/JuMP.jl:599
@doc (@doc Distributions.mode)
mode(x::Distribution) = Distributions.mode(x)
mode(a::AbstractArray{T}, r::UnitRange{T}) where T = StatsBase.mode(a, r)
mode(a::AbstractVector, wv::AbstractWeights) = StatsBase.mode(a, wv)
mode(a::AbstractArray) = StatsBase.mode(a)
#mode(x::MultivariateDistribution) = Distributions.mode(x)
#mode(x::MatrixDistribution) = Distributions.mode(x)
@doc (@doc JuMP.mode)
mode(model::JuMP.GenericModel) = JuMP.mode(model)
export mode
push!(overrides, :mode)

## :msd
# Showing duplicate methods for msd in packages Module[Distances, StatsBase]
# Methods for MeanSqDeviation() in package Distances
# (::MeanSqDeviation)(a, b) @ Distances ~/.julia/packages/Distances/-----/src/metrics.jl:593
# Methods for msd in package StatsBase
# msd(a::AbstractArray{T}, b::AbstractArray{T}) where T<:Number @ StatsBase ~/.julia/packages/StatsBase/-----/src/deviation.jl:159
msd = Distances.MeanSqDeviation()
export msd
push!(overrides, :msd)

## :mse
# Showing duplicate methods for mse in packages Module[Flux, Images, LsqFit]
# Methods for mse in package Flux.Losses
# mse(ŷ, y; agg) @ Flux.Losses ~/.julia/packages/Flux/-----/src/losses/functions.jl:45
# Methods for mse in package ImageDistances
# mse(a::AbstractArray{T} where T<:Union{Number, Colorant}, b::AbstractArray{T} where T<:Union{Number, Colorant}) @ ImageDistances ~/.julia/packages/ImageDistances/-----/src/metrics.jl:101
# Methods for mse in package LsqFit
# mse(lfr::LsqFit.LsqFitResult) @ LsqFit ~/.julia/packages/LsqFit/-----/src/curve_fit.jl:16

mse = Flux.mse
export mse
push!(overrides, :mse)

## :nan
# Showing duplicate methods for nan in packages Module[ColorVectorSpace, DoubleFloats, Images]
# Methods for nan in package ColorTypes
# nan(::Type{C}) where {T<:AbstractFloat, C<:(Colorant{T})} @ ColorTypes ~/.julia/packages/ColorTypes/-----/src/traits.jl:470
# nan(::Type{T}) where T<:AbstractFloat @ ColorTypes ~/.julia/packages/ColorTypes/-----/src/traits.jl:469
# Methods for nan in package DoubleFloats
# nan(::Type{DoubleFloat{T}}) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/-----/src/type/specialvalues.jl:4
@doc (@doc Images.nan)
nan(::Type{C}) where {T<:AbstractFloat, C<:(Colorant{T})} = Images.nan(C)
@doc (@doc DoubleFloats.nan)
nan(::Type{DoubleFloat{T}}) where T<:Union{Float16, Float32, Float64} = DoubleFloats.nan(T)
export nan
push!(overrides, :nan)

## :orthogonal
# Showing duplicate methods for orthogonal in packages Module[Flux, ReinforcementLearning]
# Methods for orthogonal in package Flux
# orthogonal(; ...) @ Flux ~/.julia/packages/Flux/-----/src/utils.jl:308
# orthogonal(dims::Integer...; kwargs...) @ Flux ~/.julia/packages/Flux/-----/src/utils.jl:307
# orthogonal(rng::AbstractRNG, d1::Integer, ds::Integer...; kwargs...) @ Flux ~/.julia/packages/Flux/-----/src/utils.jl:300
# orthogonal(rng::AbstractRNG, rows::Integer, cols::Integer; gain) @ Flux ~/.julia/packages/Flux/-----/src/utils.jl:290
# orthogonal(rng::AbstractRNG; init_kwargs...) @ Flux ~/.julia/packages/Flux/-----/src/utils.jl:308
# Methods for orthogonal in package ReinforcementLearningCore
# orthogonal(dims...) @ ReinforcementLearningCore ~/.julia/packages/ReinforcementLearningCore/-----/src/utils/basic.jl:50
# orthogonal(rng::AbstractRNG) @ ReinforcementLearningCore ~/.julia/packages/ReinforcementLearningCore/-----/src/utils/basic.jl:51
# orthogonal(rng::AbstractRNG, d1, rest_dims...) @ ReinforcementLearningCore ~/.julia/packages/ReinforcementLearningCore/-----/src/utils/basic.jl:45

# These two appear to be almost exactly the same. 
orthogonal = Flux.orthogonal 
export orthogonal
push!(overrides, :orthogonal)

## :params
# Showing duplicate methods for params in packages Module[BenchmarkTools, Distributions, Flux]
# Methods for params in package BenchmarkTools
# params(b::BenchmarkTools.Benchmark) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/-----/src/execution.jl:23
# params(group::BenchmarkGroup) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/-----/src/groups.jl:119
# params(t::BenchmarkTools.Trial) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/-----/src/trials.jl:61
# params(t::BenchmarkTools.TrialEstimate) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/-----/src/trials.jl:142
# params(t::BenchmarkTools.TrialJudgement) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/-----/src/trials.jl:219
# params(t::BenchmarkTools.TrialRatio) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/-----/src/trials.jl:170
# Methods for params in package StatsAPI
# params(::Type{D}, m::AbstractVector, S::AbstractMatrix) where D<:AbstractMvLogNormal @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvlognormal.jl:141
# params(d::Arcsine) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/arcsine.jl:57
# params(d::Bernoulli) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/bernoulli.jl:55
# params(d::BernoulliLogit) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/bernoullilogit.jl:40
# params(d::Beta) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/beta.jl:59
# params(d::BetaBinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/betabinomial.jl:52
# params(d::BetaPrime) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/betaprime.jl:59
# params(d::Binomial) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/binomial.jl:62
# params(d::Biweight) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/biweight.jl:22
# params(d::Cauchy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/cauchy.jl:54
# params(d::Chi) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chi.jl:46
# params(d::Chisq) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/chisq.jl:40
# params(d::Cosine) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/cosine.jl:41
# params(d::Dirichlet) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/dirichlet.jl:74
# params(d::DirichletMultinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/dirichletmultinomial.jl:56
# params(d::DiscreteNonParametric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/discretenonparametric.jl:50
# params(d::DiscreteUniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/discreteuniform.jl:43
# params(d::Distributions.AffineDistribution) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/locationscale.jl:104
# params(d::Distributions.Categorical{P, Ps}) where {P<:Real, Ps<:AbstractVector{P}} @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/categorical.jl:52
# params(d::Distributions.Censored) @ Distributions ~/.julia/packages/Distributions/-----/src/censored.jl:106
# params(d::Distributions.GenericMvTDist) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvtdist.jl:102
# params(d::Distributions.Normal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:57
# params(d::Distributions.ReshapedDistribution) @ Distributions ~/.julia/packages/Distributions/-----/src/reshaped.jl:30
# params(d::Epanechnikov) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/epanechnikov.jl:34
# params(d::Erlang) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/erlang.jl:55
# params(d::Exponential) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/exponential.jl:51
# params(d::FDist) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/fdist.jl:50
# params(d::Frechet) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/frechet.jl:54
# params(d::Gamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gamma.jl:56
# params(d::GeneralizedExtremeValue) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/generalizedextremevalue.jl:74
# params(d::GeneralizedPareto) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/generalizedpareto.jl:80
# params(d::Geometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/geometric.jl:50
# params(d::Gumbel) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gumbel.jl:56
# params(d::Hypergeometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/hypergeometric.jl:44
# params(d::InverseGamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/inversegamma.jl:59
# params(d::InverseGaussian) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/inversegaussian.jl:57
# params(d::InverseWishart) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/inversewishart.jl:80
# params(d::JohnsonSU) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/johnsonsu.jl:53
# params(d::JointOrderStatistics) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/jointorderstatistics.jl:89
# params(d::Kolmogorov) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/kolmogorov.jl:17
# params(d::Kumaraswamy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/kumaraswamy.jl:45
# params(d::LKJ) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/lkj.jl:94
# params(d::LKJCholesky) @ Distributions ~/.julia/packages/Distributions/-----/src/cholesky/lkjcholesky.jl:117
# params(d::Laplace) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/laplace.jl:57
# params(d::Levy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/levy.jl:50
# params(d::Lindley) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/lindley.jl:44
# params(d::LogNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/lognormal.jl:53
# params(d::LogUniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/loguniform.jl:33
# params(d::Logistic) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/logistic.jl:58
# params(d::LogitNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/logitnormal.jl:86
# params(d::MatrixBeta) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/matrixbeta.jl:81
# params(d::MatrixFDist) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/matrixfdist.jl:87
# params(d::MatrixNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/matrixnormal.jl:99
# params(d::MatrixTDist) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/matrixtdist.jl:121
# params(d::MixtureModel) @ Distributions ~/.julia/packages/Distributions/-----/src/mixtures/mixturemodel.jl:167
# params(d::Multinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/multinomial.jl:49
# params(d::MvLogNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvlognormal.jl:192
# params(d::MvLogitNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvlogitnormal.jl:57
# params(d::MvNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvnormal.jl:255
# params(d::MvNormalCanon) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvnormalcanon.jl:155
# params(d::NegativeBinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/negativebinomial.jl:62
# params(d::NoncentralBeta) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/noncentralbeta.jl:32
# params(d::NoncentralChisq) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/noncentralchisq.jl:54
# params(d::NoncentralF) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/noncentralf.jl:35
# params(d::NoncentralHypergeometric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/noncentralhypergeometric.jl:31
# params(d::NoncentralT) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/noncentralt.jl:29
# params(d::NormalCanon) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normalcanon.jl:42
# params(d::NormalInverseGaussian) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normalinversegaussian.jl:47
# params(d::OrderStatistic) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/orderstatistic.jl:59
# params(d::PGeneralizedGaussian) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pgeneralizedgaussian.jl:77
# params(d::Pareto) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pareto.jl:52
# params(d::Poisson) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/poisson.jl:46
# params(d::PoissonBinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/poissonbinomial.jl:80
# params(d::Rayleigh) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/rayleigh.jl:50
# params(d::Rician) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/rician.jl:61
# params(d::Semicircle) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/semicircle.jl:36
# params(d::Skellam) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/skellam.jl:58
# params(d::SkewNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/skewnormal.jl:46
# params(d::SkewedExponentialPower) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/skewedexponentialpower.jl:61
# params(d::StudentizedRange) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/studentizedrange.jl:59
# params(d::SymTriangularDist) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/symtriangular.jl:52
# params(d::TDist) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/tdist.jl:45
# params(d::TriangularDist) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/triangular.jl:58
# params(d::Triweight) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/triweight.jl:31
# params(d::Truncated) @ Distributions ~/.julia/packages/Distributions/-----/src/truncate.jl:126
# params(d::Uniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/uniform.jl:50
# params(d::UnivariateGMM) @ Distributions ~/.julia/packages/Distributions/-----/src/mixtures/unigmm.jl:33
# params(d::VonMises) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/vonmises.jl:49
# params(d::VonMisesFisher) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/vonmisesfisher.jl:59
# params(d::Weibull) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/weibull.jl:57
# params(d::Wishart) @ Distributions ~/.julia/packages/Distributions/-----/src/matrix/wishart.jl:106
# params(Ω::Soliton) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/soliton.jl:87
# Methods for params in package Flux
# params(m...) @ Flux ~/.julia/packages/Flux/-----/src/functor.jl:125
@doc (@doc Distributions.params)
params(x::Distribution) = Distributions.params(x)
@doc (@doc BenchmarkTools.params)
params(x::BenchmarkTools.Benchmark) = BenchmarkTools.params(x)
params(x::BenchmarkGroup) = BenchmarkTools.params(x)
params(x::BenchmarkTools.TrialJudgement) = BenchmarkTools.params(x)
params(x::BenchmarkTools.TrialRatio) = BenchmarkTools.params(x)
params(x::BenchmarkTools.TrialEstimate) = BenchmarkTools.params(x)
params(x::BenchmarkTools.Trial) = BenchmarkTools.params(x)
export params
push!(overrides, :params)

## :probs
# Showing duplicate methods for probs in packages Module[Distributions, OnlineStats]
# Methods for probs in package Distributions
# probs(d::DiscreteNonParametric) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/discrete/discretenonparametric.jl:64
# probs(d::Distribution{Univariate, Discrete}) @ Distributions ~/.julia/packages/Distributions/-----/src/deprecates.jl:5
# probs(d::MixtureModel) @ Distributions ~/.julia/packages/Distributions/-----/src/mixtures/mixturemodel.jl:166
# probs(d::Multinomial) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/multinomial.jl:46
# probs(d::UnivariateGMM) @ Distributions ~/.julia/packages/Distributions/-----/src/mixtures/unigmm.jl:24
# Methods for probs in package OnlineStatsBase
# probs(o::CountMap) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:132
# probs(o::CountMap, kys) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:132
# probs(o::FastNode) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/fasttree.jl:20
# probs(o::NBClassifier) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/nbclassifier.jl:107
# probs(o::ProbMap) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:510
# probs(o::ProbMap, levels) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:510

@doc (@doc Distributions.probs)
probs(x::Distribution) = Distributions.probs(x)
@doc (@doc OnlineStats.probs)
probs(x::OnlineStats.CountMap) = OnlineStats.probs(x)
probs(x::OnlineStats.CountMap, kys) = OnlineStats.probs(x, kys)
probs(x::OnlineStats.FastNode) = OnlineStats.probs(x)
probs(x::OnlineStats.NBClassifier) = OnlineStats.probs(x)
probs(x::OnlineStats.ProbMap) = OnlineStats.probs(x)
probs(x::OnlineStats.ProbMap,levels) = OnlineStats.probs(x, levels)
export probs
push!(overrides, :probs)


## :properties
# Showing duplicate methods for properties in packages Module[Images, IterTools]
# Methods for properties in package ImageMetadata
# properties(img::ImageMeta) @ ImageMetadata ~/.julia/packages/ImageMetadata/-----/src/ImageMetadata.jl:291
# Methods for properties in package IterTools
# properties(x::T) where T @ IterTools ~/.julia/packages/IterTools/-----/src/IterTools.jl:962
@doc (@doc Images.properties)
properties(x::ImageMeta) = Images.properties(x)
@doc (@doc IterTools.properties)
properties(x) = IterTools.properties(x)
export properties
push!(overrides, :properties)

## :radius
# Showing duplicate methods for radius in packages Module[GeometryBasics, Graphs]
# Methods for radius in package GeometryBasics
# radius(c::Cylinder{N, T}) where {N, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/cylinders.jl:25
# radius(c::HyperSphere) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/spheres.jl:29
# Methods for radius in package Graphs
# radius(eccentricities::Vector) @ Graphs ~/.julia/packages/Graphs/-----/src/distance.jl:167
# radius(g::AbstractGraph) @ Graphs ~/.julia/packages/Graphs/-----/src/distance.jl:168
# radius(g::AbstractGraph, distmx::AbstractMatrix) @ Graphs ~/.julia/packages/Graphs/-----/src/distance.jl:168
@doc (@doc GeometryBasics.radius)
radius(x::Cylinder) = GeometryBasics.radius(x)
radius(x::HyperSphere) = GeometryBasics.radius(x)
@doc (@doc Graphs.radius)
radius(x::AbstractGraph) = Graphs.radius(x)
radius(x::AbstractGraph, y) = Graphs.radius(x, y)
export radius
push!(overrides, :radius)

## :reset!
# Showing duplicate methods for reset! in packages Module[DSP, DataStructures, ReinforcementLearning]
# Methods for reset! in package DSP.Filters
# reset!(kernel::DSP.Filters.FIRArbitrary) @ DSP.Filters ~/.julia/packages/DSP/-----/src/Filters/stream_filt.jl:260
# reset!(kernel::DSP.Filters.FIRDecimator) @ DSP.Filters ~/.julia/packages/DSP/-----/src/Filters/stream_filt.jl:255
# reset!(kernel::DSP.Filters.FIRKernel) @ DSP.Filters ~/.julia/packages/DSP/-----/src/Filters/stream_filt.jl:245
# reset!(kernel::DSP.Filters.FIRRational) @ DSP.Filters ~/.julia/packages/DSP/-----/src/Filters/stream_filt.jl:249
# reset!(self::FIRFilter) @ DSP.Filters ~/.julia/packages/DSP/-----/src/Filters/stream_filt.jl:270
# Methods for reset! in package DataStructures
# reset!(blk::DataStructures.DequeBlock{T}, front::Int64) where T @ DataStructures ~/.julia/packages/DataStructures/-----/src/deque.jl:40
# reset!(ct::Accumulator{<:Any, V}, x) where V @ DataStructures ~/.julia/packages/DataStructures/-----/src/accumulator.jl:134
# Methods for reset! in package ReinforcementLearningBase
# reset!(cache::ReinforcementLearningCore.SRT) @ ReinforcementLearningCore ~/.julia/packages/ReinforcementLearningCore/-----/src/policies/agent/agent_srt_cache.jl:61
# reset!(env::AbstractEnv) @ ReinforcementLearningBase none:0
# reset!(env::AcrobotEnv{T}) where T<:Number @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/3rd_party/AcrobotEnv.jl:91
# reset!(env::BitFlippingEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/BitFlippingEnv.jl:48
# reset!(env::CartPoleEnv{T}) where T @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/CartPoleEnv.jl:98
# reset!(env::GraphShortestPathEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/GraphShortestPathEnv.jl:65
# reset!(env::KuhnPokerEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/KuhnPokerEnv.jl:83
# reset!(env::MaxTimeoutEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/MaxTimeoutEnv.jl:25
# reset!(env::MontyHallEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/MontyHallEnv.jl:101
# reset!(env::MountainCarEnv{T}) where T @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/MountainCarEnv.jl:99
# reset!(env::MultiArmBanditsEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/MultiArmBanditsEnv.jl:75
# reset!(env::PendulumEnv{A, T}) where {A, T} @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/PendulumEnv.jl:84
# reset!(env::PendulumNonInteractiveEnv{Fl, VFl} where VFl<:AbstractVector{Fl}) where Fl @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/non_interactive/pendulum.jl:74
# reset!(env::PigEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/PigEnv.jl:31
# reset!(env::RandomWalk1D) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/RandomWalk1D.jl:44
# reset!(env::ReinforcementLearningBase.RLBaseEnv) @ ReinforcementLearningBase ~/.julia/packages/ReinforcementLearningBase/-----/src/CommonRLInterface.jl:88
# reset!(env::RockPaperScissorsEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/RockPaperScissorsEnv.jl:46
# reset!(env::StochasticEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/StochasticEnv.jl:18
# reset!(env::StockTradingEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/StockTradingEnv.jl:135
# reset!(env::TicTacToeEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TicTacToeEnv.jl:24
# reset!(env::TigerProblemEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TigerProblemEnv.jl:40
# reset!(env::TinyHanabiEnv) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TinyHanabiEnv.jl:41
# reset!(p::StackFrames{T, N}) where {T, N} @ ReinforcementLearningCore ~/.julia/packages/ReinforcementLearningCore/-----/src/utils/stack_frames.jl:32
# reset!(s::StopAfterNSeconds) @ ReinforcementLearningCore ~/.julia/packages/ReinforcementLearningCore/-----/src/core/stop_conditions.jl:215
# reset!(x::AbstractEnvWrapper, args...; kwargs...) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/wrappers.jl:17
@doc (@doc DataStructures.reset!)
reset!(x::Accumulator, y) = DataStructures.reset!(x, y)
reset!(x::DataStructures.DequeBlock, y) = DataStructures.reset!(x, y)
@doc (@doc ReinforcementLearning.reset!)
reset!(x::AbstractEnv) = ReinforcementLearning.reset!(x)
@doc (@doc DSP.Filters.reset!)
reset!(x::Union{FIRFilter, DSP.Filters.FIRKernel, DSP.Filters.FIRArbitrary, DSP.Filters.FIRRational, DSP.Filters.FIRDecimator}) = DSP.reset!(x)
export reset!
push!(overrides, :reset!)

## :right
# Showing duplicate methods for right in packages Module[CairoMakie, Transducers]
# Methods for right in package Makie
# right(rect::Rect2) @ Makie ~/.julia/packages/Makie/-----/src/makielayout/geometrybasics_extension.jl:3
# Methods for right in package Transducers
# right(::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(Transducers.right)}}, x) @ Transducers ~/.julia/packages/InitialValues/-----/src/InitialValues.jl:154
# right(::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(Transducers.right)}}, x::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(Transducers.right)}}) @ Transducers ~/.julia/packages/InitialValues/-----/src/InitialValues.jl:160
# right(l, r) @ Transducers ~/.julia/packages/Transducers/-----/src/core.jl:866
# right(r) @ Transducers ~/.julia/packages/Transducers/-----/src/core.jl:867
# right(x, ::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(Transducers.right)}}) @ Transducers ~/.julia/packages/InitialValues/-----/src/InitialValues.jl:161
@doc (@doc Transducers.right)
right(x) = Transducers.right(x)
right(l,r) = Transducers.right(l,r)
@doc (@doc Makie.right)
right(x::Rect2) = Makie.right(x)

export right
push!(overrides, :right)

## :rmsd
# Showing duplicate methods for rmsd in packages Module[Distances, StatsBase]
# Methods for RMSDeviation() in package Distances
# (::RMSDeviation)(a, b) @ Distances ~/.julia/packages/Distances/-----/src/metrics.jl:596
# Methods for rmsd in package StatsBase
# rmsd(a::AbstractArray{T}, b::AbstractArray{T}; normalize) where T<:Number @ StatsBase ~/.julia/packages/StatsBase/-----/src/deviation.jl:171
rmsd = Distances.RMSDeviation()
export rmsd
push!(overrides, :rmsd)

## :rotate!
# Showing duplicate methods for rotate! in packages Module[CairoMakie, LinearAlgebra]
# Methods for rotate! in package Makie
# rotate!(::Type{T}, t::MakieCore.Transformable, axis_rot...) where T @ Makie ~/.julia/packages/Makie/-----/src/layouting/transformation.jl:115
# rotate!(::Type{T}, t::MakieCore.Transformable, q) where T @ Makie ~/.julia/packages/Makie/-----/src/layouting/transformation.jl:98
# rotate!(l::RectLight, q...) @ Makie ~/.julia/packages/Makie/-----/src/lighting.jl:194
# rotate!(t::MakieCore.Transformable, axis_rot...) @ Makie ~/.julia/packages/Makie/-----/src/layouting/transformation.jl:124
# rotate!(t::MakieCore.Transformable, axis_rot::AbstractFloat) @ Makie ~/.julia/packages/Makie/-----/src/layouting/transformation.jl:126
# rotate!(t::MakieCore.Transformable, axis_rot::Quaternion) @ Makie ~/.julia/packages/Makie/-----/src/layouting/transformation.jl:125
# Methods for rotate! in package LinearAlgebra
# rotate!(x::AbstractVector, y::AbstractVector, c, s) @ LinearAlgebra /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/LinearAlgebra/src/generic.jl:1548
# rotate!(x::GPUArraysCore.AbstractGPUArray, y::GPUArraysCore.AbstractGPUArray, c::Number, s::Number) @ GPUArrays ~/.julia/packages/GPUArrays/-----/src/host/linalg.jl:688
@doc (@doc LinearAlgebra.rotate!)
rotate!(x::AbstractVector, y::AbstractVector, c, s) = LinearAlgebra.rotate!(x, y, c, s)
@doc (@doc Makie.rotate!)
rotate!(x::RectLight, y...) = Makie.rotate!(x, y...)
rotate!(x::Makie.Transformable, y) = Makie.rotate!(x, y)
rotate!(::Type{T}, t::Makie.Transformable, y...) where T = Makie.rotate!(T, t, y...)
export rotate!
push!(overrides, :rotate!)

## :scale
# Showing duplicate methods for scale in packages Module[Distributions, Interpolations]
# Methods for scale in package Distributions
# scale(::Type{D}, s::Symbol, m::AbstractVector, S::AbstractMatrix) where D<:AbstractMvLogNormal @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvlognormal.jl:123
# scale(d::Arcsine) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/arcsine.jl:56
# scale(d::Cauchy) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/cauchy.jl:52
# scale(d::Cosine) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/cosine.jl:39
# scale(d::Distributions.AffineDistribution) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/locationscale.jl:103
# scale(d::Distributions.GenericMvTDist) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvtdist.jl:96
# scale(d::Distributions.Normal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normal.jl:61
# scale(d::Epanechnikov) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/epanechnikov.jl:33
# scale(d::Erlang) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/erlang.jl:53
# scale(d::Exponential) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/exponential.jl:48
# scale(d::Frechet) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/frechet.jl:53
# scale(d::Gamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gamma.jl:53
# scale(d::GeneralizedExtremeValue) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/generalizedextremevalue.jl:72
# scale(d::GeneralizedPareto) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/generalizedpareto.jl:78
# scale(d::Gumbel) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gumbel.jl:55
# scale(d::InverseGamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/inversegamma.jl:56
# scale(d::JohnsonSU) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/johnsonsu.jl:51
# scale(d::Laplace) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/laplace.jl:56
# scale(d::Logistic) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/logistic.jl:56
# scale(d::LogitNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/logitnormal.jl:88
# scale(d::MvLogNormal) @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvlognormal.jl:207
# scale(d::NormalCanon) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/normalcanon.jl:60
# scale(d::PGeneralizedGaussian) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pgeneralizedgaussian.jl:80
# scale(d::Pareto) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pareto.jl:50
# scale(d::Rayleigh) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/rayleigh.jl:49
# scale(d::Rician) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/rician.jl:59
# scale(d::SkewedExponentialPower) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/skewedexponentialpower.jl:64
# scale(d::SymTriangularDist) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/symtriangular.jl:50
# scale(d::Triweight) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/triweight.jl:30
# scale(d::Uniform) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/uniform.jl:54
# scale(d::Weibull) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/weibull.jl:55
# Methods for scale in package Interpolations
# scale(itp::AbstractInterpolation{T, N, IT}, ranges::NTuple{N, var"#s217"} where var"#s217"<:AbstractRange) where {T, N, IT} @ Interpolations ~/.julia/packages/Interpolations/-----/src/scaling/scaling.jl:31
# scale(itp::AbstractInterpolation{T, N, IT}, ranges::Vararg{AbstractRange, N}) where {T, N, IT} @ Interpolations ~/.julia/packages/Interpolations/-----/src/scaling/scaling.jl:27

@doc (@doc Distributions.scale)
scale(x::Distributions.Distribution) = Distributions.scale(x)
#scale(::Type{D}, s::Symbol, m::AbstractVector, S::AbstractMatrix) where D<:Distributions.AbstractMvLogNormal = Distributions.scale(Type{D}, s, m, S)
@doc (@doc Interpolations.scale)
scale(x::AbstractInterpolation, y) = Interpolations.scale(x, y)
export scale
push!(overrides, :scale)

## :scale!
# Showing duplicate methods for scale! in packages Module[CairoMakie, Distributions]
# Methods for scale! in package Makie
# scale!(::Type{T}, l::RectLight, s) where T @ Makie ~/.julia/packages/Makie/-----/src/lighting.jl:201
# scale!(l::RectLight, x::Real, y::Real) @ Makie ~/.julia/packages/Makie/-----/src/lighting.jl:213
# scale!(l::RectLight, xy::Union{NTuple{N, T}, StaticArray{Tuple{N}, T, 1}} where {N, T}) @ Makie ~/.julia/packages/Makie/-----/src/lighting.jl:214
# scale!(t::MakieCore.Transformable, s) @ Makie ~/.julia/packages/Makie/-----/src/layouting/transformation.jl:82
# scale!(t::MakieCore.Transformable, xyz...) @ Makie ~/.julia/packages/Makie/-----/src/layouting/transformation.jl:94
# Methods for scale! in package Distributions
# scale!(::Type{D}, s::Symbol, m::AbstractVector, S::AbstractMatrix, Σ::AbstractMatrix) where D<:AbstractMvLogNormal @ Distributions ~/.julia/packages/Distributions/-----/src/multivariate/mvlognormal.jl:112
@doc (@doc Distributions.scale!)
scale!(::Type{D}, s::Symbol, m::AbstractVector, S::AbstractMatrix, Σ::AbstractMatrix) where D = Distributions.scale!(D, s, m, S, Σ)
@doc (@doc Makie.scale!)
scale!(x::Makie.Transformable, y...) = Makie.scale!(x, y...)
scale!(t::RectLight, xy) = Makie.scale!(t, xy)
scale!(t::RectLight, x, y) = Makie.scale!(t, x, y)
scale!(::Type{T}, t::RectLight, s) where T = Makie.scale!(T, t, )
export scale!
push!(overrides, :scale!)

## :shape
# Showing duplicate methods for shape in packages Module[Distributions, JuMP]
# Methods for shape in package Distributions
# shape(d::Erlang) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/erlang.jl:52
# shape(d::Frechet) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/frechet.jl:52
# shape(d::Gamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/gamma.jl:52
# shape(d::GeneralizedExtremeValue) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/generalizedextremevalue.jl:71
# shape(d::GeneralizedPareto) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/generalizedpareto.jl:79
# shape(d::InverseGamma) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/inversegamma.jl:55
# shape(d::InverseGaussian) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/inversegaussian.jl:56
# shape(d::JohnsonSU) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/johnsonsu.jl:50
# shape(d::Lindley) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/lindley.jl:43
# shape(d::PGeneralizedGaussian) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pgeneralizedgaussian.jl:79
# shape(d::Pareto) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/pareto.jl:49
# shape(d::Rician) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/rician.jl:58
# shape(d::SkewedExponentialPower) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/skewedexponentialpower.jl:63
# shape(d::Weibull) @ Distributions ~/.julia/packages/Distributions/-----/src/univariate/continuous/weibull.jl:54
# Methods for shape in package JuMP
# shape(::ScalarConstraint) @ JuMP ~/.julia/packages/JuMP/-----/src/constraints.jl:881
# shape(con::VectorConstraint) @ JuMP ~/.julia/packages/JuMP/-----/src/constraints.jl:978
@doc (@doc Distributions.shape)
shape(x) = Distributions.shape(x) # they get the generic function... 
@doc (@doc JuMP.shape)
shape(con::VectorConstraint) = JuMP.shape(con)
shape(x::ScalarConstraint) = JuMP.shape(x)
export shape
push!(overrides, :shape)

## :solve!
# Showing duplicate methods for solve! in packages Module[Convex, DifferentialEquations, Krylov, NonlinearSolve, Roots]
# Methods for solve! in package Convex
# solve!(p::Problem, optimizer_factory; silent, warmstart, silent_solver) @ Convex ~/.julia/packages/Convex/-----/src/solution.jl:79
# Methods for solve! in package CommonSolve
# solve!(P::Roots.ZeroProblemIterator; verbose) @ Roots ~/.julia/packages/Roots/-----/src/find_zero.jl:443
# solve!(P::Roots.ZeroProblemIterator{𝑴, 𝑵, 𝑭, 𝑺, 𝑶, 𝑳}; verbose) where {𝑴<:Bisection, 𝑵, 𝑭, 𝑺, 𝑶<:ExactOptions, 𝑳} @ Roots ~/.julia/packages/Roots/-----/src/Bracketing/bisection.jl:172
# solve!(cache::BoundaryValueDiffEq.BoundaryValueDiffEqFIRK.FIRKCacheExpand) @ BoundaryValueDiffEq.BoundaryValueDiffEqFIRK ~/.julia/packages/BoundaryValueDiffEq/eyGpq/lib/BoundaryValueDiffEqFIRK/src/firk.jl:303
# solve!(cache::BoundaryValueDiffEq.BoundaryValueDiffEqFIRK.FIRKCacheNested) @ BoundaryValueDiffEq.BoundaryValueDiffEqFIRK ~/.julia/packages/BoundaryValueDiffEq/eyGpq/lib/BoundaryValueDiffEqFIRK/src/firk.jl:329
# solve!(cache::BoundaryValueDiffEq.BoundaryValueDiffEqMIRK.MIRKCache) @ BoundaryValueDiffEq.BoundaryValueDiffEqMIRK ~/.julia/packages/BoundaryValueDiffEq/eyGpq/lib/BoundaryValueDiffEqMIRK/src/mirk.jl:134
# solve!(cache::LineSearch.LiFukushimaLineSearchCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/-----/src/li_fukushima.jl:96
# solve!(cache::LineSearch.LineSearchesJLCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/-----/src/line_searches_ext.jl:128
# solve!(cache::LineSearch.NoLineSearchCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/-----/src/no_search.jl:19
# solve!(cache::LineSearch.RobustNonMonotoneLineSearchCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/-----/src/robust_non_monotone.jl:90
# solve!(cache::LineSearch.StaticLiFukushimaLineSearchCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/-----/src/li_fukushima.jl:136
# solve!(cache::LinearSolve.LinearCache, alg::AppleAccelerateLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/appleaccelerate.jl:237
# solve!(cache::LinearSolve.LinearCache, alg::CHOLMODFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:967
# solve!(cache::LinearSolve.LinearCache, alg::DiagonalFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:1199
# solve!(cache::LinearSolve.LinearCache, alg::DirectLdiv!, args...; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/solve_function.jl:17
# solve!(cache::LinearSolve.LinearCache, alg::FastLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:1237
# solve!(cache::LinearSolve.LinearCache, alg::FastQRFactorization{P}; kwargs...) where P @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:1290
# solve!(cache::LinearSolve.LinearCache, alg::KLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:895
# solve!(cache::LinearSolve.LinearCache, alg::KrylovJL; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/iterative_wrappers.jl:227
# solve!(cache::LinearSolve.LinearCache, alg::LUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:79
# solve!(cache::LinearSolve.LinearCache, alg::LinearSolve.AbstractFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/LinearSolve.jl:151
# solve!(cache::LinearSolve.LinearCache, alg::LinearSolve.DefaultLinearSolver, args...; assump, kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/default.jl:356
# solve!(cache::LinearSolve.LinearCache, alg::LinearSolveFunction, args...; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/solve_function.jl:6
# solve!(cache::LinearSolve.LinearCache, alg::MKLLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/mkl.jl:213
# solve!(cache::LinearSolve.LinearCache, alg::NormalBunchKaufmanFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:1172
# solve!(cache::LinearSolve.LinearCache, alg::NormalCholeskyFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:1118
# solve!(cache::LinearSolve.LinearCache, alg::Nothing, args...; assump, kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/default.jl:298
# solve!(cache::LinearSolve.LinearCache, alg::RFLUFactorization{P, T}; kwargs...) where {P, T} @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:1031
# solve!(cache::LinearSolve.LinearCache, alg::SimpleGMRES; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/simplegmres.jl:148
# solve!(cache::LinearSolve.LinearCache, alg::SimpleLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/simplelu.jl:133
# solve!(cache::LinearSolve.LinearCache, alg::SparspakFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:1373
# solve!(cache::LinearSolve.LinearCache, alg::UMFPACKFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/factorization.jl:816
# solve!(cache::LinearSolve.LinearCache, args...; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/common.jl:273
# solve!(cache::LinearSolve.SimpleGMRESCache{false}, lincache::LinearSolve.LinearCache) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/simplegmres.jl:220
# solve!(cache::LinearSolve.SimpleGMRESCache{true}, lincache::LinearSolve.LinearCache) @ LinearSolve ~/.julia/packages/LinearSolve/-----/src/simplegmres.jl:450
# solve!(cache::NonlinearSolve.AbstractNonlinearSolveCache) @ NonlinearSolve ~/.julia/packages/NonlinearSolve/-----/src/core/generic.jl:11
# solve!(cache::NonlinearSolve.NonlinearSolveForwardDiffCache) @ NonlinearSolve ~/.julia/packages/NonlinearSolve/-----/src/internal/forward_diff.jl:51
# solve!(cache::NonlinearSolve.NonlinearSolveNoInitCache) @ NonlinearSolve ~/.julia/packages/NonlinearSolve/-----/src/core/noinit.jl:35
# solve!(cache::NonlinearSolve.NonlinearSolvePolyAlgorithmCache{iip, N}) where {iip, N} @ NonlinearSolve ~/.julia/packages/NonlinearSolve/-----/src/default.jl:141
# solve!(cache::SciMLBase.AbstractOptimizationCache) @ SciMLBase ~/.julia/packages/SciMLBase/-----/src/solve.jl:185
# solve!(integ::DiffEqBase.NullODEIntegrator) @ DiffEqBase ~/.julia/packages/DiffEqBase/-----/src/solve.jl:643
# solve!(integrator::DelayDiffEq.DDEIntegrator) @ DelayDiffEq ~/.julia/packages/DelayDiffEq/-----/src/solve.jl:545
# solve!(integrator::JumpProcesses.SSAIntegrator) @ JumpProcesses ~/.julia/packages/JumpProcesses/-----/src/SSA_stepper.jl:120
# solve!(integrator::OrdinaryDiffEqCore.ODEIntegrator) @ OrdinaryDiffEqCore ~/.julia/packages/OrdinaryDiffEqCore/-----/src/solve.jl:544
# solve!(integrator::StochasticDiffEq.SDEIntegrator) @ StochasticDiffEq ~/.julia/packages/StochasticDiffEq/-----/src/solve.jl:611
# solve!(integrator::Sundials.AbstractSundialsIntegrator; early_free, calculate_error) @ Sundials ~/.julia/packages/Sundials/-----/src/common_interface/solve.jl:1406
# solve!(𝐙::Roots.ZeroProblemIterator{𝐌, 𝐍}; verbose) where {𝐌, 𝐍<:AbstractBracketingMethod} @ Roots ~/.julia/packages/Roots/-----/src/hybrid.jl:30
# Methods for solve! in package Krylov
# solve!(solver::BicgstabSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::BicgstabSolver{T, FC, S}, A, b::AbstractVector{FC}; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::BilqSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; c, transfer_to_bicg, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::BilqSolver{T, FC, S}, A, b::AbstractVector{FC}; c, transfer_to_bicg, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::BilqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; transfer_to_bicg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::BilqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; transfer_to_bicg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::BlockGmresSolver{T, FC, SV, SM}, A, B::AbstractMatrix{FC}, X0::AbstractMatrix; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, SV<:AbstractVector{FC}, SM<:AbstractMatrix{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:201
# solve!(solver::BlockGmresSolver{T, FC, SV, SM}, A, B::AbstractMatrix{FC}; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, SV<:AbstractVector{FC}, SM<:AbstractMatrix{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:188
# solve!(solver::CarSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::CarSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CgLanczosShiftSolver{T, FC, S}, A, b::AbstractVector{FC}, shifts::AbstractVector{T}; M, ldiv, check_curvature, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CgLanczosSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, check_curvature, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::CgLanczosSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, check_curvature, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CgSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, radius, linesearch, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::CgSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, radius, linesearch, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CglsLanczosShiftSolver{T, FC, S}, A, b::AbstractVector{FC}, shifts::AbstractVector{T}; M, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CglsSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, radius, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CgneSolver{T, FC, S}, A, b::AbstractVector{FC}; N, ldiv, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CgsSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::CgsSolver{T, FC, S}, A, b::AbstractVector{FC}; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CrSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, radius, linesearch, γ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::CrSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, radius, linesearch, γ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CraigSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, transfer_to_lsqr, sqd, λ, btol, conlim, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CraigmrSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, sqd, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CrlsSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, radius, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::CrmrSolver{T, FC, S}, A, b::AbstractVector{FC}; N, ldiv, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::DiomSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::DiomSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::DqgmresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::DqgmresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::FgmresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::FgmresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::FomSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::FomSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::GmresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::GmresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::GpmrSolver{T, FC, S}, A, B, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; C, D, E, F, ldiv, gsp, λ, μ, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::GpmrSolver{T, FC, S}, A, B, b::AbstractVector{FC}, c::AbstractVector{FC}; C, D, E, F, ldiv, gsp, λ, μ, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::LnlqSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, transfer_to_craig, sqd, λ, σ, utolx, utoly, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::LslqSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, transfer_to_lsqr, sqd, λ, σ, etol, utol, btol, conlim, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::LsmrSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, sqd, λ, radius, etol, axtol, btol, conlim, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::LsqrSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, sqd, λ, radius, etol, axtol, btol, conlim, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::MinaresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, λ, atol, rtol, Artol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::MinaresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, λ, atol, rtol, Artol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::MinresQlpSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, λ, atol, rtol, Artol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::MinresQlpSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, λ, atol, rtol, Artol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::MinresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, λ, atol, rtol, etol, conlim, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::MinresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, λ, atol, rtol, etol, conlim, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::QmrSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::QmrSolver{T, FC, S}, A, b::AbstractVector{FC}; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::SymmlqSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, transfer_to_cg, λ, λest, atol, rtol, etol, conlim, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::SymmlqSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, transfer_to_cg, λ, λest, atol, rtol, etol, conlim, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::TricgSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; M, N, ldiv, spd, snd, flip, τ, ν, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::TricgSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; M, N, ldiv, spd, snd, flip, τ, ν, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::TrilqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; transfer_to_usymcg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::TrilqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; transfer_to_usymcg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::TrimrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; M, N, ldiv, spd, snd, flip, sp, τ, ν, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::TrimrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; M, N, ldiv, spd, snd, flip, sp, τ, ν, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::UsymlqSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector; transfer_to_usymcg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::UsymlqSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; transfer_to_usymcg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140
# solve!(solver::UsymqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector; atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:153
# solve!(solver::UsymqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solve.jl:140

@doc (@doc DifferentialEquations.solve!)
solve!(args...;kwargs...) = DifferentialEquations.solve!(args...;kwargs...)
@doc (@doc Krylov.solve!)
solve!(solver::KrylovSolver, args...; kwargs...) = Krylov.solve!(solver, args...; kwargs...)
@doc (@doc Convex.solve!)
solve!(p::Convex.Problem, opt; kwargs...) = Convex.solve!(p, opt; kwargs...)
export solve!
push!(overrides, :solve!)

## :spectrogram
# Showing duplicate methods for spectrogram in packages Module[DSP, Flux]
# Methods for spectrogram in package DSP.Periodograms
# spectrogram(s::AbstractVector{T}, n::Int64, noverlap::Int64; onesided, nfft, fs, window) where T @ DSP.Periodograms ~/.julia/packages/DSP/-----/src/periodograms.jl:420
# spectrogram(s::AbstractVector{T}, n::Int64; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/-----/src/periodograms.jl:420
# spectrogram(s::AbstractVector{T}; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/-----/src/periodograms.jl:420
# Methods for spectrogram in package NNlib
# spectrogram(waveform; pad, n_fft, hop_length, window, center, power, normalized, window_normalized) @ NNlib ~/.julia/packages/NNlib/-----/src/audio/spectrogram.jl:28
spectrogram = DSP.spectrogram # Method for spectrogram in package DSP
export spectrogram
push!(overrides, :spectrogram)

## :square
# Showing duplicate methods for square in packages Module[Convex, DoubleFloats]
# Methods for square in package Convex
# square(x::Convex.AbstractExpr) @ Convex ~/.julia/packages/Convex/-----/src/supported_operations.jl:2145
# Methods for square in package DoubleFloats
# square(x::Complex{DoubleFloat{T}}) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/-----/src/math/elementary/complex.jl:11
# square(x::DoubleFloat{T}) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/-----/src/math/ops/arith.jl:8

@doc (@doc DoubleFloats.square)
square(x::Complex{DoubleFloat}) = DoubleFloats.square(x)
square(x::DoubleFloat) = DoubleFloats.square(x)
@doc (@doc Convex.square)
square(x::Convex.AbstractExpr) = Convex.square(x)
export square
push!(overrides, :square)

## :state
# Showing duplicate methods for state in packages Module[Flux, ReinforcementLearning]
# Methods for state in package Flux
# state(x) @ Flux ~/.julia/packages/Flux/-----/src/loading.jl:173
# Methods for state in package ReinforcementLearningBase
# state(::RockPaperScissorsEnv, ::Observation, ::AbstractPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/RockPaperScissorsEnv.jl:40
# state(env::AbstractEnv) @ ReinforcementLearningBase ~/.julia/packages/ReinforcementLearningBase/-----/src/interface.jl:519
# state(env::AbstractEnv, player) @ ReinforcementLearningBase ~/.julia/packages/ReinforcementLearningBase/-----/src/interface.jl:521
# state(env::AbstractEnv, ss::ReinforcementLearningBase.AbstractStateStyle) @ ReinforcementLearningBase ~/.julia/packages/ReinforcementLearningBase/-----/src/interface.jl:520
# state(env::AbstractEnvWrapper, ss::ReinforcementLearningBase.AbstractStateStyle) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/wrappers.jl:23
# state(env::AbstractEnvWrapper, ss::ReinforcementLearningBase.AbstractStateStyle, player::Player) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/wrappers.jl:21
# state(env::AcrobotEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/3rd_party/AcrobotEnv.jl:88
# state(env::BitFlippingEnv, ::GoalState) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/BitFlippingEnv.jl:42
# state(env::BitFlippingEnv, ::Observation) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/BitFlippingEnv.jl:41
# state(env::BitFlippingEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/BitFlippingEnv.jl:40
# state(env::CartPoleEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/CartPoleEnv.jl:86
# state(env::DefaultStateStyleEnv, ss::ReinforcementLearningBase.AbstractStateStyle) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/DefaultStateStyle.jl:19
# state(env::DefaultStateStyleEnv{S}) where S @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/DefaultStateStyle.jl:21
# state(env::DefaultStateStyleEnv{S}, ::Observation, ::DefaultPlayer) where S @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/DefaultStateStyle.jl:18
# state(env::DefaultStateStyleEnv{S}, player::AbstractPlayer) where S @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/DefaultStateStyle.jl:20
# state(env::GraphShortestPathEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/GraphShortestPathEnv.jl:57
# state(env::KuhnPokerEnv, ::InformationSet{Tuple{Vararg{Symbol}}}, ::ChancePlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/KuhnPokerEnv.jl:101
# state(env::KuhnPokerEnv, ::InformationSet{Tuple{Vararg{Symbol}}}, player::Player) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/KuhnPokerEnv.jl:93
# state(env::MontyHallEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/MontyHallEnv.jl:61
# state(env::MountainCarEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/MountainCarEnv.jl:97
# state(env::MultiArmBanditsEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/MultiArmBanditsEnv.jl:71
# state(env::PendulumEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/PendulumEnv.jl:82
# state(env::PendulumNonInteractiveEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/non_interactive/pendulum.jl:71
# state(env::PigEnv, ::Observation{Vector{Int64}}, p::AbstractPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/PigEnv.jl:46
# state(env::RandomWalk1D, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/RandomWalk1D.jl:41
# state(env::ReinforcementLearningBase.RLBaseEnv, ::InternalState) @ ReinforcementLearningBase ~/.julia/packages/ReinforcementLearningBase/-----/src/CommonRLInterface.jl:79
# state(env::ReinforcementLearningBase.RLBaseEnv, ::Observation) @ ReinforcementLearningBase ~/.julia/packages/ReinforcementLearningBase/-----/src/CommonRLInterface.jl:78
# state(env::StateCachedEnv, args...; kwargs...) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/StateCachedEnv.jl:22
# state(env::StateTransformedEnv, args...; kwargs...) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/StateTransformedEnv.jl:18
# state(env::StockTradingEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/StockTradingEnv.jl:133
# state(env::TicTacToeEnv, ::Observation, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TicTacToeEnv.jl:73
# state(env::TicTacToeEnv, ::Observation{BitArray{3}}, player) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TicTacToeEnv.jl:74
# state(env::TicTacToeEnv, ::Observation{Int64}, player::Player) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TicTacToeEnv.jl:76
# state(env::TicTacToeEnv, ::Observation{String}) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TicTacToeEnv.jl:84
# state(env::TicTacToeEnv, ::Observation{String}, player::Player) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TicTacToeEnv.jl:86
# state(env::TicTacToeEnv, ::ReinforcementLearningBase.AbstractStateStyle) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TicTacToeEnv.jl:75
# state(env::TigerProblemEnv, ::InternalState, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TigerProblemEnv.jl:65
# state(env::TigerProblemEnv, ::Observation{Int64}, ::DefaultPlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TigerProblemEnv.jl:50
# state(env::TinyHanabiEnv, ::InformationSet, ::ChancePlayer) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TinyHanabiEnv.jl:81
# state(env::TinyHanabiEnv, ::InformationSet, player::Player) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/examples/TinyHanabiEnv.jl:90
# state(x::AbstractEnvWrapper, args...; kwargs...) @ ReinforcementLearningEnvironments ~/.julia/packages/ReinforcementLearningEnvironments/-----/src/environments/wrappers/wrappers.jl:17

state = ReinforcementLearning.state # Method for state in package ReinforcementLearningBase
export state
push!(overrides, :state)

## :statistics
# Showing duplicate methods for statistics in packages Module[DelaunayTriangulation, Krylov]
# Methods for statistics in package DelaunayTriangulation
# statistics(tri::Triangulation) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/-----/src/data_structures/statistics/triangulation_statistics.jl:84
# Methods for statistics in package Krylov
# statistics(solver::BicgstabSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::BilqSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::BilqrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::BlockGmresSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/block_krylov_solvers.jl:77
# statistics(solver::CarSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CgLanczosShiftSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CgLanczosSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CgSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CglsLanczosShiftSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CglsSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CgneSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CgsSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CraigSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CraigmrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CrlsSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::CrmrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::DiomSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::DqgmresSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::FgmresSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::FomSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::GmresSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::GpmrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::LnlqSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::LslqSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::LsmrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::LsqrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::MinaresSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::MinresQlpSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::MinresSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::QmrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::SymmlqSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::TricgSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::TrilqrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::TrimrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::UsymlqSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
# statistics(solver::UsymqrSolver) @ Krylov ~/.julia/packages/Krylov/-----/src/krylov_solvers.jl:2021
@doc (@doc Krylov.statistics)
statistics(solver::KrylovSolver) = Krylov.statistics(solver)
@doc (@doc DelaunayTriangulation.statistics)
statistics(tri::DelaunayTriangulation.Triangulation) = DelaunayTriangulation.statistics(tri)
export statistics
push!(overrides, :statistics)

## :stft
# Showing duplicate methods for stft in packages Module[DSP, Flux]
# Methods for stft in package DSP.Periodograms
# stft(s::AbstractVector{T}, n::Int64, noverlap::Int64, psdonly::Union{Nothing, DSP.Periodograms.PSDOnly}; onesided, nfft, fs, window) where T @ DSP.Periodograms ~/.julia/packages/DSP/-----/src/periodograms.jl:443
# stft(s::AbstractVector{T}, n::Int64, noverlap::Int64; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/-----/src/periodograms.jl:443
# stft(s::AbstractVector{T}, n::Int64; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/-----/src/periodograms.jl:443
# stft(s::AbstractVector{T}; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/-----/src/periodograms.jl:443
# Methods for stft in package NNlib
# stft(x; n_fft, hop_length, window, center, normalized) @ NNlibFFTWExt ~/.julia/packages/NNlib/CkJqS/ext/NNlibFFTWExt/stft.jl:1
stft = DSP.stft # Method for stft in package DSP
export stft
push!(overrides, :stft)

## :top
# Showing duplicate methods for top in packages Module[CairoMakie, DataStructures]
# Methods for top in package Makie
# top(rect::Rect2) @ Makie ~/.julia/packages/Makie/-----/src/makielayout/geometrybasics_extension.jl:5
# Methods for top in package DataStructures
# top(args...; kwargs...) @ DataStructures deprecated.jl:113
top = Makie.top 
export top
push!(overrides, :top)

## :transform
# Showing duplicate methods for transform in packages Module[DataFrames, MultivariateStats, RDatasets]
# Methods for transform in package DataFrames
# transform(arg::Union{Function, Type}, df::AbstractDataFrame; renamecols, threads) @ DataFrames ~/.julia/packages/DataFrames/-----/src/abstractdataframe/selection.jl:1388
# transform(df::AbstractDataFrame, args...; copycols, renamecols, threads) @ DataFrames ~/.julia/packages/DataFrames/-----/src/abstractdataframe/selection.jl:1383
# transform(f::Union{Function, Type}, gd::GroupedDataFrame; copycols, keepkeys, ungroup, renamecols, threads) @ DataFrames ~/.julia/packages/DataFrames/-----/src/groupeddataframe/splitapplycombine.jl:902
# transform(gd::GroupedDataFrame, args::Union{Regex, AbstractString, Function, Signed, Symbol, Unsigned, Pair, Type, All, Between, Cols, InvertedIndex, AbstractVecOrMat}...; copycols, keepkeys, ungroup, renamecols, threads) @ DataFrames ~/.julia/packages/DataFrames/-----/src/groupeddataframe/splitapplycombine.jl:912
# Methods for transform in package MultivariateStats
# transform(f, x) @ MultivariateStats deprecated.jl:103
# transform(f::MDS) @ MultivariateStats deprecated.jl:103
# transform(f::Whitening, x::AbstractVecOrMat{<:Real}) @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/whiten.jl:87
transform = DataFrames.transform  # the stuff in MultivariateStats seems deprecated
export transform
push!(overrides, :transform)

## :trim
# Showing duplicate methods for trim in packages Module[BenchmarkTools, StatsBase]
# Methods for trim in package BenchmarkTools
# trim(t::BenchmarkTools.Trial) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/-----/src/trials.jl:87
# trim(t::BenchmarkTools.Trial, percentage) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/-----/src/trials.jl:87
# Methods for trim in package StatsBase
# trim(x::AbstractVector; prop, count) @ StatsBase ~/.julia/packages/StatsBase/-----/src/robust.jl:52
@doc (@doc StatsBase.trim)
trim(x::AbstractVector; kwargs...) = StatsBase.trim(x; kwargs...)
@doc (@doc BenchmarkTools.trim)
trim(t::BenchmarkTools.Trial) = BenchmarkTools.trim(t)
trim(t::BenchmarkTools.Trial, percentage) = BenchmarkTools.trim(t, percentage)
export trim
push!(overrides, :trim)

## :trim!
# Showing duplicate methods for trim! in packages Module[CairoMakie, StatsBase]
# Methods for trim! in package GridLayoutBase
# trim!(gl::GridLayout) @ GridLayoutBase ~/.julia/packages/GridLayoutBase/-----/src/gridlayout.jl:574
# Methods for trim! in package StatsBase
# trim!(x::AbstractVector; prop, count) @ StatsBase ~/.julia/packages/StatsBase/-----/src/robust.jl:63
@doc (@doc StatsBase.trim!)
trim!(x::AbstractVector; kwargs...) = StatsBase.trim!(x; kwargs...)
@doc (@doc Makie.trim!)
trim!(gl::Makie.GridLayout) = Makie.trim!(gl)
export trim!
push!(overrides, :trim!)

## :unit
# Showing duplicate methods for unit in packages Module[GeometryBasics, Unitful]
# Methods for unit in package GeometryBasics
# unit(::Type{T}, i::Integer) where T<:(StaticArray{Tuple{N}, T, 1} where {N, T}) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/fixed_arrays.jl:2
# Methods for unit in package Unitful
# unit(::Type{<:Unitful.AbstractQuantity{T, D, U}}) where {T, D, U} @ Unitful ~/.julia/packages/Unitful/-----/src/utils.jl:119
# unit(::Type{Day}) @ Unitful ~/.julia/packages/Unitful/-----/src/dates.jl:8
# unit(::Type{Hour}) @ Unitful ~/.julia/packages/Unitful/-----/src/dates.jl:8
# unit(::Type{Microsecond}) @ Unitful ~/.julia/packages/Unitful/-----/src/dates.jl:8
# unit(::Type{Millisecond}) @ Unitful ~/.julia/packages/Unitful/-----/src/dates.jl:8
# unit(::Type{Minute}) @ Unitful ~/.julia/packages/Unitful/-----/src/dates.jl:8
# unit(::Type{Nanosecond}) @ Unitful ~/.julia/packages/Unitful/-----/src/dates.jl:8
# unit(::Type{Second}) @ Unitful ~/.julia/packages/Unitful/-----/src/dates.jl:8
# unit(::Type{Week}) @ Unitful ~/.julia/packages/Unitful/-----/src/dates.jl:8
# unit(a::Unitful.MixedUnits{L, U}) where {L, U} @ Unitful ~/.julia/packages/Unitful/-----/src/logarithm.jl:98
# unit(p::Union{Day, Hour, Microsecond, Millisecond, Minute, Nanosecond, Second, Week}) @ Unitful ~/.julia/packages/Unitful/-----/src/dates.jl:31
# unit(x::Base.TwicePrecision{Q}) where Q<:Quantity @ Unitful ~/.julia/packages/Unitful/-----/src/units.jl:288
# unit(x::Missing) @ Unitful ~/.julia/packages/Unitful/-----/src/utils.jl:144
# unit(x::Number) @ Unitful ~/.julia/packages/Unitful/-----/src/utils.jl:140
# unit(x::Type{Missing}) @ Unitful ~/.julia/packages/Unitful/-----/src/utils.jl:143
# unit(x::Type{T}) where T<:Number @ Unitful ~/.julia/packages/Unitful/-----/src/utils.jl:141
# unit(x::Type{Union{Missing, T}}) where T @ Unitful ~/.julia/packages/Unitful/-----/src/utils.jl:142
# unit(x::Unitful.AbstractQuantity{T, D, U}) where {T, D, U} @ Unitful ~/.julia/packages/Unitful/-----/src/utils.jl:118

# Unit in GeometryBasics is just used to create a column of the identity matrix. 
@doc Unitful.unit 
unit = Unitful.unit 
export unit
push!(overrides, :unit)

## :update!
# Showing duplicate methods for update! in packages Module[DataStructures, Flux, ProgressMeter, TaylorSeries]
# Methods for update! in package DataStructures
# update!(h::MutableBinaryHeap{T}, i::Int64, v) where T @ DataStructures ~/.julia/packages/DataStructures/-----/src/heaps/mutable_binary_heap.jl:255
# update!(pt::JumpProcesses.PriorityTable, pid, oldpriority, newpriority) @ JumpProcesses ~/.julia/packages/JumpProcesses/-----/src/aggregators/prioritytable.jl:186
# Methods for update! in package Optimisers
# update!(opt, model::Chain, grads::Tuple) @ Flux ~/.julia/packages/Flux/-----/src/deprecations.jl:94
# update!(opt::Flux.Optimise.AbstractOptimiser, ::Zygote.Params, grads::Union{Tuple, NamedTuple}) @ Flux ~/.julia/packages/Flux/-----/src/deprecations.jl:107
# update!(opt::Flux.Optimise.AbstractOptimiser, model, grad) @ Flux ~/.julia/packages/Flux/-----/src/deprecations.jl:81
# update!(opt::Flux.Optimise.AbstractOptimiser, model::Chain, grads::Tuple) @ Flux ~/.julia/packages/Flux/-----/src/deprecations.jl:101
# update!(opt::Flux.Optimise.AbstractOptimiser, x::AbstractArray, x̄) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/train.jl:22
# update!(opt::Flux.Optimise.AbstractOptimiser, xs::Zygote.Params, gs) @ Flux.Optimise ~/.julia/packages/Flux/-----/src/optimise/train.jl:28
# update!(tree, model, grad, higher...) @ Optimisers ~/.julia/packages/Optimisers/-----/src/interface.jl:70
# Methods for update! in package ProgressMeter
# update!(p::ProgressMeter.AbstractProgress, val, color; options...) @ ProgressMeter deprecated.jl:103
# update!(p::ProgressThresh, val; increment, options...) @ ProgressMeter ~/.julia/packages/ProgressMeter/-----/src/ProgressMeter.jl:499
# update!(p::ProgressThresh; ...) @ ProgressMeter ~/.julia/packages/ProgressMeter/-----/src/ProgressMeter.jl:499
# update!(p::Union{Progress, ProgressUnknown}, counter::Int64; options...) @ ProgressMeter ~/.julia/packages/ProgressMeter/-----/src/ProgressMeter.jl:486
# update!(p::Union{Progress, ProgressUnknown}; ...) @ ProgressMeter ~/.julia/packages/ProgressMeter/-----/src/ProgressMeter.jl:486
# Methods for update! in package TaylorSeries
# update!(a::Taylor1{T}, x0::S) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/other_functions.jl:261
# update!(a::Taylor1{T}, x0::T) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/other_functions.jl:257
# update!(a::TaylorN{T}, vals::Vector{S}) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/other_functions.jl:271
# update!(a::TaylorN{T}, vals::Vector{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/other_functions.jl:267
# update!(a::Union{Taylor1, TaylorN}) @ TaylorSeries ~/.julia/packages/TaylorSeries/-----/src/other_functions.jl:276
@doc (@doc DataStructures.update!) 
update!(h::DataStructures.MutableBinaryHeap, i, v) = DataStructures.update!(h, i, v)
update!(pt::JumpProcesses.PriorityTable, pid, oldpriority, newpriority) = JumpProcesses.update!(pt, pid, oldpriority, newpriority)
@doc (@doc ProgressMeter.update!) 
update!(p::ProgressMeter.AbstractProgress, val, color; options...) = ProgressMeter.update!(p, val, color; options...)
update!(p::Union{ProgressMeter.Progress,ProgressMeter.ProgressUnknown,ProgressMeter.ProgressThresh}, val; options...) = ProgressMeter.update!(p, val; options...)
update!(p::Union{ProgressMeter.Progress,ProgressMeter.ProgressUnknown,ProgressMeter.ProgressThresh}; options...) = ProgressMeter.update!(p; options...)
@doc (@doc Flux.update!)
update!(opt::Flux.Optimise.AbstractOptimiser, args...) = Flux.update!(opt, args...)
@doc (@doc TaylorSeries.update!)
update!(a::TaylorSeries.Taylor1, x0) = TaylorSeries.update!(a, x0)
update!(a::TaylorSeries.TaylorN, vals::Vector) = TaylorSeries.update!(a, vals)
export update!
push!(overrides, :update!)

## :value
# Showing duplicate methods for value in packages Module[JuMP, OnlineStats]
# Methods for value in package JuMP
# value(::AbstractArray{<:AbstractJuMPScalar}) @ JuMP ~/.julia/packages/JuMP/-----/src/variables.jl:2453
# value(::Function, x::Number) @ JuMP ~/.julia/packages/JuMP/-----/src/variables.jl:2465
# value(::MutableArithmetics.Zero; result) @ JuMP ~/.julia/packages/JuMP/-----/src/variables.jl:2461
# value(Q::Symmetric{V, Matrix{V}}) where V<:AbstractVariableRef @ JuMP ~/.julia/packages/JuMP/-----/src/sd.jl:430
# value(a::GenericAffExpr; result) @ JuMP ~/.julia/packages/JuMP/-----/src/aff_expr.jl:700
# value(a::GenericNonlinearExpr; result) @ JuMP ~/.julia/packages/JuMP/-----/src/nlp_expr.jl:654
# value(c::NonlinearConstraintRef; result) @ JuMP ~/.julia/packages/JuMP/-----/src/nlp.jl:616
# value(con_ref::ConstraintRef{<:AbstractModel, <:MathOptInterface.ConstraintIndex}; result) @ JuMP ~/.julia/packages/JuMP/-----/src/constraints.jl:1297
# value(ex::GenericQuadExpr; result) @ JuMP ~/.julia/packages/JuMP/-----/src/quad_expr.jl:854
# value(ex::NonlinearExpression; result) @ JuMP ~/.julia/packages/JuMP/-----/src/nlp.jl:413
# value(f::Function, expr::GenericNonlinearExpr) @ JuMP ~/.julia/packages/JuMP/-----/src/nlp_expr.jl:650
# value(p::NonlinearParameter) @ JuMP ~/.julia/packages/JuMP/-----/src/nlp.jl:285
# value(v::GenericVariableRef{T}; result) where T @ JuMP ~/.julia/packages/JuMP/-----/src/variables.jl:1905
# value(var_value::Function, c::NonlinearConstraintRef) @ JuMP ~/.julia/packages/JuMP/-----/src/nlp.jl:627
# value(var_value::Function, con_ref::ConstraintRef{<:AbstractModel, <:MathOptInterface.ConstraintIndex}) @ JuMP ~/.julia/packages/JuMP/-----/src/constraints.jl:1310
# value(var_value::Function, ex::GenericAffExpr{T, V}) where {T, V} @ JuMP ~/.julia/packages/JuMP/-----/src/aff_expr.jl:404
# value(var_value::Function, ex::GenericQuadExpr{CoefType, VarType}) where {CoefType, VarType} @ JuMP ~/.julia/packages/JuMP/-----/src/quad_expr.jl:828
# value(var_value::Function, ex::NonlinearExpression) @ JuMP ~/.julia/packages/JuMP/-----/src/nlp.jl:448
# value(var_value::Function, v::GenericVariableRef) @ JuMP ~/.julia/packages/JuMP/-----/src/variables.jl:1914
# value(x::Number; result) @ JuMP ~/.julia/packages/JuMP/-----/src/variables.jl:2463
# Methods for value in package OnlineStatsBase
# value(::CountMinSketch) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/probabilistic.jl:38
# value(::CountMinSketch{T}, val::S) where {T, S} @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/probabilistic.jl:39
# value(o::Ash) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/ash.jl:43
# value(o::Ash, kernel::Function) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/ash.jl:41
# value(o::Ash, kernel::Function, m::Int64) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/ash.jl:41
# value(o::Ash, m::Int64) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/ash.jl:43
# value(o::Ash, m::Int64, kernel::Function) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/ash.jl:43
# value(o::AutoCov) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:85
# value(o::CallFun) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:158
# value(o::CountMinSketch{T, I}, val::T) where {T, I} @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/probabilistic.jl:43
# value(o::CovMatrix; corrected) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:192
# value(o::DPMM{T}) where T<:Real @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/dpmm.jl:275
# value(o::ExpandingHist) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/histograms.jl:180
# value(o::Extrema) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:286
# value(o::FitBeta) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/distributions.jl:16
# value(o::FitCauchy) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/distributions.jl:46
# value(o::FitGamma) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/distributions.jl:72
# value(o::FitLogNormal) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/distributions.jl:99
# value(o::FitMultinomial) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/distributions.jl:176
# value(o::FitMvNormal) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/distributions.jl:203
# value(o::FitNormal) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/distributions.jl:124
# value(o::GeometricMean) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:212
# value(o::HeatMap) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/heatmap.jl:45
# value(o::HyperLogLog; original_estimator) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/probabilistic.jl:124
# value(o::IndexedPartition) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/partition.jl:93
# value(o::KHist) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/khist.jl:68
# value(o::KIndexedPartition) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/partition.jl:156
# value(o::KahanVariance{W, T}) where {W, T} @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/kahan.jl:210
# value(o::LinReg, args...) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/linreg.jl:41
# value(o::LinRegBuilder, args...) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/linreg.jl:103
# value(o::LogSumExp) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:723
# value(o::Mosaic) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/mosaicplot.jl:18
# value(o::MovingTimeWindow) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:347
# value(o::MovingWindow) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:383
# value(o::OnlineStats.Cluster) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:254
# value(o::OnlineStats.Hist) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/histograms.jl:69
# value(o::OnlineStats.Partition) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/viz/partition.jl:26
# value(o::OnlineStats.Trace) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/trace.jl:28
# value(o::OnlineStatsBase.CircBuff; ordered) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:60
# value(o::OnlineStatsBase.ExtremeValues) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:313
# value(o::OnlineStatsBase.Series) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:619
# value(o::OnlineStatsBase.StatWrapper) @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/wrappers.jl:4
# value(o::P2Quantile) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:544
# value(o::Quantile) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:637
# value(o::Quantile, q) @ OnlineStats ~/.julia/packages/OnlineStats/-----/src/stats/stats.jl:637
# value(o::T) where T<:OnlineStat @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/OnlineStatsBase.jl:39
# value(o::Variance{T}) where T @ OnlineStatsBase ~/.julia/packages/OnlineStatsBase/-----/src/stats.jl:588

# I hope this one works.. 
@doc (@doc JuMP.value)
# need to specialize this one more... 
value(args...) = JuMP.value(args...)
value(arg; result) = JuMP.value(arg; result)
@doc (@doc OnlineStats.value)
value(o::OnlineStat, args...) = OnlineStats.value(o, args...)
export value
push!(overrides, :value)

## :volume
# Showing duplicate methods for volume in packages Module[CairoMakie, GeometryBasics]
# Methods for volume in package MakieCore
# volume() @ MakieCore ~/.julia/packages/MakieCore/-----/src/recipes.jl:432
# volume(args...; kw...) @ MakieCore ~/.julia/packages/MakieCore/-----/src/recipes.jl:447
# Methods for volume in package GeometryBasics
# volume(mesh::GeometryBasics.Mesh) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/meshes.jl:233
# volume(prim::HyperRectangle) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/rectangles.jl:189
# volume(triangle::Triangle) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/meshes.jl:221
@doc (@doc GeometryBasics.volume)
volume(mesh::GeometryBasics.Mesh) = GeometryBasics.volume(mesh)
volume(triangle::GeometryBasics.Triangle) = GeometryBasics.volume(triangle)
volume(prim::GeometryBasics.HyperRectangle) = GeometryBasics.volume(prim)
@doc (@doc Makie.volume)
volume() = Makie.volume()
volume(args...; kw...) = Makie.volume(args...; kw...)

export volume
push!(overrides, :volume)

## :weights
# Showing duplicate methods for weights in packages Module[Graphs, LsqFit, StatsBase]
# Methods for weights in package Graphs
# weights(g::AbstractGraph) @ Graphs ~/.julia/packages/Graphs/-----/src/core.jl:431
# weights(g::MetaGraphs.AbstractMetaGraph) @ MetaGraphs ~/.julia/packages/MetaGraphs/-----/src/MetaGraphs.jl:236
# weights(g::SimpleWeightedDiGraph) @ SimpleWeightedGraphs ~/.julia/packages/SimpleWeightedGraphs/-----/src/simpleweighteddigraph.jl:138
# weights(g::SimpleWeightedGraph) @ SimpleWeightedGraphs ~/.julia/packages/SimpleWeightedGraphs/-----/src/simpleweightedgraph.jl:166
# Methods for weights in package StatsAPI
# weights(f::LinearDiscriminant) @ MultivariateStats ~/.julia/packages/MultivariateStats/-----/src/lda.jl:121
# weights(lfr::LsqFit.LsqFitResult) @ LsqFit ~/.julia/packages/LsqFit/-----/src/curve_fit.jl:14
# weights(vs::AbstractArray{<:Real}) @ StatsBase ~/.julia/packages/StatsBase/-----/src/weights.jl:88
# weights(vs::AbstractVector{<:Real}) @ StatsBase ~/.julia/packages/StatsBase/-----/src/weights.jl:89
@doc (@doc StatsBase.weights)
weights(vs::AbstractVector) = StatsBase.weights(vs)
weights(vs::AbstractArray) = StatsBase.weights(vs)
weights(f::MultivariateStats.LinearDiscriminant) = MultivariateStats.weights(f)
@doc (@doc Graphs.weights)
weights(g::Graphs.AbstractGraph) = Graphs.weights(g)
export weights
push!(overrides, :weights)

## :width
# Showing duplicate methods for width in packages Module[CairoMakie, GeometryBasics, Measures]
# Methods for width in package GeometryBasics
# width(prim::HyperRectangle) @ GeometryBasics ~/.julia/packages/GeometryBasics/-----/src/primitives/rectangles.jl:186
# Methods for width in package Measures
# width(x::BoundingBox) @ Measures ~/.julia/packages/Measures/-----/src/boundingbox.jl:43
@doc (@doc GeometryBasics.width)
width(prim::HyperRectangle) = GeometryBasics.width(prim)
@doc (@doc Measures.width)
width(x::BoundingBox) = Measures.width(x)
export width 
push!(overrides, :width)

## :write_to_file
# Showing duplicate methods for write_to_file in packages Module[Convex, JuMP]
# Methods for write_to_file in package Convex
# write_to_file(p::Problem{T}, filename::String) where T<:Float64 @ Convex ~/.julia/packages/Convex/-----/src/problems.jl:310
# Methods for write_to_file in package JuMP
# write_to_file(model::GenericModel, filename::String; format, kwargs...) @ JuMP ~/.julia/packages/JuMP/-----/src/file_formats.jl:46

@doc (@doc JuMP.write_to_file)
write_to_file(model::JuMP.GenericModel, filename::String; format, kwargs...) = JuMP.write_to_file(model, filename; format, kwargs...)
@doc (@doc Convex.write_to_file)
write_to_file(p::Convex.Problem, filename::String) = Convex.write_to_file(p, filename)
export write_to_file
push!(overrides, :write_to_file)

## :⊕
# Showing duplicate methods for ⊕ in packages Module[DoubleFloats, LinearMaps]
# Methods for ⊕ in package DoubleFloats
# ⊕(x::T, y::T) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/-----/src/math/ops/arith.jl:47
# Methods for ⊕ in package LinearMaps
# ⊕(a, b, c...) @ LinearMaps ~/.julia/packages/LinearMaps/-----/src/kronecker.jl:420
# ⊕(k::Integer) @ LinearMaps ~/.julia/packages/LinearMaps/-----/src/kronecker.jl:418

@doc (@doc getfield(LinearMaps, :⊕))
⊕(k::Integer) = LinearMaps.⊕(k)
⊕(A,B,Cs...) = LinearMaps.⊕(A,B,Cs...)
@doc (@doc getfield(DoubleFloats, :⊕))
⊕(x::T, y::T) where T<:Union{Float16, Float32, Float64} = DoubleFloats.⊕(x, y)
export ⊕
push!(overrides, :⊕)


## :⊗
# Showing duplicate methods for ⊗ in packages Module[ColorVectorSpace, DoubleFloats, Images, LinearMaps]
# Methods for tensor in package TensorCore
# tensor(A::AbstractArray, B::AbstractArray) @ TensorCore ~/.julia/packages/TensorCore/-----/src/TensorCore.jl:83
# tensor(a::AbstractRGB, b::AbstractRGB) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/-----/src/ColorVectorSpace.jl:421
# tensor(a::C, b::C) where C<:(Union{TransparentColor{C, T}, C} where {T, C<:Union{AbstractRGB{T}, AbstractGray{T}}}) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/-----/src/ColorVectorSpace.jl:257
# tensor(a::Union{TransparentColor{C, T}, C} where {T, C<:Union{AbstractRGB{T}, AbstractGray{T}}}, b::Union{TransparentColor{C, T}, C} where {T, C<:Union{AbstractRGB{T}, AbstractGray{T}}}) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/-----/src/ColorVectorSpace.jl:264
# tensor(u::AbstractArray, v::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}} where T) @ TensorCore ~/.julia/packages/TensorCore/-----/src/TensorCore.jl:87
# tensor(u::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}} where T, v::AbstractArray) @ TensorCore ~/.julia/packages/TensorCore/-----/src/TensorCore.jl:93
# tensor(u::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}} where T, v::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}} where T) @ TensorCore ~/.julia/packages/TensorCore/-----/src/TensorCore.jl:96
# tensor(x::C, y::C) where C<:(AbstractGray) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/-----/src/ColorVectorSpace.jl:332
# Methods for ⊗ in package DoubleFloats
# ⊗(x::T, y::T) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/-----/src/math/ops/arith.jl:82
# Methods for ⊗ in package LinearMaps
# ⊗(A, B, Cs...) @ LinearMaps ~/.julia/packages/LinearMaps/-----/src/kronecker.jl:136
# ⊗(k::Integer) @ LinearMaps ~/.julia/packages/LinearMaps/-----/src/kronecker.jl:134

# This ignores images, which also exports it's function under tensor. 
@doc (@doc getfield(LinearMaps, :⊗))
⊗(k::Integer) = LinearMaps.⊗(k)
⊗(A,B,Cs...) = LinearMaps.⊗(A,B,Cs...)
@doc (@doc getfield(DoubleFloats, :⊗))
⊗(a::AbstractRGB, b::AbstractRGB) = ColorVectorSpace.tensor(a,b)
⊗(a::C, b::C) where C<:(Union{TransparentColor{C, T}, C} where {T, C<:Union{AbstractRGB{T}, AbstractGray{T}}}) = ColorVectorSpace.tensor(a,b)
@doc (@doc getfield(DoubleFloats, :⊗))
⊗(x::T, y::T) where T<:Union{Float16, Float32, Float64} = DoubleFloats.⊗(x, y)
export ⊗
push!(overrides, :⊗)

##-Unused overrides
#=
## :order
order = DataFrames.order 

=#
