const overrides = Set{Symbol}()

## :Axis
# Showing duplicate methods for Axis in packages Module[Images, AxisArrays, CairoMakie]
# Methods for AxisArrays.Axis in package Core
# Methods for Makie.Axis in package Core
# Makie.Axis(parent::Union{Nothing, Figure, Scene}, layoutobservables::LayoutObservables{GridLayout}, blockscene::Scene) @ Makie ~/.julia/packages/Makie/YkotL/src/makielayout/blocks.jl:50
# (::Type{T})(args...; kwargs...) where T<:Block @ Makie ~/.julia/packages/Makie/YkotL/src/makielayout/blocks.jl:236
@doc (@doc Makie.Axis) 
Axis = Makie.Axis
@doc (@doc Images.Axis)
ArrayAxis = Images.Axis
export PlotAxis, ArrayAxis 
push!(overrides, :Axis)

## :Bisection
# Showing duplicate methods for Bisection in packages Module[Roots, DifferentialEquations]
# Methods for Roots.Bisection in package Core
# Roots.Bisection() @ Roots ~/.julia/packages/Roots/KNVCY/src/Bracketing/bisection.jl:28
# Methods for SimpleNonlinearSolve.Bisection in package Core
# SimpleNonlinearSolve.Bisection(exact_left::Bool, exact_right::Bool) @ SimpleNonlinearSolve ~/.julia/packages/SimpleNonlinearSolve/YQl3A/src/bracketing/bisection.jl:18
# SimpleNonlinearSolve.Bisection(exact_left, exact_right) @ SimpleNonlinearSolve ~/.julia/packages/SimpleNonlinearSolve/YQl3A/src/bracketing/bisection.jl:18
# SimpleNonlinearSolve.Bisection(; exact_left, exact_right) @ SimpleNonlinearSolve ~/.julia/packages/SimpleNonlinearSolve/YQl3A/src/bracketing/bisection.jl:17
@doc (@doc Roots.Bisection)
Bisection() = Roots.Bisection()
@doc (@doc SimpleNonlinearSolve.Bisection)
Bisection(left,right) = SimpleNonlinearSolve.Bisection(left,right)
#Bisection(;exact_left,exact_right) = SimpleNonlinearSolve.Bisection(;exact_left,exact_right)
export Bisection 
push!(overrides, :Bisection)

## :Categorical
# Showing duplicate methods for Categorical in packages Module[Distributions, CairoMakie]
# Methods for Distributions.Categorical{P} where P<:Real in package Core
# (Distributions.Categorical{P} where P<:Real)(k::Integer; check_args) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/categorical.jl:37
# (Distributions.Categorical{P} where P<:Real)(probabilities::Real...; check_args) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/categorical.jl:42
# (Distributions.Categorical{P} where P<:Real)(p::AbstractVector{P}; check_args) where P<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/categorical.jl:34
# Methods for Makie.Categorical in package Core
# Makie.Categorical(values) @ Makie ~/.julia/packages/Makie/YkotL/src/colorsampler.jl:229
@doc (@doc Distributions.Categorical)
Categorical = Distributions.Categorical
@doc (@doc Makie.Categorical)
CategoricalColormap = Makie.Categorical
export Categorical, CategoricalColormap
push!(overrides, :Categorical)

## :EllipticalArc
# Showing duplicate methods for EllipticalArc in packages Module[CairoMakie, DelaunayTriangulation]
# Methods for Makie.EllipticalArc in package Core
# Makie.EllipticalArc(c::Point{2, Float64}, r1::Float64, r2::Float64, angle::Float64, a1::Float64, a2::Float64) @ Makie ~/.julia/packages/Makie/YkotL/src/bezier.jl:75
# Makie.EllipticalArc(c, r1, r2, angle, a1, a2) @ Makie ~/.julia/packages/Makie/YkotL/src/bezier.jl:75
# Makie.EllipticalArc(cx, cy, r1, r2, angle, a1, a2) @ Makie ~/.julia/packages/Makie/YkotL/src/bezier.jl:83
# Makie.EllipticalArc(x1, y1, x2, y2, rx, ry, ϕ, largearc::Bool, sweepflag::Bool) @ Makie ~/.julia/packages/Makie/YkotL/src/bezier.jl:546
# Methods for DelaunayTriangulation.EllipticalArc in package Core
# DelaunayTriangulation.EllipticalArc(center::Tuple{Float64, Float64}, horz_radius::Float64, vert_radius::Float64, rotation_scales::Tuple{Float64, Float64}, start_angle::Float64, sector_angle::Float64, first::Tuple{Float64, Float64}, last::Tuple{Float64, Float64}) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/mesh_refinement/curves/ellipticalarc.jl:26
# DelaunayTriangulation.EllipticalArc(center, horz_radius, vert_radius, rotation_scales, start_angle, sector_angle, first, last) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/mesh_refinement/curves/ellipticalarc.jl:26
# DelaunayTriangulation.EllipticalArc(p, q, c, α, β, θ°; positive) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/mesh_refinement/curves/ellipticalarc.jl:47
@doc (@doc Makie.EllipticalArc)
EllipticalArc(c::Point{2, Float64}, r1::Float64, r2::Float64, angle::Float64, a1::Float64, a2::Float64)= Makie.EllipticalArc(c, r1, r2, angle, a1, a2)
EllipticalArc(x1, y1, x2, y2, rx, ry, ϕ, largearc::Bool, sweepflag::Bool) = Makie.EllipticalArc(x1, y1, x2, y2, rx, ry, ϕ, largearc, sweepflag)
@doc (@doc DelaunayTriangulation.EllipticalArc)
EllipticalArc(p, q, c, α, β, θ°; positive) = DelaunayTriangulation.EllipticalArc(p, q, c, α, β, θ°; positive)
export EllipticalArc
push!(overrides, :EllipticalArc)

## :Fill
# Showing duplicate methods for Fill in packages Module[Images, FillArrays]
# Methods for ImageFiltering.Fill in package Core
# ImageFiltering.Fill(value::T, ::Tuple{}) where T @ ImageFiltering ~/.julia/packages/ImageFiltering/QW8Jn/src/border.jl:528
# ImageFiltering.Fill(value, lo::AbstractVector, hi::AbstractVector) @ ImageFiltering ~/.julia/packages/ImageFiltering/QW8Jn/src/border.jl:527
# ImageFiltering.Fill(value::T, lo::Tuple{Vararg{Int64, N}}, hi::Tuple{Vararg{Int64, N}}) where {T, N} @ ImageFiltering ~/.julia/packages/ImageFiltering/QW8Jn/src/border.jl:509
# ImageFiltering.Fill(value::T, both::Tuple{Vararg{Int64, N}}) where {T, N} @ ImageFiltering ~/.julia/packages/ImageFiltering/QW8Jn/src/border.jl:525
# ImageFiltering.Fill(value::T, inds::Tuple{Vararg{AbstractUnitRange, N}}) where {T, N} @ ImageFiltering ~/.julia/packages/ImageFiltering/QW8Jn/src/border.jl:529
# ImageFiltering.Fill(value::T) where T @ ImageFiltering ~/.julia/packages/ImageFiltering/QW8Jn/src/border.jl:485
# ImageFiltering.Fill(value, kernel) @ ImageFiltering ~/.julia/packages/ImageFiltering/QW8Jn/src/border.jl:547
# Methods for FillArrays.Fill in package Core
# FillArrays.Fill(x::T, sz::Tuple{Vararg{Any, N}}) where {T, N} @ FillArrays ~/.julia/packages/FillArrays/lVl4c/src/FillArrays.jl:138
# FillArrays.Fill(x::T, sz::Vararg{Integer, N}) where {T, N} @ FillArrays ~/.julia/packages/FillArrays/lVl4c/src/FillArrays.jl:136
@doc (@doc ImageFiltering.Fill)
FillValue = ImageFiltering.Fill
@doc (@doc FillArrays.Fill)
FillArray = FillArrays.Fill 
export FillValue, FillArray
push!(overrides, :Fill)

## :Filters
# Showing duplicate methods for Filters in packages Module[HDF5, DSP]
# Methods for HDF5.Filters in package Core
# Methods for DSP.Filters in package Core
@doc (@doc DSP.Filters)
Filters = DSP.Filters
export Filters 
push!(overrides, :Filters)

## :Fixed
# Showing duplicate methods for Fixed in packages Module[Images, CairoMakie]
# Methods for FixedPointNumbers.Fixed in package Core
# (::Type{X})(x::X) where X<:FixedPoint @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/Dn4hv/src/FixedPointNumbers.jl:57
# (::Type{X})(x::Complex) where X<:FixedPoint @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/Dn4hv/src/FixedPointNumbers.jl:63
# (::Type{X})(x::Number) where X<:FixedPoint @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/Dn4hv/src/FixedPointNumbers.jl:58
# (::Type{<:FixedPoint})(x::AbstractChar) @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/Dn4hv/src/FixedPointNumbers.jl:60
# (::Type{X})(x::Base.TwicePrecision) where X<:FixedPoint @ FixedPointNumbers ~/.julia/packages/FixedPointNumbers/Dn4hv/src/FixedPointNumbers.jl:64
# (::Type{T})(x::AbstractGray) where T<:Real @ ColorTypes ~/.julia/packages/ColorTypes/vpFgh/src/conversions.jl:115
# (::Type{T})(x::Base.TwicePrecision) where T<:Number @ Base twiceprecision.jl:265
# (::Type{T})(x::T) where T<:Number @ Core boot.jl:792
# (::Type{T})(x::AbstractChar) where T<:Union{AbstractChar, Number} @ Base char.jl:50
# Methods for GridLayoutBase.Fixed in package Core
# GridLayoutBase.Fixed(x::Float32) @ GridLayoutBase ~/.julia/packages/GridLayoutBase/TSvez/src/types.jl:160
# GridLayoutBase.Fixed(x) @ GridLayoutBase ~/.julia/packages/GridLayoutBase/TSvez/src/types.jl:160
@doc (@doc FixedPointNumbers.Fixed)
Fixed = FixedPointNumbers.Fixed
@doc (@doc Makie.Fixed)
FixedSize = Makie.Fixed
export Fixed, FixedSize 
push!(overrides, :Fixed)

## :FunctionMap
# Showing duplicate methods for FunctionMap in packages Module[LinearMaps, DifferentialEquations]
# Methods for LinearMaps.FunctionMap in package Core
# Methods for OrdinaryDiffEqFunctionMap.FunctionMap in package Core
# OrdinaryDiffEqFunctionMap.FunctionMap(; scale_by_time) @ OrdinaryDiffEqFunctionMap ~/.julia/packages/OrdinaryDiffEqFunctionMap/15mZr/src/algorithms.jl:2
@doc (@doc LinearMaps.FunctionMap)
FunctionMap = LinearMaps.FunctionMap
export FunctionMap
push!(overrides, :FunctionMap)

## :Length
# Showing duplicate methods for Length in packages Module[Measures, StaticArrays]
# Methods for Measures.Length in package Core
# Measures.Length(unit::Symbol, x::T) where T @ Measures ~/.julia/packages/Measures/PKOxJ/src/length.jl:7
# Methods for StaticArrays.Length in package Core
# StaticArrays.Length(L::Int64) @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/traits.jl:39
# StaticArrays.Length(a::AbstractArray) @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/traits.jl:37
# StaticArrays.Length(::Type{A}) where A<:AbstractArray @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/traits.jl:38
# StaticArrays.Length(::Size{S}) where S @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/traits.jl:40
# StaticArrays.Length(x::StaticArrays.Args) @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/convert.jl:9

# Neither of these seems to be the right one...

## :Mesh
# Showing duplicate methods for Mesh in packages Module[GeometryBasics, CairoMakie]
# Methods for GeometryBasics.Mesh in package Core
# GeometryBasics.Mesh(simplices::V) where {Dim, T<:Number, Element<:Polytope{Dim, T}, V<:AbstractVector{Element}} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/basic_types.jl:375
# GeometryBasics.Mesh(elements::AbstractVector{<:Polytope{Dim, T}}) where {Dim, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/basic_types.jl:408
# GeometryBasics.Mesh(points::AbstractVector{<:AbstractPoint}, faces::AbstractVector{<:AbstractFace}) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/basic_types.jl:412
# GeometryBasics.Mesh(points::AbstractVector{<:AbstractPoint}, faces::AbstractVector{<:Integer}) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/basic_types.jl:417
# GeometryBasics.Mesh(points::AbstractVector{<:AbstractPoint}, faces::AbstractVector{<:Integer}, facetype) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/basic_types.jl:417
# GeometryBasics.Mesh(points::AbstractVector{<:AbstractPoint}, faces::AbstractVector{<:Integer}, facetype, skip) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/basic_types.jl:417
# Methods for MakieCore.Mesh in package Core
# (Plot{Func})(user_args::Tuple, user_attributes::Dict) where Func @ Makie ~/.julia/packages/Makie/YkotL/src/interfaces.jl:260
Mesh = GeometryBasics.Mesh 
export Mesh
push!(overrides, :Mesh)

## :Normal
# Showing duplicate methods for Normal in packages Module[GeometryBasics, Distributions]
# Methods for GeometryBasics.Normal in package Core
# GeometryBasics.Normal() @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/interfaces.jl:100
# GeometryBasics.Normal(::Type{T}) where T @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/interfaces.jl:99
# Methods for Distributions.Normal in package Core
# Distributions.Normal() @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normal.jl:44
# Distributions.Normal(μ::Integer, σ::Integer; check_args) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normal.jl:43
# Distributions.Normal(μ::T, σ::T; check_args) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normal.jl:36
# Distributions.Normal(μ::Real, σ::Real; check_args) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normal.jl:42
# Distributions.Normal(μ::Real) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normal.jl:44
@doc (@doc Distributions.Normal)
Normal = Distributions.Normal
push!(overrides, :Normal)
NormalVector = GeometryBasics.Normal
export Normal, NormalVector 

## :Partition
# Showing duplicate methods for Partition in packages Module[Transducers, Combinatorics]
# Methods for Transducers.Partition in package Core
# Transducers.Partition(size, step, flush) @ Transducers ~/.julia/packages/Transducers/txnl6/src/library.jl:826
# Transducers.Partition(size, step; flush) @ Transducers ~/.julia/packages/Transducers/txnl6/src/library.jl:833
# Transducers.Partition(size; step, flush) @ Transducers ~/.julia/packages/Transducers/txnl6/src/library.jl:834
# Methods for Combinatorics.Partition in package Core
# Combinatorics.Partition(x::Vector{Int64}) @ Combinatorics ~/.julia/packages/Combinatorics/Udg6X/src/youngdiagrams.jl:6
# Combinatorics.Partition(x) @ Combinatorics ~/.julia/packages/Combinatorics/Udg6X/src/youngdiagrams.jl:6
Partition = Transducers.Partition
export Partition
push!(overrides, :Partition)

## :Vec
# Showing duplicate methods for Vec in packages Module[GeometryBasics, Measures, CairoMakie]
# Methods for GeometryBasics.Vec in package Core
# (::Type{SV})(x::StaticArray{Tuple{N}, T, 1} where {N, T}) where SV<:Vec @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/fixed_arrays.jl:76
# GeometryBasics.Vec(x::Tuple{Vararg{T, S}} where T) where S @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/fixed_arrays.jl:52
# GeometryBasics.Vec(x::T) where {S, T<:Tuple{Vararg{Any, S}}} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/fixed_arrays.jl:53
# (::Type{SA})(gen::Base.Generator) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/SArray.jl:57
# (::Type{SA})(sa::StaticArray) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/convert.jl:178
# (T::Type{<:StaticArray})(a::AbstractArray) @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/convert.jl:182
# (::Type{SA})(x...) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/convert.jl:173
# Methods for Tuple{Vararg{Measure, N}} where N in package Core
# (::Type{T})(x::Tuple) where T<:Tuple @ Base tuple.jl:386
# (::Type{T})(nt::NamedTuple) where T<:Tuple @ Base namedtuple.jl:201
# (::Type{T})(itr) where T<:Tuple @ Base tuple.jl:391

# We are just ignoring the Measures vectors... 
@doc (@doc GeometryBasics.Vec)
Vec = GeometryBasics.Vec
push!(overrides, :Vec)
export Vec

## :Vec2
# Showing duplicate methods for Vec2 in packages Module[GeometryBasics, Measures, CairoMakie]
# Methods for GeometryBasics.Vec2 in package Core
# (GeometryBasics.Vec{S})(x::T) where {S, T<:Tuple} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/fixed_arrays.jl:57
# (GeometryBasics.Vec{S})(x::AbstractVector{T}) where {S, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/fixed_arrays.jl:36
# (GeometryBasics.Vec{S})(x::T) where {S, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/fixed_arrays.jl:50
# (::Type{SA})(x...) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/convert.jl:173
# Methods for Tuple{Measure, Measure} in package Core
# (::Type{T})(x::Tuple) where T<:Tuple @ Base tuple.jl:386
# (::Type{T})(nt::NamedTuple) where T<:Tuple @ Base namedtuple.jl:201
# Tuple{Vararg{T, N}}(v::SIMD.Vec{N}) where {T, N} @ SIMD ~/.julia/packages/SIMD/0q83J/src/simdvec.jl:68
# (::Type{T})(itr) where T<:Tuple @ Base tuple.jl:391

@doc (@doc GeometryBasics.Vec2)
Vec2 = GeometryBasics.Vec2
export Vec2
push!(overrides, :Vec2)

## :Vec3
# Showing duplicate methods for Vec3 in packages Module[GeometryBasics, Measures, CairoMakie]
# Methods for GeometryBasics.Vec3 in package Core
# (GeometryBasics.Vec{S})(x::T) where {S, T<:Tuple} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/fixed_arrays.jl:57
# (GeometryBasics.Vec{S})(x::AbstractVector{T}) where {S, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/fixed_arrays.jl:36
# (GeometryBasics.Vec{S})(x::T) where {S, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/fixed_arrays.jl:50
# (::Type{SA})(x...) where SA<:StaticArray @ StaticArrays ~/.julia/packages/StaticArrays/MSJcA/src/convert.jl:173
# Methods for Tuple{Measure, Measure, Measure} in package Core
# (::Type{T})(x::Tuple) where T<:Tuple @ Base tuple.jl:386
# (::Type{T})(nt::NamedTuple) where T<:Tuple @ Base namedtuple.jl:201
# Tuple{Vararg{T, N}}(v::SIMD.Vec{N}) where {T, N} @ SIMD ~/.julia/packages/SIMD/0q83J/src/simdvec.jl:68
# (::Type{T})(itr) where T<:Tuple @ Base tuple.jl:391

@doc (@doc GeometryBasics.Vec3)
Vec3 = GeometryBasics.Vec3
export Vec3
push!(overrides, :Vec3)

## :Zeros
# Showing duplicate methods for Zeros in packages Module[FillArrays, JuMP]
# Methods for FillArrays.Zeros in package Core
# FillArrays.Zeros(::Type{T}, m...) where T @ FillArrays ~/.julia/packages/FillArrays/lVl4c/src/FillArrays.jl:317
# FillArrays.Zeros(n::Integer) @ FillArrays ~/.julia/packages/FillArrays/lVl4c/src/FillArrays.jl:311
# FillArrays.Zeros(A::AbstractArray) @ FillArrays ~/.julia/packages/FillArrays/lVl4c/src/FillArrays.jl:316
# FillArrays.Zeros(sz::SZ) where {N, SZ<:Tuple{Vararg{Any, N}}} @ FillArrays ~/.julia/packages/FillArrays/lVl4c/src/FillArrays.jl:309
# FillArrays.Zeros(sz::Vararg{Any, N}) where N @ FillArrays ~/.julia/packages/FillArrays/lVl4c/src/FillArrays.jl:308
# Methods for JuMP.Zeros in package Core
# JuMP.Zeros() @ JuMP ~/.julia/packages/JuMP/6RAQ9/src/macros/@constraint.jl:704
@doc (@doc FillArrays.Zeros)
Zeros = FillArrays.Zeros
export Zeros


## :attributes
# Showing duplicate methods for attributes in packages Module[HDF5, CairoMakie]
# Methods for attributes in package HDF5
# attributes(p::Union{HDF5.Dataset, HDF5.Datatype, HDF5.File, HDF5.Group}) @ HDF5 ~/.julia/packages/HDF5/Z859u/src/attributes.jl:374
# Methods for attributes in package MakieCore
# attributes(x::Attributes) @ MakieCore ~/.julia/packages/MakieCore/NeQjl/src/attributes.jl:34
# attributes(x::AbstractPlot) @ MakieCore ~/.julia/packages/MakieCore/NeQjl/src/attributes.jl:35
@doc (@doc HDF5.attributes)
attributes(p::Union{HDF5.Dataset, HDF5.Datatype, HDF5.File, HDF5.Group}) = HDF5.attributes(p)
@doc (@doc Makie.attributes)
attributes(x::Attributes) = Makie.attributes(x)
attributes(x::AbstractPlot) = Makie.attributes(x)
export attributes
push!(overrides, :attributes)

## :center
# Showing duplicate methods for center in packages Module[Images, Graphs]
# Methods for center in package ImageTransformations
# center(img::AbstractArray{T, N}) where {T, N} @ ImageTransformations ~/.julia/packages/ImageTransformations/LUoQM/src/ImageTransformations.jl:80
# Methods for center in package Graphs
# center(g::AbstractGraph, distmx::AbstractMatrix) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/distance.jl:198
# center(g::AbstractGraph) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/distance.jl:198
# center(eccentricities::Vector) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/distance.jl:193
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
# centered(R::Type{HyperRectangle}) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/rectangles.jl:537
# centered(S::Type{HyperSphere{N, T}}) where {N, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/spheres.jl:39
# centered(::Type{T}) where T<:HyperSphere @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/spheres.jl:40
# centered(R::Type{HyperRectangle{N}}) where N @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/rectangles.jl:536
# centered(R::Type{HyperRectangle{N, T}}) where {N, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/rectangles.jl:535
# Methods for centered in package OffsetArrays
# centered(ax::AxisArrays.Axis{name}) where name @ ImageAxes ~/.julia/packages/ImageAxes/wmYCV/src/offsetarrays.jl:5
# centered(a::ImageMeta) @ ImageMetadata ~/.julia/packages/ImageMetadata/QsAtm/src/ImageMetadata.jl:272
# centered(a::AxisArray) @ ImageAxes ~/.julia/packages/ImageAxes/wmYCV/src/offsetarrays.jl:6
# centered(A::ImageMorphology.StructuringElements.MorphologySEArray) @ ImageMorphology.StructuringElements ~/.julia/packages/ImageMorphology/zxdrG/src/StructuringElements/StructuringElements.jl:20
# centered(A::AbstractArray) @ OffsetArrays ~/.julia/packages/OffsetArrays/hwmnB/src/OffsetArrays.jl:823
# centered(A::AbstractArray, cp::Tuple{Vararg{Int64, N}} where N) @ OffsetArrays ~/.julia/packages/OffsetArrays/hwmnB/src/OffsetArrays.jl:823
# centered(A::AbstractArray, i::CartesianIndex) @ OffsetArrays ~/.julia/packages/OffsetArrays/hwmnB/src/OffsetArrays.jl:825
# centered(A::AbstractArray, r::RoundingMode) @ OffsetArrays deprecated.jl:103
@doc (@doc GeometryBasics.centered)
centered(R::Union{HyperRectangle, HyperRectangle{N}, HyperSphere{N, T}, HyperRectangle{N, T}}) where {N, T} = GeometryBasics.centered(R)
centered(::Type{T}) where T<:HyperSphere = GeometryBasics.centered(T)
@doc (@doc OffsetArrays.centered)
centered(ax::Images.Axis) = OffsetArrays.centered(ax)
centered(a::ImageMeta) = OffsetArrays.centered(a)
centered(a::AxisArray) = OffsetArrays.centered(a)
centered(A::ImageMorphology.StructuringElements.MorphologySEArray) = OffsetArrays.centered(A)
centered(A::AbstractArray) = OffsetArrays.centered(A)
centered(A::AbstractArray, r) = OffsetArrays.centered(A, r)
export centered
push!(overrides, :centered)

## :complement
# Showing duplicate methods for complement in packages Module[DataStructures, Images, Graphs]
# Methods for complement in package DataStructures
# complement(s::DataStructures.IntSet) @ DataStructures ~/.julia/packages/DataStructures/95DJa/src/int_set.jl:193
# Methods for complement in package ColorVectorSpace
# complement(x::TransparentColor) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/tLy1N/src/ColorVectorSpace.jl:238
# complement(x::Union{Number, Colorant}) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/tLy1N/src/ColorVectorSpace.jl:237
# Methods for complement in package Graphs
# complement(g::SimpleDiGraph) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/operators.jl:49
# complement(g::SimpleGraph) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/operators.jl:36
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

## :conv
# Showing duplicate methods for conv in packages Module[Flux, DSP]
# Methods for conv in package NNlib
# conv(a::AbstractArray{Flux.NilNumber.Nil}, b::AbstractArray{Flux.NilNumber.Nil}, dims::DenseConvDims) @ Flux ~/.julia/packages/Flux/hiqg1/src/outputsize.jl:148
# conv(a::AbstractArray{<:Real}, b::AbstractArray{Flux.NilNumber.Nil}, dims::DenseConvDims) @ Flux ~/.julia/packages/Flux/hiqg1/src/outputsize.jl:152
# conv(a::AbstractArray{Flux.NilNumber.Nil}, b::AbstractArray{<:Real}, dims::DenseConvDims) @ Flux ~/.julia/packages/Flux/hiqg1/src/outputsize.jl:156
# conv(x::AbstractArray{xT, N}, w::AbstractArray{wT, N}, cdims::ConvDims; kwargs...) where {xT, wT, N} @ NNlib ~/.julia/packages/NNlib/CkJqS/src/conv.jl:83
# conv(x, w::AbstractArray{T, N}; stride, pad, dilation, flipped, groups) where {T, N} @ NNlib ~/.julia/packages/NNlib/CkJqS/src/conv.jl:50
# Methods for conv in package DSP
# conv(u::AbstractVector{T}, v::AbstractVector{T}, A::AbstractMatrix{T}) where T @ DSP ~/.julia/packages/DSP/eKP6r/src/dspbase.jl:739
# conv(u::AbstractArray{T, N}, v::AbstractArray{T, N}) where {T<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N} @ DSP ~/.julia/packages/DSP/eKP6r/src/dspbase.jl:689
# conv(u::AbstractArray{<:Integer, N}, v::AbstractArray{<:Integer, N}) where N @ DSP ~/.julia/packages/DSP/eKP6r/src/dspbase.jl:706
# conv(u::AbstractArray{<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N}, v::AbstractArray{<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N}) where N @ DSP ~/.julia/packages/DSP/eKP6r/src/dspbase.jl:700
# conv(u::AbstractArray{<:Number, N}, v::AbstractArray{<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N}) where N @ DSP ~/.julia/packages/DSP/eKP6r/src/dspbase.jl:712
# conv(u::AbstractArray{<:Union{AbstractFloat, Complex{T} where T<:AbstractFloat}, N}, v::AbstractArray{<:Number, N}) where N @ DSP ~/.julia/packages/DSP/eKP6r/src/dspbase.jl:717
# conv(u::AbstractArray{<:Number, N}, v::AbstractArray{<:Number, N}) where N @ DSP ~/.julia/packages/DSP/eKP6r/src/dspbase.jl:709
# conv(A::AbstractArray{<:Number, M}, B::AbstractArray{<:Number, N}) where {M, N} @ DSP ~/.julia/packages/DSP/eKP6r/src/dspbase.jl:722
conv = DSP.conv # the ML folks will have to specialize in their modules... 
export conv
push!(overrides, :conv)

## :degree
# Showing duplicate methods for degree in packages Module[Polynomials, Graphs]
# Methods for degree in package Polynomials
# degree(p::Polynomials.MutableDensePolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomial-container-types/mutable-dense-polynomial.jl:103
# degree(p::Polynomials.ImmutableDensePolynomial{B, T, X, 0}) where {B, T, X} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomial-container-types/immutable-dense-polynomial.jl:262
# degree(p::Polynomials.ImmutableDensePolynomial{B, T, X, N}) where {B, T, X, N} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomial-container-types/immutable-dense-polynomial.jl:263
# degree(p::Polynomials.MutableDenseViewPolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomial-container-types/mutable-dense-view-polynomial.jl:64
# degree(p::Polynomials.AbstractDenseUnivariatePolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/abstract-polynomial.jl:123
# degree(p::Polynomials.MutableDenseLaurentPolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomial-container-types/mutable-dense-laurent-polynomial.jl:106
# degree(p::Polynomials.MutableSparsePolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomial-container-types/mutable-sparse-polynomial.jl:93
# degree(p::Polynomials.AbstractLaurentUnivariatePolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/abstract-polynomial.jl:124
# degree(pq::Polynomials.AbstractRationalFunction) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/rational-functions/common.jl:222
# degree(p::P) where P<:FactoredPolynomial @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/factored_polynomial.jl:232
# degree(p::Polynomials.MutableSparseVectorPolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomial-container-types/mutable-sparse-vector-polynomial.jl:96
# degree(p::AbstractPolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:702
# Methods for degree in package Graphs
# degree(g::AbstractGraph, v::Integer) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/core.jl:130
# degree(g::AbstractGraph, vs) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/core.jl:137
# degree(g::AbstractGraph) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/core.jl:137
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
# density() @ Makie ~/.julia/packages/MakieCore/NeQjl/src/recipes.jl:432
# density(args...; kw...) @ Makie ~/.julia/packages/MakieCore/NeQjl/src/recipes.jl:447
# Methods for density in package Graphs
# density(::Type{SimpleTraits.Not{IsDirected{var"##228"}}}, g::var"##228") where var"##228" @ Graphs ~/.julia/packages/Graphs/1ALGD/src/core.jl:393
# density(::Type{IsDirected{var"##227"}}, g::var"##227") where var"##227" @ Graphs ~/.julia/packages/Graphs/1ALGD/src/core.jl:392
# density(g::var"##227") where var"##227" @ Graphs ~/.julia/packages/SimpleTraits/l1ZsK/src/SimpleTraits.jl:331
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
# derivative(p::Polynomials.ImmutableDensePolynomial{B, T, X, 0}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/immutable-polynomial.jl:142
# derivative(p::Polynomials.ImmutableDensePolynomial{B, T, X, N}) where {B<:StandardBasis, T, X, N} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/immutable-polynomial.jl:143
# derivative(p::P) where {B<:ChebyshevTBasis, T, X, P<:MutableDensePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/chebyshev.jl:182
# derivative(p::P) where {B<:StandardBasis, T, X, P<:AbstractDenseUnivariatePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:126
# derivative(p::Polynomials.MutableSparsePolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/sparse-polynomial.jl:89
# derivative(p::LaurentPolynomial{T, X}) where {T, X} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/laurent-polynomial.jl:155
# derivative(p::P) where {B<:StandardBasis, T, X, P<:AbstractLaurentUnivariatePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:140
# derivative(pq::P, n::Int64) where P<:AbstractRationalFunction @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/rational-functions/common.jl:368
# derivative(pq::P) where P<:AbstractRationalFunction @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/rational-functions/common.jl:368
# derivative(p::P, n::Int64) where P<:FactoredPolynomial @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/factored_polynomial.jl:372
# derivative(p::P) where P<:FactoredPolynomial @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/factored_polynomial.jl:372
# derivative(p::Polynomials.MutableSparseVectorPolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/sparse-vector-polynomial.jl:55
# derivative(p::AbstractUnivariatePolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/abstract-polynomial.jl:231
# derivative(p::AbstractUnivariatePolynomial, n::Int64) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/abstract-polynomial.jl:231
# Methods for differentiate in package TaylorSeries
# differentiate(n::Int64, a::Taylor1{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:99
# differentiate(a::Taylor1) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:18
# differentiate(a::Taylor1{T}, n::Int64) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:79
# differentiate(ntup::Tuple{Vararg{Int64, N}}, a::TaylorN) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:211
# differentiate(a::TaylorN, ntup::Tuple{Vararg{Int64, N}}) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:187
# differentiate(a::TaylorN, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:177
# differentiate(a::TaylorN) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:168
# differentiate(a::TaylorN, r) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:168
# differentiate(a::HomogeneousPolynomial, r::Int64) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:136
# differentiate(a::HomogeneousPolynomial, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:159
@doc (@doc Polynomials.derivative)
derivative(p::AbstractPolynomial) = Polynomials.derivative(p)
derivative(p::AbstractPolynomial, n::Integer) = Polynomials.derivative(p, n)
derivative(pq::Polynomials.AbstractRationalFunction) = Polynomials.derivative(pq)
derivative(pq::Polynomials.AbstractRationalFunction, n::Integer) = Polynomials.derivative(pq, n)
@doc (@doc TaylorSeries.differentiate)
derivative(a::AbstractSeries) = TaylorSeries.differentiate(a)
derivative(a::AbstractSeries, r) = TaylorSeries.differentiate(a, r)

## :entropy
# Showing duplicate methods for entropy in packages Module[Distributions, StatsBase, Images]
# Methods for entropy in package StatsBase
# entropy(d::Chernoff) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/chernoff.jl:213
# entropy(d::Hypergeometric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/hypergeometric.jl:78
# entropy(d::DiscreteUniform) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/discreteuniform.jl:65
# entropy(d::Poisson{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/poisson.jl:68
# entropy(d::BetaBinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/betabinomial.jl:109
# entropy(d::PGeneralizedGaussian) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/pgeneralizedgaussian.jl:94
# entropy(d::DiscreteNonParametric, b::Real) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/discretenonparametric.jl:204
# entropy(d::UnivariateDistribution, b::Real) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariates.jl:224
# entropy(d::MultivariateDistribution, b::Real) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariates.jl:77
# entropy(d::FDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/fdist.jl:88
# entropy(d::LogNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/lognormal.jl:87
# entropy(d::Rayleigh{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/rayleigh.jl:66
# entropy(d::Beta) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/beta.jl:107
# entropy(d::Dirac{T}) where T @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/dirac.jl:38
# entropy(d::Levy) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/levy.jl:63
# entropy(d::Semicircle) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/semicircle.jl:43
# entropy(d::Chisq) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/chisq.jl:68
# entropy(d::AbstractMvNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvnormal.jl:95
# entropy(d::Wishart) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/wishart.jl:128
# entropy(d::Laplace) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/laplace.jl:72
# entropy(d::Distributions.ProductDistribution{1, 0, <:Tuple}) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/product.jl:103
# entropy(d::Distributions.ProductDistribution{N, 0} where N) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/product.jl:98
# entropy(d::Erlang) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/erlang.jl:74
# entropy(d::Distributions.Censored{D, S, T, T, Nothing} where {D<:(UnivariateDistribution), S<:ValueSupport, T<:Real}) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/censored.jl:245
# entropy(d::Distributions.Censored{D, S, T, Nothing, T} where {D<:(UnivariateDistribution), S<:ValueSupport, T<:Real}) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/censored.jl:263
# entropy(d::Distributions.Censored) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/censored.jl:281
# entropy(d::Gumbel) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/gumbel.jl:82
# entropy(d::Distributions.Normal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normal.jl:76
# entropy(d::WalleniusNoncentralHypergeometric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/noncentralhypergeometric.jl:266
# entropy(d::MvLogNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvlognormal.jl:232
# entropy(d::PoissonBinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/poissonbinomial.jl:110
# entropy(d::Kumaraswamy) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/kumaraswamy.jl:80
# entropy(d::Cauchy) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/cauchy.jl:68
# entropy(d::Pareto) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/pareto.jl:80
# entropy(d::LocationScale{T, Discrete, D} where {T<:Real, D<:Distribution{Univariate, Discrete}}) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/locationscale.jl:124
# entropy(d::LocationScale{T, Continuous, D} where {T<:Real, D<:Distribution{Univariate, Continuous}}) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/locationscale.jl:123
# entropy(d::DiscreteNonParametric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/discretenonparametric.jl:203
# entropy(d::Frechet) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/frechet.jl:103
# entropy(d::Binomial; approx) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/binomial.jl:90
# entropy(d::Chi{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/chi.jl:70
# entropy(d::Bernoulli) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/bernoulli.jl:77
# entropy(d::NormalCanon) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normalcanon.jl:57
# entropy(d::Lindley) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/lindley.jl:74
# entropy(d::LogUniform) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/loguniform.jl:50
# entropy(d::BernoulliLogit) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/bernoullilogit.jl:64
# entropy(d::Uniform) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/uniform.jl:69
# entropy(d::Weibull) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/weibull.jl:88
# entropy(d::Multinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/multinomial.jl:120
# entropy(d::GeneralizedExtremeValue) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/generalizedextremevalue.jl:158
# entropy(d::TDist{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/tdist.jl:70
# entropy(d::VonMises) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/vonmises.jl:61
# entropy(d::Dirichlet) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/dirichlet.jl:109
# entropy(d::Product) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/product.jl:52
# entropy(d::Exponential{T}) where T @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/exponential.jl:64
# entropy(d::InverseGamma) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/inversegamma.jl:84
# entropy(d::SymTriangularDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/symtriangular.jl:66
# entropy(d::Truncated{var"#s689", Continuous, T} where {var"#s689"<:(Distributions.Normal), T<:Real}) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/truncated/normal.jl:108
# entropy(d::Logistic) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/logistic.jl:73
# entropy(d::Gamma) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/gamma.jl:74
# entropy(d::TriangularDist{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/triangular.jl:89
# entropy(d::Distributions.GenericMvTDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvtdist.jl:107
# entropy(d::Arcsine) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/arcsine.jl:72
# entropy(d::Geometric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/geometric.jl:68
# entropy(p) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/scalarstats.jl:735
# entropy(p, b::Real) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/scalarstats.jl:743
# Methods for entropy in package ImageQualityIndexes
# entropy(logᵦ::Log, img::AbstractArray{Bool}) where Log<:Function @ ImageQualityIndexes ~/.julia/packages/ImageQualityIndexes/dLuqZ/src/entropy.jl:62
# entropy(logᵦ::Log, img; nbins) where Log<:Function @ ImageQualityIndexes ~/.julia/packages/ImageQualityIndexes/dLuqZ/src/entropy.jl:54
# entropy(img::AbstractArray; kind, nbins) @ ImageQualityIndexes ~/.julia/packages/ImageQualityIndexes/dLuqZ/src/entropy.jl:53
@doc (@doc Distributions.entropy)
entropy(d::Distributions.Distribution) = Distributions.entropy(d)
@doc (@doc Images.entropy)
entropy(logᵦ::Function, img::AbstractArray{Bool}) = Images.entropy(logᵦ, img)
entropy(logᵦ::Function, img; nbins) = Images.entropy(logᵦ, img; nbins)
entropy(img::AbstractArray; kind, nbins) = Images.entropy(img; kind, nbins)
export entropy  
push!(overrides, :entropy)

## :evaluate
# Showing duplicate methods for evaluate in packages Module[MultivariateStats, Distances, Images, TaylorSeries]
# Methods for evaluate in package MultivariateStats
# evaluate(f::LinearDiscriminant, X::AbstractMatrix) @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/lda.jl:71
# evaluate(f::LinearDiscriminant, x::AbstractVector) @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/lda.jl:64
# Methods for evaluate in package Distances
# evaluate(d::Haversine, p₁::Meshes.Point, p₂::Meshes.Point) @ Meshes ~/.julia/packages/Meshes/1tCoG/src/distances.jl:56
# evaluate(d::SphericalAngle, p₁::Meshes.Point, p₂::Meshes.Point) @ Meshes ~/.julia/packages/Meshes/1tCoG/src/distances.jl:69
# evaluate(d::PreMetric, p₁::Meshes.Point, p₂::Meshes.Point) @ Meshes ~/.julia/packages/Meshes/1tCoG/src/distances.jl:43
# evaluate(d::PreMetric, g::Meshes.Geometry, p::Meshes.Point) @ Meshes ~/.julia/packages/Meshes/1tCoG/src/distances.jl:6
# evaluate(::Euclidean, p::Meshes.Point, l::Meshes.Line) @ Meshes ~/.julia/packages/Meshes/1tCoG/src/distances.jl:13
# evaluate(d::Euclidean, l₁::Meshes.Line, l₂::Meshes.Line) @ Meshes ~/.julia/packages/Meshes/1tCoG/src/distances.jl:26
# evaluate(dist::PreMetric, a, b) @ Distances ~/.julia/packages/Distances/n9q0L/src/generic.jl:24
# Methods for evaluate in package TaylorSeries
# evaluate(A::Array{TaylorN{T}}, δx::Vector{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:449
# evaluate(a::Taylor1{TaylorN{T}}, ind::Int64, dx::TaylorN{T}) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:125
# evaluate(a::Taylor1{TaylorN{T}}, ind::Int64, dx::T) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:114
# evaluate(a::Taylor1{T}, x::Taylor1{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:59
# evaluate(a::Taylor1{T}, dx::Taylor1{TaylorN{T}}) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:94
# evaluate(a::Taylor1{Taylor1{T}}, x::Taylor1{T}) where T<:Union{Real, Complex, Taylor1} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:69
# evaluate(a::Taylor1{T}, x::Taylor1{Taylor1{T}}) where T<:Union{Real, Complex, Taylor1} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:76
# evaluate(a::Taylor1{T}, x::Taylor1{S}) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:56
# evaluate(a::Taylor1, dx::IntervalArithmetic.Interval{S}) where S<:Real @ TaylorSeriesIAExt ~/.julia/packages/TaylorSeries/XsXwM/ext/TaylorSeriesIAExt.jl:162
# evaluate(a::Taylor1{T}, dx::T) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:17
# evaluate(a::Taylor1{T}, dx::TaylorN{T}) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:87
# evaluate(a::Taylor1{T}, dx::S) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:25
# evaluate(a::Taylor1{TaylorN{T}}, dx::Array{TaylorN{T}, 1}) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:107
# evaluate(p::Taylor1{T}, x::AbstractArray{S}) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:83
# evaluate(a::Taylor1{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:33
# evaluate(A::AbstractArray{TaylorN{T}}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:454
# evaluate(A::AbstractArray{TaylorN{T}, N}, δx::Vector{S}) where {T<:Number, S<:Number, N} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:444
# evaluate(x::AbstractArray{Taylor1{T}}, δt::S) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:44
# evaluate(a::AbstractArray{Taylor1{T}}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:47
# evaluate(a::HomogeneousPolynomial, dx::IntervalArithmetic.IntervalBox{N, T}) where {T<:Real, N} @ TaylorSeriesIAExt ~/.julia/packages/TaylorSeries/XsXwM/ext/TaylorSeriesIAExt.jl:193
# evaluate(a::HomogeneousPolynomial{T}) where T @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:169
# evaluate(a::HomogeneousPolynomial{T}, vals::AbstractVector{S}) where {T<:Number, S<:Union{Real, Complex, Taylor1}} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:161
# evaluate(a::HomogeneousPolynomial, vals::Tuple{Vararg{var"#s481", N}} where var"#s481"<:Number) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:156
# evaluate(a::HomogeneousPolynomial, v) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:167
# evaluate(a::HomogeneousPolynomial, v, vals::Vararg{Number, N}) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:164
# evaluate(a::TaylorN{T}, s::Symbol, val::TaylorN) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:295
# evaluate(a::TaylorN{T}, s::Symbol, val::S) where {T<:Number, S<:Union{Real, Complex, Taylor1}} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:281
# evaluate(a::TaylorN{T}, ind::Int64, val::TaylorN) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:301
# evaluate(a::TaylorN{T}, ind::Int64, val::S) where {T<:Number, S<:Union{Real, Complex, Taylor1}} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:288
# evaluate(a::TaylorN, dx::IntervalArithmetic.IntervalBox{N, T}) where {T<:Real, N} @ TaylorSeriesIAExt ~/.julia/packages/TaylorSeries/XsXwM/ext/TaylorSeriesIAExt.jl:182
# evaluate(a::TaylorN{T}, x::Pair{Symbol, S}) where {T, S} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:308
# evaluate(a::TaylorN{T}, vals::AbstractVector{<:AbstractSeries}; sorting) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:273
# evaluate(a::TaylorN{T}, vals::AbstractVector{<:Number}; sorting) where T<:Union{Real, Complex} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:269
# evaluate(a::TaylorN{Taylor1{T}}, vals::AbstractVector{S}; sorting) where {T, S} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:277
# evaluate(a::TaylorN, vals::Tuple{Vararg{var"#s478", N}} where var"#s478"<:AbstractSeries; sorting) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:263
# evaluate(a::TaylorN, vals::Tuple{Vararg{var"#s478", N}} where var"#s478"<:Number; sorting) where N @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:257
# evaluate(a::TaylorN{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/evaluate.jl:311
@doc (@doc MultivariateStats.evaluate)
evaluate(f::MultivariateStats.LinearDiscriminant, X::AbstractMatrix) = MultivariateStats.evaluate(f, X)
evaluate(f::MultivariateStats.LinearDiscriminant, x::AbstractVector) = MultivariateStats.evaluate(f, x)
@doc (@doc Distances.evaluate)
evalaute(d::Distances.PreMetric, a, b) = Distances.evaluate(d, a, b)
@doc (@doc TaylorSeries.evaluate)
evaluate(A::AbstractArray{TaylorN{T}}) where T = TaylorSeries.evaluate(A)
evaluate(A::AbstractArray{TaylorN{T}, N}, δx::Vector{S}) where {T, S, N} = TaylorSeries.evaluate(A, δx)

## :fit
# Showing duplicate methods for fit in packages Module[Distributions, StatsBase, MultivariateStats, Polynomials]
# Methods for fit in package StatsAPI
# fit(::Type{UnitRangeTransform}, X::AbstractVector{<:Real}; dims, unit) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/transformations.jl:287
# fit(::Type{UnitRangeTransform}, X::AbstractMatrix{<:Real}; dims, unit) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/transformations.jl:263
# fit(::Type{T}, data::Tuple{Int64, AbstractArray}) where T<:Binomial @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/binomial.jl:204
# fit(::Type{<:Distributions.Categorical{P} where P<:Real}, data::Tuple{Int64, AbstractArray}) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/categorical.jl:182
# fit(::Type{T}, data::Tuple{Int64, AbstractArray}, w::AbstractArray{<:Real}) where T<:Binomial @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/binomial.jl:205
# fit(::Type{<:Distributions.Categorical{P} where P<:Real}, data::Tuple{Int64, AbstractArray}, w::AbstractArray{Float64}) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/categorical.jl:183
# fit(::Type{MDS}, X::AbstractMatrix{T}; maxoutdim, distances) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/cmds.jl:232
# fit(::Type{<:Beta}, x::AbstractArray{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/beta.jl:241
# fit(::Type{<:Cauchy}, x::AbstractArray{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/cauchy.jl:110
# fit(::Type{<:Rician}, x::AbstractArray{T}; tol, maxiters) where T @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/rician.jl:141
# fit(dt::Type{D}, x) where D<:Distribution @ Distributions ~/.julia/packages/Distributions/uuqsE/src/genericfit.jl:46
# fit(dt::Type{D}, args...) where D<:Distribution @ Distributions ~/.julia/packages/Distributions/uuqsE/src/genericfit.jl:47
# fit(::Type{CCA}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}; outdim, method, xmean, ymean) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/cca.jl:309
# fit(::Type{Whitening}, X::AbstractMatrix{T}; dims, mean, regcoef) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/whiten.jl:124
# fit(::Type{MultivariateStats.KernelCenter}, K::AbstractMatrix{<:Real}) @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/kpca.jl:12
# fit(::Type{Histogram{T}}, v::AbstractVector, wv::AbstractWeights; closed, nbins) where T @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:304
# fit(::Type{Histogram{T}}, v::AbstractVector, edg::AbstractVector; closed) where T @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:298
# fit(::Type{Histogram{T}}, v::AbstractVector; closed, nbins) where T @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:300
# fit(::Type{Histogram{T}}, v::AbstractVector, wv::AbstractWeights, edg::AbstractVector; closed) where T @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:302
# fit(::Type{Histogram}, v::AbstractVector, wv::AbstractWeights{W, T} where T<:Real, args...; kwargs...) where W @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:307
# fit(::Type{Histogram{T}}, vs::Tuple{Vararg{AbstractVector, N}}, edges::Tuple{Vararg{AbstractVector, N}}; closed) where {T, N} @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:353
# fit(::Type{Histogram{T}}, vs::Tuple{Vararg{AbstractVector, N}}; closed, nbins) where {T, N} @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:356
# fit(::Type{Histogram{T}}, vs::Tuple{Vararg{AbstractVector, N}}, wv::AbstractWeights{W, T} where T<:Real, edges::Tuple{Vararg{AbstractVector, N}}; closed) where {T, N, W} @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:359
# fit(::Type{Histogram}, vs::Tuple{Vararg{AbstractVector, N}}, wv::AbstractWeights{W, T} where T<:Real, args...; kwargs...) where {N, W} @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:414
# fit(::Type{Histogram{T}}, vs::Tuple{Vararg{AbstractVector, N}}, wv::AbstractWeights; closed, nbins) where {T, N} @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:362
# fit(::Type{Histogram}, args...; kwargs...) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/hist.jl:413
# fit(::Type{FactorAnalysis}, X::AbstractMatrix{T}; method, maxoutdim, mean, tol, η, maxiter) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/fa.jl:257
# fit(::Type{ICA}, X::AbstractMatrix{T}, k::Int64; alg, fun, do_whiten, maxiter, tol, mean, winit) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/ica.jl:212
# fit(::Type{KernelPCA}, X::AbstractMatrix{T}; kernel, maxoutdim, remove_zero_eig, atol, solver, inverse, β, tol, maxiter) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/kpca.jl:146
# fit(::Type{PPCA}, X::AbstractMatrix{T}; method, maxoutdim, mean, tol, maxiter) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/ppca.jl:306
# fit(::Type{ZScoreTransform}, X::AbstractMatrix{<:Real}; dims, center, scale) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/transformations.jl:109
# fit(::Type{ZScoreTransform}, X::AbstractVector{<:Real}; dims, center, scale) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/transformations.jl:130
# fit(::Type{MulticlassLDA}, X::AbstractMatrix{T}, y::AbstractVector; method, outdim, regcoef, covestimator_within, covestimator_between) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/lda.jl:331
# fit(::Type{SubspaceLDA}, X::AbstractMatrix{T}, y::AbstractVector; normalize) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/lda.jl:463
# fit(::Type{PCA}, X::AbstractMatrix{T}; method, maxoutdim, pratio, mean) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/pca.jl:281
# fit(::Type{MetricMDS}, X::AbstractMatrix{T}; maxoutdim, metric, tol, maxiter, initial, weights, distances) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/mmds.jl:125
# fit(::Type{LinearDiscriminant}, Xp::DenseMatrix{T}, Xn::DenseMatrix{T}; covestimator) where T<:Real @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/lda.jl:142
# Methods for fit in package Polynomials
# fit(::Type{ArnoldiFit}, x::AbstractVector{T}, y::AbstractVector{T}; ...) where T @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:764
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}; ...) where T @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:554
# fit(P::Type{<:AbstractPolynomial}, x::AbstractVector{T}, y::AbstractVector{T}; ...) where T @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:110
# fit(::Type{ArnoldiFit}, x::AbstractVector{T}, y::AbstractVector{T}, deg::Int64; var, kwargs...) where T @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:764
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}, deg::Integer; weights, var) where T @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:554
# fit(P::Type{<:AbstractPolynomial}, x::AbstractVector{T}, y::AbstractVector{T}, deg::Integer; weights, var) where T @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:110
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}, J; ...) where T @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:591
# fit(P::Type{<:AbstractPolynomial}, x, y, deg::Integer; weights, var) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:119
# fit(P::Type{<:AbstractPolynomial}, x, y; ...) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:119
# fit(::Type{RationalFunction}, r::Polynomial, m::Integer, n::Integer; var) @ Polynomials.RationalFunctionFit ~/.julia/packages/Polynomials/6i39P/src/rational-functions/fit.jl:114
# fit(::Type{PQ}, xs::AbstractVector{S}, ys::AbstractVector{T}, m, n; var) where {T, S, PQ<:RationalFunction} @ Polynomials.RationalFunctionFit ~/.julia/packages/Polynomials/6i39P/src/rational-functions/fit.jl:71
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}, J, cs::Dict; weights, var) where T @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:602
# fit(P::Type{<:AbstractUnivariatePolynomial{<:Polynomials.StandardBasis, T, X} where {T, X}}, x::AbstractVector{T}, y::AbstractVector{T}, J, cs; weights, var) where T @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:591
# fit(::Type{P}, x::AbstractVector{T}, y::AbstractVector{T}, deg, cs::Dict; kwargs...) where {T, P<:AbstractUnivariatePolynomial} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/abstract-polynomial.jl:243
# fit(x::AbstractVector, y::AbstractVector; ...) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:134
# fit(x::AbstractVector, y::AbstractVector, deg::Integer; weights, var) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:134
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

## :get_weight
# Showing duplicate methods for get_weight in packages Module[DelaunayTriangulation, SimpleWeightedGraphs]
# Methods for get_weight in package DelaunayTriangulation
# get_weight(tri::Triangulation, i::Integer) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/triangulation/methods/weights.jl:52
# get_weight(weights::DelaunayTriangulation.ZeroWeight{T}, i::Integer) where T @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/triangulation/methods/weights.jl:28
# get_weight(tri::Triangulation, i) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/triangulation/methods/weights.jl:48
# get_weight(weights, i::Integer) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/triangulation/methods/weights.jl:16
# get_weight(weights::DelaunayTriangulation.ZeroWeight{T}, i) where T @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/triangulation/methods/weights.jl:27
# get_weight(weights, i) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/triangulation/methods/weights.jl:17
# Methods for get_weight in package SimpleWeightedGraphs
# get_weight(g::AbstractSimpleWeightedGraph, u::Integer, v::Integer) @ SimpleWeightedGraphs ~/.julia/packages/SimpleWeightedGraphs/byp3k/src/abstractsimpleweightedgraph.jl:34
@doc (@doc DelaunayTriangulation.get_weight)
get_weight(tri::DelaunayTriangulation.Triangulation, i::Integer) = DelaunayTriangulation.get_weight(tri, i)
get_weight(tri::DelaunayTriangulation.ZeroWeight, i::Integer) = DelaunayTriangulation.get_weight(tri, i)
@doc (@doc SimpleWeightedGraphs.get_weight)
get_weight(g::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, u::Integer, v::Integer) = SimpleWeightedGraphs.get_weight(g, u, v)
export get_weight
push!(overrides, :get_weight)


## :groupby
# Showing duplicate methods for groupby in packages Module[DataFrames, IterTools]
# Methods for groupby in package DataAPI
# groupby(df::AbstractDataFrame, cols; sort, skipmissing) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/groupeddataframe/groupeddataframe.jl:218
# Methods for groupby in package IterTools
# groupby(keyfunc::F, xs::I) where {F<:Union{Function, Type}, I} @ IterTools ~/.julia/packages/IterTools/cLYFo/src/IterTools.jl:396
@doc (@doc DataFrames.groupby)
groupby(df::DataFrames.AbstractDataFrame, cols; sort, skipmissing) = DataFrames.groupby(df, cols; sort, skipmissing)
@doc (@doc IterTools.groupby)
groupby(keyfunc::F, xs::I) where {F <: Union{Function, Type}, I} = IterTools.groupby(keyfunc, xs)
export groupby
push!(overrides, :groupby)

## :hamming
# Showing duplicate methods for hamming in packages Module[Distances, Images, DSP]
# Methods for Hamming() in package Distances
# (dist::Hamming)(a, b) @ Distances ~/.julia/packages/Distances/n9q0L/src/metrics.jl:328
# Methods for hamming in package DSP.Windows
# hamming(dims::Tuple; padding, zerophase) @ DSP.Windows ~/.julia/packages/DSP/eKP6r/src/windows.jl:645
# hamming(n::Integer; padding, zerophase) @ DSP.Windows ~/.julia/packages/DSP/eKP6r/src/windows.jl:200
@doc (@doc Distances.hamming)
hamming(a, b) = Distances.hamming(a, b)
@doc (@doc DSP.Windows.hamming)
hamming(dims::Tuple; padding, zerophase) = DSP.Windows.hamming(dims; padding, zerophase)
hamming(n::Integer; padding, zerophase) = DSP.Windows.hamming(n; padding, zerophase)
export hamming
push!(overrides, :hamming)

## :height
# Showing duplicate methods for height in packages Module[GeometryBasics, Measures, CairoMakie]
# Methods for height in package GeometryBasics
# height(c::Cylinder{N, T}) where {N, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/cylinders.jl:26
# height(prim::HyperRectangle) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/rectangles.jl:187
# Methods for height in package Measures
# height(x::BoundingBox) @ Measures ~/.julia/packages/Measures/PKOxJ/src/boundingbox.jl:44
@doc (@doc GeometryBasics.height)
height(prim::HyperRectangle) = GeometryBasics.height(prim)
height(prim::Cylinder) = GeometryBasics.height(prim)
@doc (@doc Measures.height)
height(x::BoundingBox) = Measures.height(x)
export height
push!(overrides, :height)

## :imrotate
# Showing duplicate methods for imrotate in packages Module[Flux, Images]
# Methods for imrotate in package NNlib
# imrotate(arr::AbstractArray{T, 4}, θ; method, rotation_center) where T @ NNlib ~/.julia/packages/NNlib/CkJqS/src/rotation.jl:165
# Methods for imrotate in package ImageTransformations
# imrotate(img::AbstractArray{T}, θ::Real; ...) where T @ ImageTransformations ~/.julia/packages/ImageTransformations/LUoQM/src/warp.jl:245
# imrotate(img::AbstractArray{T}, θ::Real, inds::Union{Nothing, Tuple}; kwargs...) where T @ ImageTransformations ~/.julia/packages/ImageTransformations/LUoQM/src/warp.jl:245
# imrotate(img::AbstractArray, θ::Real, method::Union{Interpolations.InterpolationType, Interpolations.Degree}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, fillvalue::Union{Number, Colorant, Interpolations.Flat, Interpolations.Periodic, Interpolations.Reflect}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, method::Union{Interpolations.InterpolationType, Interpolations.Degree}, fillvalue::Union{Number, Colorant, Interpolations.Flat, Interpolations.Periodic, Interpolations.Reflect}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, fillvalue::Union{Number, Colorant, Interpolations.Flat, Interpolations.Periodic, Interpolations.Reflect}, method::Union{Interpolations.InterpolationType, Interpolations.Degree}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, inds, fillvalue::Union{Number, Colorant, Interpolations.Flat, Interpolations.Periodic, Interpolations.Reflect}, method::Union{Interpolations.InterpolationType, Interpolations.Degree}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, inds, method::Union{Interpolations.InterpolationType, Interpolations.Degree}, fillvalue::Union{Number, Colorant, Interpolations.Flat, Interpolations.Periodic, Interpolations.Reflect}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, inds, fillvalue::Union{Number, Colorant, Interpolations.Flat, Interpolations.Periodic, Interpolations.Reflect}) @ ImageTransformations deprecated.jl:103
# imrotate(img::AbstractArray, θ::Real, inds, method::Union{Interpolations.InterpolationType, Interpolations.Degree}) @ ImageTransformations deprecated.jl:103
imrotate = ImageTransformations.imrotate
export imrotate
push!(overrides, :imrotate)

## :integrate
# Showing duplicate methods for integrate in packages Module[Polynomials, TaylorSeries]
# Methods for integrate in package Polynomials
# integrate(p::P) where {B<:ChebyshevTBasis, T, X, P<:MutableDensePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/chebyshev.jl:206
# integrate(p::Polynomials.ImmutableDensePolynomial{B, T, X, 0}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/immutable-polynomial.jl:151
# integrate(p::Polynomials.ImmutableDensePolynomial{B, T, X, N}) where {B<:StandardBasis, T, X, N} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/immutable-polynomial.jl:153
# integrate(p::P) where {B<:StandardBasis, T, X, P<:AbstractDenseUnivariatePolynomial{B, T, X}} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:144
# integrate(p::Polynomials.MutableSparsePolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/sparse-polynomial.jl:104
# integrate(p::Polynomials.AbstractLaurentUnivariatePolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/standard-basis.jl:162
# integrate(pq::P) where P<:AbstractRationalFunction @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/rational-functions/common.jl:378
# integrate(p::P) where P<:FactoredPolynomial @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/factored_polynomial.jl:365
# integrate(p::Polynomials.MutableSparseVectorPolynomial{B, T, X}) where {B<:StandardBasis, T, X} @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/polynomials/standard-basis/sparse-vector-polynomial.jl:73
# integrate(P::AbstractPolynomial) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:228
# integrate(p::P, C) where P<:AbstractPolynomial @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:236
# integrate(p::AbstractPolynomial, a, b) @ Polynomials ~/.julia/packages/Polynomials/6i39P/src/common.jl:248
# Methods for integrate in package TaylorSeries
# integrate(a::Taylor1{T}, x::S) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:113
# integrate(a::Taylor1{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:125
# integrate(a::TaylorN, s::Symbol, x0::TaylorN) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:423
# integrate(a::TaylorN, s::Symbol, x0) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:424
# integrate(a::TaylorN, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:422
# integrate(a::TaylorN, r::Int64) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:399
# integrate(a::TaylorN, r::Int64, x0::TaylorN) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:410
# integrate(a::TaylorN, r::Int64, x0) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:419
# integrate(a::HomogeneousPolynomial, r::Int64) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:362
# integrate(a::HomogeneousPolynomial, s::Symbol) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/calculus.jl:389
@doc (@doc Polynomials.integrate)
integrate(p::AbstractPolynomial) = Polynomials.integrate(p)
integrate(p::AbstractPolynomial, a, b) = Polynomials.integrate(p, a, b)
integrate(p::AbstractPolynomial, C) = Polynomials.integrate(p, C)
integrate(pq::Polynomials.AbstractRationalFunction) = Polynomials.integrate(pq)
@doc (@doc TaylorSeries.integrate)
integrate(a::AbstractSeries) = TaylorSeries.integrate(a)
integrate(a::AbstractSeries, r) = TaylorSeries.integrate(a, r)
integrate(a::AbstractSeries, r, x0) = TaylorSeries.integrate(a, r, x0)

## :islinear
# Showing duplicate methods for islinear in packages Module[StatsBase, DifferentialEquations]
# Methods for islinear in package StatsAPI
# Methods for islinear in package SciMLOperators
# islinear(::IdentityOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/basic.jl:33
# islinear(::NullOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/basic.jl:126
# islinear(::MatrixOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/matrix.jl:102
# islinear(f::ODEFunction) @ SciMLBase ~/.julia/packages/SciMLBase/PTTHz/src/scimlfunctions.jl:4469
# islinear(f::SplitFunction) @ SciMLBase ~/.julia/packages/SciMLBase/PTTHz/src/scimlfunctions.jl:4470
# islinear(::SciMLBase.AbstractDiffEqFunction) @ SciMLBase ~/.julia/packages/SciMLBase/PTTHz/src/scimlfunctions.jl:4468
# islinear(L::FunctionOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/func.jl:579
# islinear(L::SciMLOperators.ScaledOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/basic.jl:250
# islinear(L::SciMLOperators.TransposedOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/left.jl:78
# islinear(L::SciMLOperators.AddedOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/basic.jl:418
# islinear(::AffineOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/matrix.jl:536
# islinear(::SciMLOperators.AbstractSciMLScalarOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/scalar.jl:32
# islinear(o::SciMLBase.AbstractDiffEqLinearOperator) @ SciMLBase ~/.julia/packages/SciMLBase/PTTHz/src/operators/operators.jl:4
# islinear(L::SciMLOperators.ComposedOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/basic.jl:583
# islinear(::SciMLOperators.BatchedDiagonalOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/batch.jl:98
# islinear(L::SciMLOperators.InvertedOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/basic.jl:766
# islinear(L::SciMLOperators.AdjointOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/left.jl:77
# islinear(L::TensorProductOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/tensor.jl:117
# islinear(L::InvertibleOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/matrix.jl:343
# islinear(::SciMLOperators.AbstractSciMLOperator) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/interface.jl:309
# islinear(::Union{Number, Factorization, UniformScaling, AbstractMatrix}) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/interface.jl:311
# islinear(L) @ SciMLBase ~/.julia/packages/SciMLBase/PTTHz/src/operators/operators.jl:7
islinear = DifferentialEquations.islinear

## :issquare
# Showing duplicate methods for issquare in packages Module[DoubleFloats, DifferentialEquations]
# Methods for issquare in package DoubleFloats
# issquare(m::Array{DoubleFloat{T}, 2}) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/iU3tv/src/math/linearalgebra/support.jl:6
# issquare(m::AbstractMatrix{T}) where T<:Number @ DoubleFloats ~/.julia/packages/DoubleFloats/iU3tv/src/math/linearalgebra/support.jl:1
# Methods for issquare in package SciMLOperators
# issquare(x::MatrixOperator, args...; kwargs...) @ SciMLOperators ~/.julia/packages/MacroTools/Cf2ok/src/examples/forward.jl:17
# issquare(::Union{Number, UniformScaling, SciMLOperators.AbstractSciMLScalarOperator}) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/interface.jl:361
# issquare(::AbstractVector) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/interface.jl:360
# issquare(L) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/interface.jl:359
# issquare(A...) @ SciMLOperators ~/.julia/packages/SciMLOperators/778OM/src/interface.jl:366
issquare = DifferentialEquations.issquare

## :meanad
# Showing duplicate methods for meanad in packages Module[StatsBase, Distances]
# Methods for meanad in package StatsBase
# meanad(a::AbstractArray{T}, b::AbstractArray{T}) where T<:Number @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/deviation.jl:140
# Methods for MeanAbsDeviation() in package Distances
# (::MeanAbsDeviation)(a, b) @ Distances ~/.julia/packages/Distances/n9q0L/src/metrics.jl:590
meanad = Distances.MeanAbsDeviation()
export meanad
push!(overrides, :meanad)

## :metadata
# Showing duplicate methods for metadata in packages Module[DataFrames, FileIO]
# Methods for metadata in package DataAPI
# metadata(df::DataFrame, key::AbstractString; ...) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/other/metadata.jl:102
# metadata(df::DataFrame, key::AbstractString, default; style) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/other/metadata.jl:102
# metadata(x::Union{DataFrames.DataFrameColumns, DataFrames.DataFrameRows}, key::AbstractString; ...) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/other/metadata.jl:115
# metadata(x::Union{DataFrames.DataFrameColumns, DataFrames.DataFrameRows}, key::AbstractString, default; style) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/other/metadata.jl:115
# metadata(x::Union{DataFrameRow, SubDataFrame}, key::AbstractString; ...) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/other/metadata.jl:119
# metadata(x::Union{DataFrameRow, SubDataFrame}, key::AbstractString, default; style) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/other/metadata.jl:119
# metadata(x::T; style) where T @ DataAPI ~/.julia/packages/DataAPI/atdEM/src/DataAPI.jl:371
# Methods for metadata in package FileIO
# metadata(file::Formatted, args...; options...) @ FileIO ~/.julia/packages/FileIO/xOKyx/src/loadsave.jl:116
# metadata(file, args...; options...) @ FileIO ~/.julia/packages/FileIO/xOKyx/src/loadsave.jl:109
@doc (@doc DataFrames.metadata)
#metadata(x; style) = DataFrames.metadata(x; style)
metadata(df::Union{DataFrame, DataFrames.DataFrameColumns, DataFrames.DataFrameRows, DataFrameRow, SubDataFrame}, key::AbstractString; kwargs...) = DataFrames.metadata(df, key; kwargs...)
metadata(df::Union{DataFrame, DataFrames.DataFrameColumns, DataFrames.DataFrameRows, DataFrameRow, SubDataFrame}, key::AbstractString, default; style) = DataFrames.metadata(df, key, default; style)
@doc (@doc FileIO.metadata)
metadata(file::FileIO.Formatted, args...; options...) = FileIO.metadata(file, args...; options...)
export metadata
push!(overrides, :metadata)

## :mode
# Showing duplicate methods for mode in packages Module[Distributions, StatsBase, JuMP]
# Methods for mode in package StatsBase
# mode(d::Kolmogorov) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/kolmogorov.jl:26
# mode(d::Chernoff) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/chernoff.jl:209
# mode(d::Hypergeometric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/hypergeometric.jl:56
# mode(d::DiscreteUniform) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/discreteuniform.jl:67
# mode(d::PGeneralizedGaussian) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/pgeneralizedgaussian.jl:87
# mode(d::BetaBinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/betabinomial.jl:111
# mode(d::Rayleigh) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/rayleigh.jl:58
# mode(d::Semicircle) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/semicircle.jl:42
# mode(d::Geometric{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/geometric.jl:60
# mode(d::LogNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/lognormal.jl:64
# mode(d::Beta; check_args) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/beta.jl:67
# mode(d::Levy) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/levy.jl:61
# mode(d::Chisq{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/chisq.jl:58
# mode(d::Laplace) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/laplace.jl:65
# mode(d::Erlang; check_args) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/erlang.jl:65
# mode(d::SkewNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/skewnormal.jl:62
# mode(d::Distributions.Normal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normal.jl:69
# mode(d::WalleniusNoncentralHypergeometric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/noncentralhypergeometric.jl:264
# mode(d::Biweight) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/biweight.jl:28
# mode(d::PoissonBinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/poissonbinomial.jl:112
# mode(d::Kumaraswamy) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/kumaraswamy.jl:134
# mode(d::BetaPrime{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/betaprime.jl:68
# mode(d::Cauchy) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/cauchy.jl:62
# mode(d::SkewedExponentialPower) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/skewedexponentialpower.jl:80
# mode(d::LogUniform) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/loguniform.jl:47
# mode(d::Chi; check_args) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/chi.jl:73
# mode(d::Frechet) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/frechet.jl:67
# mode(d::Bernoulli) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/bernoulli.jl:67
# mode(d::NormalCanon) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normalcanon.jl:49
# mode(d::Lindley) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/lindley.jl:57
# mode(d::Binomial{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/binomial.jl:70
# mode(d::GeneralizedExtremeValue) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/generalizedextremevalue.jl:105
# mode(d::TDist{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/tdist.jl:53
# mode(d::VonMises) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/vonmises.jl:57
# mode(d::MatrixNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/matrixnormal.jl:91
# mode(d::Distributions.ReshapedDistribution) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/reshaped.jl:44
# mode(::Exponential{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/exponential.jl:58
# mode(d::InverseGamma) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/inversegamma.jl:67
# mode(d::Truncated{var"#s689", Continuous, T, TL, TU} where {var"#s689"<:(Distributions.Normal), TL<:Union{Nothing, T}, TU<:Union{Nothing, T}}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/truncated/normal.jl:3
# mode(d::Logistic) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/logistic.jl:66
# mode(d::Rician) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/rician.jl:72
# mode(d::Gamma) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/gamma.jl:69
# mode(d::TriangularDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/triangular.jl:64
# mode(d::FisherNoncentralHypergeometric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/noncentralhypergeometric.jl:74
# mode(d::Poisson) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/poisson.jl:55
# mode(d::LKJ; check_args) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/lkj.jl:78
# mode(d::NegativeBinomial{T}) where T @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/negativebinomial.jl:81
# mode(d::Epanechnikov) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/epanechnikov.jl:40
# mode(d::FDist{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/fdist.jl:58
# mode(d::AbstractMvNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvnormal.jl:85
# mode(d::Dirac) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/dirac.jl:36
# mode(d::Wishart) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/wishart.jl:111
# mode(d::Gumbel) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/gumbel.jl:74
# mode(d::MvLogNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvlognormal.jl:224
# mode(d::InverseWishart) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/inversewishart.jl:91
# mode(d::Pareto) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/pareto.jl:63
# mode(d::Distributions.AffineDistribution) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/locationscale.jl:111
# mode(d::DiscreteNonParametric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/discretenonparametric.jl:206
# mode(d::Weibull{T}) where T<:Real @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/weibull.jl:65
# mode(d::Distributions.DirichletCanon) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/dirichlet.jl:135
# mode(d::BernoulliLogit) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/bernoullilogit.jl:54
# mode(d::Uniform) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/uniform.jl:61
# mode(d::InverseGaussian) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/inversegaussian.jl:70
# mode(d::Dirichlet) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/dirichlet.jl:134
# mode(d::SymTriangularDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/symtriangular.jl:60
# mode(d::MatrixTDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/matrixtdist.jl:113
# mode(d::Triweight) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/triweight.jl:38
# mode(d::Cosine) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/cosine.jl:51
# mode(a::AbstractArray{T}, r::UnitRange{T}) where T<:Integer @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/scalarstats.jl:56
# mode(a::AbstractVector, wv::AbstractWeights{T, T1, V} where {T1<:Real, V<:AbstractVector{T1}}) where T<:Real @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/scalarstats.jl:164
# mode(d::LKJCholesky) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/cholesky/lkjcholesky.jl:112
# mode(d::Distributions.GenericMvTDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvtdist.jl:92
# mode(d::Arcsine) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/arcsine.jl:65
# mode(a) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/scalarstats.jl:111
# Methods for mode in package JuMP
# mode(model::GenericModel) @ JuMP ~/.julia/packages/JuMP/6RAQ9/src/JuMP.jl:599
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
# Showing duplicate methods for msd in packages Module[StatsBase, Distances]
# Methods for msd in package StatsBase
# msd(a::AbstractArray{T}, b::AbstractArray{T}) where T<:Number @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/deviation.jl:159
# Methods for MeanSqDeviation() in package Distances
# (::MeanSqDeviation)(a, b) @ Distances ~/.julia/packages/Distances/n9q0L/src/metrics.jl:593
msd = Distances.MeanSqDeviation()
export msd
push!(overrides, :msd)

## :nan
# Showing duplicate methods for nan in packages Module[Images, DoubleFloats]
# Methods for nan in package ColorTypes
# nan(::Type{C}) where {T<:AbstractFloat, C<:(Colorant{T})} @ ColorTypes ~/.julia/packages/ColorTypes/vpFgh/src/traits.jl:470
# nan(::Type{T}) where T<:AbstractFloat @ ColorTypes ~/.julia/packages/ColorTypes/vpFgh/src/traits.jl:469
# Methods for nan in package DoubleFloats
# nan(::Type{DoubleFloat{T}}) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/iU3tv/src/type/specialvalues.jl:4
@doc (@doc Images.nan)
nan(::Type{C}) where {T<:AbstractFloat, C<:(Colorant{T})} = Images.nan(C)
@doc (@doc DoubleFloats.nan)
nan(::Type{DoubleFloat{T}}) where T<:Union{Float16, Float32, Float64} = DoubleFloats.nan(T)
export nan
push!(overrides, :nan)

## :params
# Showing duplicate methods for params in packages Module[Distributions, BenchmarkTools]
# Methods for params in package StatsAPI
# params(Ω::Soliton) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/soliton.jl:87
# params(d::Kolmogorov) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/kolmogorov.jl:17
# params(d::Hypergeometric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/hypergeometric.jl:44
# params(d::DiscreteUniform) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/discreteuniform.jl:43
# params(::Type{D}, m::AbstractVector, S::AbstractMatrix) where D<:AbstractMvLogNormal @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvlognormal.jl:141
# params(d::PGeneralizedGaussian) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/pgeneralizedgaussian.jl:77
# params(d::BetaBinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/betabinomial.jl:52
# params(d::NoncentralHypergeometric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/noncentralhypergeometric.jl:31
# params(d::Rayleigh) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/rayleigh.jl:50
# params(d::Semicircle) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/semicircle.jl:36
# params(d::LogNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/lognormal.jl:53
# params(d::MatrixBeta) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/matrixbeta.jl:81
# params(d::Beta) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/beta.jl:59
# params(d::Levy) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/levy.jl:50
# params(d::Arcsine) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/arcsine.jl:57
# params(d::Chisq) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/chisq.jl:40
# params(d::JointOrderStatistics) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/jointorderstatistics.jl:89
# params(d::Laplace) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/laplace.jl:57
# params(d::Erlang) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/erlang.jl:55
# params(d::SkewNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/skewnormal.jl:46
# params(d::Distributions.Normal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normal.jl:57
# params(d::UnivariateGMM) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/mixtures/unigmm.jl:33
# params(d::Biweight) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/biweight.jl:22
# params(d::PoissonBinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/poissonbinomial.jl:80
# params(d::GeneralizedPareto) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/generalizedpareto.jl:80
# params(d::BetaPrime) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/betaprime.jl:59
# params(d::Cauchy) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/cauchy.jl:54
# params(d::Kumaraswamy) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/kumaraswamy.jl:45
# params(d::SkewedExponentialPower) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/skewedexponentialpower.jl:61
# params(d::NoncentralChisq) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/noncentralchisq.jl:54
# params(d::Frechet) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/frechet.jl:54
# params(d::Binomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/binomial.jl:62
# params(d::Chi) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/chi.jl:46
# params(d::Bernoulli) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/bernoulli.jl:55
# params(d::NormalCanon) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normalcanon.jl:42
# params(d::Lindley) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/lindley.jl:44
# params(d::LogUniform) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/loguniform.jl:33
# params(d::MvNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvnormal.jl:255
# params(d::MvNormalCanon) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvnormalcanon.jl:155
# params(d::MvLogitNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvlogitnormal.jl:57
# params(d::GeneralizedExtremeValue) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/generalizedextremevalue.jl:74
# params(d::TDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/tdist.jl:45
# params(d::VonMises) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/vonmises.jl:49
# params(d::MatrixNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/matrixnormal.jl:99
# params(d::Distributions.ReshapedDistribution) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/reshaped.jl:30
# params(d::Exponential) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/exponential.jl:51
# params(d::InverseGamma) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/inversegamma.jl:59
# params(d::StudentizedRange) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/studentizedrange.jl:59
# params(d::Truncated) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/truncate.jl:126
# params(d::Logistic) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/logistic.jl:58
# params(d::Rician) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/rician.jl:61
# params(d::Gamma) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/gamma.jl:56
# params(d::TriangularDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/triangular.jl:58
# params(d::Skellam) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/skellam.jl:58
# params(d::Poisson) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/poisson.jl:46
# params(d::JohnsonSU) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/johnsonsu.jl:53
# params(d::NegativeBinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/negativebinomial.jl:62
# params(d::LKJ) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/lkj.jl:94
# params(d::Epanechnikov) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/epanechnikov.jl:34
# params(d::FDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/fdist.jl:50
# params(d::NoncentralF) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/noncentralf.jl:35
# params(d::Distributions.Censored) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/censored.jl:106
# params(d::MixtureModel) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/mixtures/mixturemodel.jl:167
# params(d::Wishart) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/wishart.jl:106
# params(d::LogitNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/logitnormal.jl:86
# params(d::Gumbel) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/gumbel.jl:56
# params(d::MvLogNormal) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvlognormal.jl:192
# params(d::VonMisesFisher) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/vonmisesfisher.jl:59
# params(d::InverseWishart) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/inversewishart.jl:80
# params(d::MatrixFDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/matrixfdist.jl:87
# params(d::Pareto) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/pareto.jl:52
# params(d::Distributions.AffineDistribution) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/locationscale.jl:104
# params(d::Distributions.Categorical{P, Ps}) where {P<:Real, Ps<:AbstractVector{P}} @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/categorical.jl:52
# params(d::DiscreteNonParametric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/discretenonparametric.jl:50
# params(d::NoncentralBeta) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/noncentralbeta.jl:32
# params(d::Weibull) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/weibull.jl:57
# params(d::OrderStatistic) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/orderstatistic.jl:59
# params(d::NormalInverseGaussian) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/normalinversegaussian.jl:47
# params(d::Multinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/multinomial.jl:49
# params(d::BernoulliLogit) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/bernoullilogit.jl:40
# params(d::Uniform) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/uniform.jl:50
# params(d::InverseGaussian) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/inversegaussian.jl:57
# params(d::NoncentralT) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/noncentralt.jl:29
# params(d::Dirichlet) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/dirichlet.jl:74
# params(d::SymTriangularDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/symtriangular.jl:52
# params(d::MatrixTDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/matrix/matrixtdist.jl:121
# params(d::Triweight) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/triweight.jl:31
# params(d::Cosine) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/cosine.jl:41
# params(d::LKJCholesky) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/cholesky/lkjcholesky.jl:117
# params(d::Distributions.GenericMvTDist) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvtdist.jl:102
# params(d::DirichletMultinomial) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/dirichletmultinomial.jl:56
# params(d::Geometric) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/discrete/geometric.jl:50
# Methods for params in package BenchmarkTools
# params(b::BenchmarkTools.Benchmark) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/QNsku/src/execution.jl:23
# params(group::BenchmarkGroup) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/QNsku/src/groups.jl:119
# params(t::BenchmarkTools.TrialJudgement) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/QNsku/src/trials.jl:219
# params(t::BenchmarkTools.TrialRatio) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/QNsku/src/trials.jl:170
# params(t::BenchmarkTools.TrialEstimate) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/QNsku/src/trials.jl:142
# params(t::BenchmarkTools.Trial) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/QNsku/src/trials.jl:61
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

## :properties
# Showing duplicate methods for properties in packages Module[Images, IterTools]
# Methods for properties in package ImageMetadata
# properties(img::ImageMeta) @ ImageMetadata ~/.julia/packages/ImageMetadata/QsAtm/src/ImageMetadata.jl:291
# Methods for properties in package IterTools
# properties(x::T) where T @ IterTools ~/.julia/packages/IterTools/cLYFo/src/IterTools.jl:962
@doc (@doc Images.properties)
properties(x::ImageMeta) = Images.properties(x)
@doc (@doc IterTools.properties)
properties(x) = IterTools.properties(x)
export properties
push!(overrides, :properties)

## :radius
# Showing duplicate methods for radius in packages Module[GeometryBasics, Graphs]
# Methods for radius in package GeometryBasics
# radius(c::Cylinder{N, T}) where {N, T} @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/cylinders.jl:25
# radius(c::HyperSphere) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/spheres.jl:29
# Methods for radius in package Graphs
# radius(g::AbstractGraph, distmx::AbstractMatrix) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/distance.jl:168
# radius(g::AbstractGraph) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/distance.jl:168
# radius(eccentricities::Vector) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/distance.jl:167
@doc (@doc GeometryBasics.radius)
radius(x::Cylinder) = GeometryBasics.radius(x)
radius(x::HyperSphere) = GeometryBasics.radius(x)
@doc (@doc Graphs.radius)
radius(x::AbstractGraph) = Graphs.radius(x)
radius(x::AbstractGraph, y) = Graphs.radius(x, y)
export radius
push!(overrides, :radius)

## :reset!
# Showing duplicate methods for reset! in packages Module[DataStructures, DSP]
# Methods for reset! in package DataStructures
# reset!(ct::Accumulator{<:Any, V}, x) where V @ DataStructures ~/.julia/packages/DataStructures/95DJa/src/accumulator.jl:134
# reset!(blk::DataStructures.DequeBlock{T}, front::Int64) where T @ DataStructures ~/.julia/packages/DataStructures/95DJa/src/deque.jl:40
# Methods for reset! in package DSP.Filters
# reset!(self::FIRFilter) @ DSP.Filters ~/.julia/packages/DSP/eKP6r/src/Filters/stream_filt.jl:270
# reset!(kernel::DSP.Filters.FIRArbitrary) @ DSP.Filters ~/.julia/packages/DSP/eKP6r/src/Filters/stream_filt.jl:260
# reset!(kernel::DSP.Filters.FIRDecimator) @ DSP.Filters ~/.julia/packages/DSP/eKP6r/src/Filters/stream_filt.jl:255
# reset!(kernel::DSP.Filters.FIRRational) @ DSP.Filters ~/.julia/packages/DSP/eKP6r/src/Filters/stream_filt.jl:249
# reset!(kernel::DSP.Filters.FIRKernel) @ DSP.Filters ~/.julia/packages/DSP/eKP6r/src/Filters/stream_filt.jl:245
@doc (@doc DataStructures.reset!)
reset!(x::Accumulator, y) = DataStructures.reset!(x, y)
reset!(x::DataStructures.DequeBlock, y) = DataStructures.reset!(x, y)
@doc (@doc DSP.Filters.reset!)
reset!(x::Union{FIRFilter, DSP.Filters.FIRKernel, DSP.Filters.FIRArbitrary, DSP.Filters.FIRRational, DSP.Filters.FIRDecimator}) = DSP.reset!(x)
export reset!
push!(overrides, :reset!)

## :right
# Showing duplicate methods for right in packages Module[Transducers, CairoMakie]
# Methods for right in package Transducers
# right(r) @ Transducers ~/.julia/packages/Transducers/txnl6/src/core.jl:869
# right(::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(Transducers.right)}}, x::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(Transducers.right)}}) @ Transducers ~/.julia/packages/InitialValues/OWP8V/src/InitialValues.jl:160
# right(::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(Transducers.right)}}, x) @ Transducers ~/.julia/packages/InitialValues/OWP8V/src/InitialValues.jl:154
# right(x, ::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(Transducers.right)}}) @ Transducers ~/.julia/packages/InitialValues/OWP8V/src/InitialValues.jl:161
# right(l, r) @ Transducers ~/.julia/packages/Transducers/txnl6/src/core.jl:868
# Methods for right in package Makie
# right(rect::Rect2) @ Makie ~/.julia/packages/Makie/YkotL/src/makielayout/geometrybasics_extension.jl:3
@doc (@doc Transducers.right)
right(x) = Transducers.right(x)
right(l,r) = Transducers.right(l,r)
@doc (@doc Makie.right)
right(x::Rect2) = Makie.right(x)

## :rmsd
# Showing duplicate methods for rmsd in packages Module[StatsBase, Distances]
# Methods for rmsd in package StatsBase
# rmsd(a::AbstractArray{T}, b::AbstractArray{T}; normalize) where T<:Number @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/deviation.jl:171
# Methods for RMSDeviation() in package Distances
# (::RMSDeviation)(a, b) @ Distances ~/.julia/packages/Distances/n9q0L/src/metrics.jl:596
rmsd = Distances.RMSDeviation()
export rmsd
push!(overrides, :rmsd)

## :rotate!
# Showing duplicate methods for rotate! in packages Module[LinearAlgebra, CairoMakie]
# Methods for rotate! in package LinearAlgebra
# rotate!(x::AbstractVector, y::AbstractVector, c, s) @ LinearAlgebra /Applications/Julia-1.10.app/Contents/Resources/julia/share/julia/stdlib/v1.10/LinearAlgebra/src/generic.jl:1535
# Methods for rotate! in package Makie
# rotate!(l::RectLight, q...) @ Makie ~/.julia/packages/Makie/YkotL/src/lighting.jl:194
# rotate!(t::MakieCore.Transformable, axis_rot::AbstractFloat) @ Makie ~/.julia/packages/Makie/YkotL/src/layouting/transformation.jl:126
# rotate!(t::MakieCore.Transformable, axis_rot::Quaternion) @ Makie ~/.julia/packages/Makie/YkotL/src/layouting/transformation.jl:125
# rotate!(t::MakieCore.Transformable, axis_rot...) @ Makie ~/.julia/packages/Makie/YkotL/src/layouting/transformation.jl:124
# rotate!(::Type{T}, t::MakieCore.Transformable, q) where T @ Makie ~/.julia/packages/Makie/YkotL/src/layouting/transformation.jl:98
# rotate!(::Type{T}, t::MakieCore.Transformable, axis_rot...) where T @ Makie ~/.julia/packages/Makie/YkotL/src/layouting/transformation.jl:115
@doc (@doc LinearAlgebra.rotate!)
rotate!(x::AbstractVector, y::AbstractVector, c, s) = LinearAlgebra.rotate!(x, y, c, s)
@doc (@doc Makie.rotate!)
rotate!(x::RectLight, y...) = Makie.rotate!(x, y...)
rotate!(x::Makie.Transformable, y) = Makie.rotate!(x, y)
rotate!(::Type{T}, t::Makie.Transformable, y...) where T = Makie.rotate!(T, t, y...)
export rotate!
push!(overrides, :rotate!)

## :scale!
# Showing duplicate methods for scale! in packages Module[Distributions, CairoMakie]
# Methods for scale! in package Distributions
# scale!(::Type{D}, s::Symbol, m::AbstractVector, S::AbstractMatrix, Σ::AbstractMatrix) where D<:AbstractMvLogNormal @ Distributions ~/.julia/packages/Distributions/uuqsE/src/multivariate/mvlognormal.jl:112
# Methods for scale! in package Makie
# scale!(t::MakieCore.Transformable, s) @ Makie ~/.julia/packages/Makie/YkotL/src/layouting/transformation.jl:82
# scale!(t::MakieCore.Transformable, xyz...) @ Makie ~/.julia/packages/Makie/YkotL/src/layouting/transformation.jl:94
# scale!(l::RectLight, xy::Union{Tuple{Vararg{T, N}}, StaticArray{Tuple{N}, T, 1}} where {N, T}) @ Makie ~/.julia/packages/Makie/YkotL/src/lighting.jl:214
# scale!(l::RectLight, x::Real, y::Real) @ Makie ~/.julia/packages/Makie/YkotL/src/lighting.jl:213
# scale!(::Type{T}, l::RectLight, s) where T @ Makie ~/.julia/packages/Makie/YkotL/src/lighting.jl:201
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
# shape(d::PGeneralizedGaussian) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/pgeneralizedgaussian.jl:79
# shape(d::InverseGamma) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/inversegamma.jl:55
# shape(d::Erlang) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/erlang.jl:52
# shape(d::JohnsonSU) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/johnsonsu.jl:50
# shape(d::Lindley) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/lindley.jl:43
# shape(d::GeneralizedPareto) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/generalizedpareto.jl:79
# shape(d::SkewedExponentialPower) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/skewedexponentialpower.jl:63
# shape(d::GeneralizedExtremeValue) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/generalizedextremevalue.jl:71
# shape(d::Gamma) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/gamma.jl:52
# shape(d::InverseGaussian) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/inversegaussian.jl:56
# shape(d::Pareto) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/pareto.jl:49
# shape(d::Rician) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/rician.jl:58
# shape(d::Frechet) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/frechet.jl:52
# shape(d::Weibull) @ Distributions ~/.julia/packages/Distributions/uuqsE/src/univariate/continuous/weibull.jl:54
# Methods for shape in package JuMP
# shape(con::VectorConstraint) @ JuMP ~/.julia/packages/JuMP/6RAQ9/src/constraints.jl:978
# shape(::ScalarConstraint) @ JuMP ~/.julia/packages/JuMP/6RAQ9/src/constraints.jl:881
@doc (@doc Distributions.shape)
shape(x) = Distributions.shape(x) # they get the generic function... 
@doc (@doc JuMP.shape)
shape(con::VectorConstraint) = JuMP.shape(con)
shape(x::ScalarConstraint) = JuMP.shape(x)
export shape
push!(overrides, :shape)

## :solve!
# Showing duplicate methods for solve! in packages Module[Krylov, Roots, DifferentialEquations]
# Methods for solve! in package Krylov
# solve!(solver::TrimrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; M, N, ldiv, spd, snd, flip, sp, τ, ν, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::TrimrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; M, N, ldiv, spd, snd, flip, sp, τ, ν, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::QmrSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; c, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::QmrSolver{T, FC, S}, A, b::AbstractVector{FC}; c, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::BilqSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; c, transfer_to_bicg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::BilqSolver{T, FC, S}, A, b::AbstractVector{FC}; c, transfer_to_bicg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CgLanczosShiftSolver{T, FC, S}, A, b::AbstractVector{FC}, shifts::AbstractVector{T}; M, ldiv, check_curvature, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CgneSolver{T, FC, S}, A, b::AbstractVector{FC}; N, ldiv, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::DqgmresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::DqgmresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CgsSolver{T, FC, S}, A, b::AbstractVector{FC}; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CgsSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::FomSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::FomSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::BicgstabSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::BicgstabSolver{T, FC, S}, A, b::AbstractVector{FC}; c, M, N, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::MinaresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, λ, atol, rtol, Artol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::MinaresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, λ, atol, rtol, Artol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::FgmresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::FgmresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::TrilqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; transfer_to_usymcg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::TrilqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; transfer_to_usymcg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::SymmlqSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, transfer_to_cg, λ, λest, atol, rtol, etol, conlim, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::SymmlqSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, transfer_to_cg, λ, λest, atol, rtol, etol, conlim, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::LsqrSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, sqd, λ, radius, etol, axtol, btol, conlim, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CraigSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, transfer_to_lsqr, sqd, λ, btol, conlim, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::MinresQlpSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, λ, atol, rtol, Artol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::MinresQlpSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, λ, atol, rtol, Artol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::BlockGmresSolver{T, FC, SV, SM}, A, B::AbstractMatrix{FC}, X0::AbstractMatrix; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, SV<:AbstractVector{FC}, SM<:AbstractMatrix{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:81
# solve!(solver::BlockGmresSolver{T, FC, SV, SM}, A, B::AbstractMatrix{FC}; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, SV<:AbstractVector{FC}, SM<:AbstractMatrix{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:68
# solve!(solver::LslqSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, transfer_to_lsqr, sqd, λ, σ, etol, utol, btol, conlim, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CrSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, radius, linesearch, γ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::CrSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, radius, linesearch, γ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CarSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::CarSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CrmrSolver{T, FC, S}, A, b::AbstractVector{FC}; N, ldiv, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CrlsSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, radius, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CglsSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, radius, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CgSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, radius, linesearch, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::CgSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, radius, linesearch, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::GmresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::GmresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, restart, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CraigmrSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, sqd, λ, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::BilqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; transfer_to_bicg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::BilqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; transfer_to_bicg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::GpmrSolver{T, FC, S}, A, B, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; C, D, E, F, ldiv, gsp, λ, μ, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::GpmrSolver{T, FC, S}, A, B, b::AbstractVector{FC}, c::AbstractVector{FC}; C, D, E, F, ldiv, gsp, λ, μ, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::DiomSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, N, ldiv, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::DiomSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, reorthogonalization, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::UsymqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector; atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::UsymqrSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::MinresSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, λ, atol, rtol, etol, conlim, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::MinresSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, λ, atol, rtol, etol, conlim, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::UsymlqSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; transfer_to_usymcg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::UsymlqSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector; transfer_to_usymcg, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::LsmrSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, sqd, λ, radius, etol, axtol, btol, conlim, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::CgLanczosSolver{T, FC, S}, A, b::AbstractVector{FC}, x0::AbstractVector; M, ldiv, check_curvature, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::CgLanczosSolver{T, FC, S}, A, b::AbstractVector{FC}; M, ldiv, check_curvature, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::TricgSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}, x0::AbstractVector, y0::AbstractVector; M, N, ldiv, spd, snd, flip, τ, ν, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:59
# solve!(solver::TricgSolver{T, FC, S}, A, b::AbstractVector{FC}, c::AbstractVector{FC}; M, N, ldiv, spd, snd, flip, τ, ν, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# solve!(solver::LnlqSolver{T, FC, S}, A, b::AbstractVector{FC}; M, N, ldiv, transfer_to_craig, sqd, λ, σ, utolx, utoly, atol, rtol, itmax, timemax, verbose, history, callback, iostream) where {T<:AbstractFloat, FC<:Union{Complex{T}, T}, S<:AbstractVector{FC}} @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solve.jl:46
# Methods for solve! in package CommonSolve
# solve!(cache::NonlinearSolve.NonlinearSolvePolyAlgorithmCache{iip, N}) where {iip, N} @ NonlinearSolve ~/.julia/packages/NonlinearSolve/sBl1H/src/default.jl:141
# solve!(cache::LineSearch.NoLineSearchCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/O1LT8/src/no_search.jl:19
# solve!(cache::LineSearch.RobustNonMonotoneLineSearchCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/O1LT8/src/robust_non_monotone.jl:90
# solve!(cache::LineSearch.LineSearchesJLCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/O1LT8/src/line_searches_ext.jl:128
# solve!(integrator::JumpProcesses.SSAIntegrator) @ JumpProcesses ~/.julia/packages/JumpProcesses/3mbsw/src/SSA_stepper.jl:120
# solve!(cache::LineSearch.StaticLiFukushimaLineSearchCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/O1LT8/src/li_fukushima.jl:136
# solve!(cache::LineSearch.LiFukushimaLineSearchCache, u, du) @ LineSearch ~/.julia/packages/LineSearch/O1LT8/src/li_fukushima.jl:96
# solve!(integrator::OrdinaryDiffEqCore.ODEIntegrator) @ OrdinaryDiffEqCore ~/.julia/packages/OrdinaryDiffEqCore/55UVY/src/solve.jl:544
# solve!(cache::SciMLBase.AbstractOptimizationCache) @ SciMLBase ~/.julia/packages/SciMLBase/PTTHz/src/solve.jl:185
# solve!(cache::LinearSolve.SimpleGMRESCache{false}, lincache::LinearSolve.LinearCache) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/simplegmres.jl:220
# solve!(cache::LinearSolve.SimpleGMRESCache{true}, lincache::LinearSolve.LinearCache) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/simplegmres.jl:450
# solve!(cache::BoundaryValueDiffEq.FIRKCacheExpand) @ BoundaryValueDiffEq ~/.julia/packages/BoundaryValueDiffEq/YN0of/src/solve/firk.jl:289
# solve!(integrator::StochasticDiffEq.SDEIntegrator) @ StochasticDiffEq ~/.julia/packages/StochasticDiffEq/t06NJ/src/solve.jl:611
# solve!(integ::DiffEqBase.NullODEIntegrator) @ DiffEqBase ~/.julia/packages/DiffEqBase/uqSeD/src/solve.jl:643
# solve!(cache::NonlinearSolve.NonlinearSolveNoInitCache) @ NonlinearSolve ~/.julia/packages/NonlinearSolve/sBl1H/src/core/noinit.jl:35
# solve!(cache::NonlinearSolve.NonlinearSolveForwardDiffCache) @ NonlinearSolve ~/.julia/packages/NonlinearSolve/sBl1H/src/internal/forward_diff.jl:51
# solve!(cache::NonlinearSolve.AbstractNonlinearSolveCache) @ NonlinearSolve ~/.julia/packages/NonlinearSolve/sBl1H/src/core/generic.jl:11
# solve!(cache::LinearSolve.LinearCache, alg::LinearSolve.DefaultLinearSolver, args...; assump, kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/default.jl:356
# solve!(cache::LinearSolve.LinearCache, alg::SimpleLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/simplelu.jl:133
# solve!(cache::LinearSolve.LinearCache, alg::UMFPACKFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:816
# solve!(cache::LinearSolve.LinearCache, alg::DirectLdiv!, args...; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/solve_function.jl:17
# solve!(cache::LinearSolve.LinearCache, alg::DiagonalFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:1199
# solve!(cache::LinearSolve.LinearCache, alg::SparspakFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:1373
# solve!(cache::LinearSolve.LinearCache, alg::KLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:895
# solve!(cache::LinearSolve.LinearCache, alg::NormalBunchKaufmanFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:1172
# solve!(cache::LinearSolve.LinearCache, alg::Nothing, args...; assump, kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/default.jl:298
# solve!(cache::LinearSolve.LinearCache, alg::AppleAccelerateLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/appleaccelerate.jl:237
# solve!(cache::LinearSolve.LinearCache, alg::MKLLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/mkl.jl:213
# solve!(cache::LinearSolve.LinearCache, alg::FastLUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:1237
# solve!(cache::LinearSolve.LinearCache, alg::RFLUFactorization{P, T}; kwargs...) where {P, T} @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:1031
# solve!(cache::LinearSolve.LinearCache, alg::KrylovJL; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/iterative_wrappers.jl:227
# solve!(cache::LinearSolve.LinearCache, alg::SimpleGMRES; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/simplegmres.jl:148
# solve!(cache::LinearSolve.LinearCache, alg::NormalCholeskyFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:1118
# solve!(cache::LinearSolve.LinearCache, alg::CHOLMODFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:967
# solve!(cache::LinearSolve.LinearCache, alg::FastQRFactorization{P}; kwargs...) where P @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:1290
# solve!(cache::LinearSolve.LinearCache, alg::LUFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/factorization.jl:79
# solve!(cache::LinearSolve.LinearCache, alg::LinearSolve.AbstractFactorization; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/LinearSolve.jl:151
# solve!(cache::LinearSolve.LinearCache, alg::LinearSolveFunction, args...; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/solve_function.jl:6
# solve!(cache::LinearSolve.LinearCache, args...; kwargs...) @ LinearSolve ~/.julia/packages/LinearSolve/0Q5jw/src/common.jl:273
# solve!(P::Roots.ZeroProblemIterator{𝑴, 𝑵, 𝑭, 𝑺, 𝑶, 𝑳}; verbose) where {𝑴<:Bisection, 𝑵, 𝑭, 𝑺, 𝑶<:ExactOptions, 𝑳} @ Roots ~/.julia/packages/Roots/KNVCY/src/Bracketing/bisection.jl:172
# solve!(𝐙::Roots.ZeroProblemIterator{𝐌, 𝐍}; verbose) where {𝐌, 𝐍<:AbstractBracketingMethod} @ Roots ~/.julia/packages/Roots/KNVCY/src/hybrid.jl:30
# solve!(P::Roots.ZeroProblemIterator; verbose) @ Roots ~/.julia/packages/Roots/KNVCY/src/find_zero.jl:443
# solve!(integrator::Sundials.AbstractSundialsIntegrator; early_free, calculate_error) @ Sundials ~/.julia/packages/Sundials/KMu6U/src/common_interface/solve.jl:1402
# solve!(integrator::DelayDiffEq.DDEIntegrator) @ DelayDiffEq ~/.julia/packages/DelayDiffEq/xs5DA/src/solve.jl:545
# solve!(cache::Union{BoundaryValueDiffEq.FIRKCacheNested, BoundaryValueDiffEq.MIRKCache}) @ BoundaryValueDiffEq ~/.julia/packages/BoundaryValueDiffEq/YN0of/src/solve/mirk.jl:146

@doc (@doc DifferentialEquations.solve!)
solve!(args...;kwargs...) = DifferentialEquations.solve!(args...;kwargs...)
@doc (@doc Krylov.solve!)
solve!(solver::KrylovSolver, args...; kwargs...) = Krylov.solve!(solver, args...; kwargs...)

## :spectrogram
# Showing duplicate methods for spectrogram in packages Module[Flux, DSP]
# Methods for spectrogram in package NNlib
# spectrogram(waveform; pad, n_fft, hop_length, window, center, power, normalized, window_normalized) @ NNlib ~/.julia/packages/NNlib/CkJqS/src/audio/spectrogram.jl:28
# Methods for spectrogram in package DSP.Periodograms
# spectrogram(s::AbstractVector{T}, n::Int64, noverlap::Int64; onesided, nfft, fs, window) where T @ DSP.Periodograms ~/.julia/packages/DSP/eKP6r/src/periodograms.jl:420
# spectrogram(s::AbstractVector{T}, n::Int64; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/eKP6r/src/periodograms.jl:420
# spectrogram(s::AbstractVector{T}; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/eKP6r/src/periodograms.jl:420
spectrogram = DSP.spectrogram # Method for spectrogram in package DSP

## :statistics
# Showing duplicate methods for statistics in packages Module[Krylov, DelaunayTriangulation]
# Methods for statistics in package Krylov
# statistics(solver::TrimrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::QmrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::BilqSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CgLanczosShiftSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CgneSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::DqgmresSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CgsSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::FomSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::BicgstabSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::MinaresSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::FgmresSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::TrilqrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::SymmlqSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::LsqrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CraigSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::MinresQlpSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::BlockGmresSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/block_krylov_solvers.jl:76
# statistics(solver::LslqSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CarSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CrmrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CrlsSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CglsSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CgSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::GmresSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CraigmrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::BilqrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::GpmrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::DiomSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::UsymqrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::MinresSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::UsymlqSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::LsmrSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::CgLanczosSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::TricgSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# statistics(solver::LnlqSolver) @ Krylov ~/.julia/packages/Krylov/fwLT6/src/krylov_solvers.jl:1930
# Methods for statistics in package DelaunayTriangulation
# statistics(tri::Triangulation) @ DelaunayTriangulation ~/.julia/packages/DelaunayTriangulation/JBYjR/src/data_structures/statistics/triangulation_statistics.jl:84
@doc (@doc Krylov.statistics)
statistics(solver::KrylovSolver) = Krylov.statistics(solver)
@doc (@doc DelaunayTriangulation.statistics)
statistics(tri::DelaunayTriangulation.Triangulation) = DelaunayTriangulation.statistics(tri)
export statistics
push!(overrides, :statistics)

## :stft
# Showing duplicate methods for stft in packages Module[Flux, DSP]
# Methods for stft in package NNlib
# stft(x; n_fft, hop_length, window, center, normalized) @ NNlibFFTWExt ~/.julia/packages/NNlib/CkJqS/ext/NNlibFFTWExt/stft.jl:1
# Methods for stft in package DSP.Periodograms
# stft(s::AbstractVector{T}, n::Int64, noverlap::Int64, psdonly::Union{Nothing, DSP.Periodograms.PSDOnly}; onesided, nfft, fs, window) where T @ DSP.Periodograms ~/.julia/packages/DSP/eKP6r/src/periodograms.jl:443
# stft(s::AbstractVector{T}, n::Int64, noverlap::Int64; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/eKP6r/src/periodograms.jl:443
# stft(s::AbstractVector{T}, n::Int64; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/eKP6r/src/periodograms.jl:443
# stft(s::AbstractVector{T}; ...) where T @ DSP.Periodograms ~/.julia/packages/DSP/eKP6r/src/periodograms.jl:443
stft = DSP.stft # Method for stft in package DSP

## :top
# Showing duplicate methods for top in packages Module[DataStructures, CairoMakie]
# Methods for top in package DataStructures
# top(args...; kwargs...) @ DataStructures deprecated.jl:113
# Methods for top in package Makie
# top(rect::Rect2) @ Makie ~/.julia/packages/Makie/YkotL/src/makielayout/geometrybasics_extension.jl:5
top = Makie.top 
export top
push!(overrides, :top)

## :transform
# Showing duplicate methods for transform in packages Module[MultivariateStats, DataFrames]
# Methods for transform in package MultivariateStats
# transform(f::Whitening, x::AbstractVecOrMat{<:Real}) @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/whiten.jl:87
# transform(f, x) @ MultivariateStats deprecated.jl:103
# transform(f::MDS) @ MultivariateStats deprecated.jl:103
# Methods for transform in package DataFrames
# transform(gd::GroupedDataFrame, args::Union{Regex, AbstractString, Function, Signed, Symbol, Unsigned, Pair, Type, All, Between, Cols, InvertedIndex, AbstractVecOrMat}...; copycols, keepkeys, ungroup, renamecols, threads) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/groupeddataframe/splitapplycombine.jl:912
# transform(f::Union{Function, Type}, gd::GroupedDataFrame; copycols, keepkeys, ungroup, renamecols, threads) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/groupeddataframe/splitapplycombine.jl:902
# transform(arg::Union{Function, Type}, df::AbstractDataFrame; renamecols, threads) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/abstractdataframe/selection.jl:1388
# transform(df::AbstractDataFrame, args...; copycols, renamecols, threads) @ DataFrames ~/.julia/packages/DataFrames/kcA9R/src/abstractdataframe/selection.jl:1383
transform = DataFrames.transform  # the stuff in MultivariateStats seems deprecated
export transform
push!(overrides, :transform)

## :trim
# Showing duplicate methods for trim in packages Module[StatsBase, BenchmarkTools]
# Methods for trim in package StatsBase
# trim(x::AbstractVector; prop, count) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/robust.jl:52
# Methods for trim in package BenchmarkTools
# trim(t::BenchmarkTools.Trial) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/QNsku/src/trials.jl:87
# trim(t::BenchmarkTools.Trial, percentage) @ BenchmarkTools ~/.julia/packages/BenchmarkTools/QNsku/src/trials.jl:87
@doc (@doc StatsBase.trim)
trim(x::AbstractVector; kwargs...) = StatsBase.trim(x; kwargs...)
@doc (@doc BenchmarkTools.trim)
trim(t::BenchmarkTools.Trial) = BenchmarkTools.trim(t)
trim(t::BenchmarkTools.Trial, percentage) = BenchmarkTools.trim(t, percentage)
export trim
push!(overrides, :trim)

## :trim!
# Showing duplicate methods for trim! in packages Module[StatsBase, CairoMakie]
# Methods for trim! in package StatsBase
# trim!(x::AbstractVector; prop, count) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/robust.jl:63
# Methods for trim! in package GridLayoutBase
# trim!(gl::GridLayout) @ GridLayoutBase ~/.julia/packages/GridLayoutBase/TSvez/src/gridlayout.jl:574
@doc (@doc StatsBase.trim!)
trim!(x::AbstractVector; kwargs...) = StatsBase.trim!(x; kwargs...)
@doc (@doc Makie.trim!)
trim!(gl::Makie.GridLayout) = Makie.trim!(gl)
export trim!
push!(overrides, :trim!)

## :update!
# Showing duplicate methods for update! in packages Module[DataStructures, ProgressMeter, TaylorSeries]
# Methods for update! in package DataStructures
# update!(pt::JumpProcesses.PriorityTable, pid, oldpriority, newpriority) @ JumpProcesses ~/.julia/packages/JumpProcesses/3mbsw/src/aggregators/prioritytable.jl:186
# update!(h::MutableBinaryHeap{T}, i::Int64, v) where T @ DataStructures ~/.julia/packages/DataStructures/95DJa/src/heaps/mutable_binary_heap.jl:255
# Methods for update! in package ProgressMeter
# update!(p::ProgressMeter.AbstractProgress, val, color; options...) @ ProgressMeter deprecated.jl:103
# update!(p::ProgressThresh, val; increment, options...) @ ProgressMeter ~/.julia/packages/ProgressMeter/kVZZH/src/ProgressMeter.jl:499
# update!(p::ProgressThresh; ...) @ ProgressMeter ~/.julia/packages/ProgressMeter/kVZZH/src/ProgressMeter.jl:499
# update!(p::Union{Progress, ProgressUnknown}, counter::Int64; options...) @ ProgressMeter ~/.julia/packages/ProgressMeter/kVZZH/src/ProgressMeter.jl:486
# update!(p::Union{Progress, ProgressUnknown}; ...) @ ProgressMeter ~/.julia/packages/ProgressMeter/kVZZH/src/ProgressMeter.jl:486
# Methods for update! in package TaylorSeries
# update!(a::Union{Taylor1, TaylorN}) @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/other_functions.jl:276
# update!(a::TaylorN{T}, vals::Vector{T}) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/other_functions.jl:267
# update!(a::TaylorN{T}, vals::Vector{S}) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/other_functions.jl:271
# update!(a::Taylor1{T}, x0::T) where T<:Number @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/other_functions.jl:257
# update!(a::Taylor1{T}, x0::S) where {T<:Number, S<:Number} @ TaylorSeries ~/.julia/packages/TaylorSeries/XsXwM/src/other_functions.jl:261
@doc (@doc DataStructures.update!) 
update!(h::DataStructures.MutableBinaryHeap, i, v) = DataStructures.update!(h, i, v)
update!(pt::JumpProcesses.PriorityTable, pid, oldpriority, newpriority) = JumpProcesses.update!(pt, pid, oldpriority, newpriority)
@doc (@doc ProgressMeter.update!) 
update!(p::ProgressMeter.AbstractProgress, val, color; options...) = ProgressMeter.update!(p, val, color; options...)
update!(p::Union{ProgressMeter.Progress,ProgressMeter.ProgressUnknown,ProgressMeter.ProgressThresh}, val; options...) = ProgressMeter.update!(p, val; options...)
update!(p::Union{ProgressMeter.Progress,ProgressMeter.ProgressUnknown,ProgressMeter.ProgressThresh}; options...) = ProgressMeter.update!(p; options...)
@doc (@doc TaylorSeries.update!)
update!(a::TaylorSeries.Taylor1, x0) = TaylorSeries.update!(a, x0)
update!(a::TaylorSeries.TaylorN, vals::Vector) = TaylorSeries.update!(a, vals)
export update!
push!(overrides, :update!)

## :volume
# Showing duplicate methods for volume in packages Module[GeometryBasics, CairoMakie]
# Methods for volume in package GeometryBasics
# volume(mesh::GeometryBasics.Mesh) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/meshes.jl:233
# volume(triangle::Triangle) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/meshes.jl:221
# volume(prim::HyperRectangle) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/rectangles.jl:189
# Methods for volume in package MakieCore
# volume() @ MakieCore ~/.julia/packages/MakieCore/NeQjl/src/recipes.jl:432
# volume(args...; kw...) @ MakieCore ~/.julia/packages/MakieCore/NeQjl/src/recipes.jl:447
@doc (@doc GeometryBasics.volume)
volume(mesh::GeometryBasics.Mesh) = GeometryBasics.volume(mesh)
volume(triangle::GeometryBasics.Triangle) = GeometryBasics.volume(triangle)
volume(prim::GeometryBasics.HyperRectangle) = GeometryBasics.volume(prim)
@doc (@doc Makie.volume)
volume() = Makie.volume()
volume(args...; kw...) = Makie.volume(args...; kw...)

## :weights
# Showing duplicate methods for weights in packages Module[StatsBase, Graphs]
# Methods for weights in package StatsAPI
# weights(f::LinearDiscriminant) @ MultivariateStats ~/.julia/packages/MultivariateStats/u1yuF/src/lda.jl:121
# weights(vs::AbstractVector{<:Real}) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/weights.jl:89
# weights(vs::AbstractArray{<:Real}) @ StatsBase ~/.julia/packages/StatsBase/ebrT3/src/weights.jl:88
# Methods for weights in package Graphs
# weights(g::SimpleWeightedDiGraph) @ SimpleWeightedGraphs ~/.julia/packages/SimpleWeightedGraphs/byp3k/src/simpleweighteddigraph.jl:138
# weights(g::SimpleWeightedGraph) @ SimpleWeightedGraphs ~/.julia/packages/SimpleWeightedGraphs/byp3k/src/simpleweightedgraph.jl:166
# weights(g::MetaGraphs.AbstractMetaGraph) @ MetaGraphs ~/.julia/packages/MetaGraphs/qq8oz/src/MetaGraphs.jl:236
# weights(g::AbstractGraph) @ Graphs ~/.julia/packages/Graphs/1ALGD/src/core.jl:431
@doc (@doc StatsBase.weights)
weights(vs::AbstractVector) = StatsBase.weights(vs)
weights(vs::AbstractArray) = StatsBase.weights(vs)
weights(f::MultivariateStats.LinearDiscriminant) = MultivariateStats.weights(f)
@doc (@doc Graphs.weights)
weights(g::Graphs.AbstractGraph) = Graphs.weights(g)
export weights
push!(overrides, :weights)

## :width
# Showing duplicate methods for width in packages Module[GeometryBasics, Measures, CairoMakie]
# Methods for width in package GeometryBasics
# width(prim::HyperRectangle) @ GeometryBasics ~/.julia/packages/GeometryBasics/ebXl0/src/primitives/rectangles.jl:186
# Methods for width in package Measures
# width(x::BoundingBox) @ Measures ~/.julia/packages/Measures/PKOxJ/src/boundingbox.jl:43
@doc (@doc GeometryBasics.width)
width(prim::HyperRectangle) = GeometryBasics.width(prim)
@doc (@doc Measures.width)
width(x::BoundingBox) = Measures.width(x)
export width 
push!(overrides, :width)

## :⊕
# Showing duplicate methods for ⊕ in packages Module[LinearMaps, DoubleFloats]
# Methods for ⊕ in package LinearMaps
# ⊕(a, b, c...) @ LinearMaps ~/.julia/packages/LinearMaps/GRDXH/src/kronecker.jl:420
# ⊕(k::Integer) @ LinearMaps ~/.julia/packages/LinearMaps/GRDXH/src/kronecker.jl:418
# Methods for ⊕ in package DoubleFloats
# ⊕(x::T, y::T) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/iU3tv/src/math/ops/arith.jl:47

@doc (@doc getfield(LinearMaps, :⊕))
⊕(k::Integer) = LinearMaps.⊕(k)
⊕(A,B,Cs...) = LinearMaps.⊕(A,B,Cs...)
@doc (@doc getfield(DoubleFloats, :⊕))
⊕(x::T, y::T) where T<:Union{Float16, Float32, Float64} = DoubleFloats.⊕(x, y)
export ⊕
push!(overrides, :⊕)


## :⊗
# Showing duplicate methods for ⊗ in packages Module[Images, LinearMaps, DoubleFloats]
# Methods for tensor in package TensorCore
# tensor(x::C, y::C) where C<:(AbstractGray) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/tLy1N/src/ColorVectorSpace.jl:332
# tensor(u::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}} where T, v::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}} where T) @ TensorCore ~/.julia/packages/TensorCore/77QBu/src/TensorCore.jl:96
# tensor(u::AbstractArray, v::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}} where T) @ TensorCore ~/.julia/packages/TensorCore/77QBu/src/TensorCore.jl:87
# tensor(u::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}} where T, v::AbstractArray) @ TensorCore ~/.julia/packages/TensorCore/77QBu/src/TensorCore.jl:93
# tensor(A::AbstractArray, B::AbstractArray) @ TensorCore ~/.julia/packages/TensorCore/77QBu/src/TensorCore.jl:83
# tensor(a::AbstractRGB, b::AbstractRGB) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/tLy1N/src/ColorVectorSpace.jl:421
# tensor(a::C, b::C) where C<:(Union{TransparentColor{C, T}, C} where {T, C<:Union{AbstractRGB{T}, AbstractGray{T}}}) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/tLy1N/src/ColorVectorSpace.jl:257
# tensor(a::Union{TransparentColor{C, T}, C} where {T, C<:Union{AbstractRGB{T}, AbstractGray{T}}}, b::Union{TransparentColor{C, T}, C} where {T, C<:Union{AbstractRGB{T}, AbstractGray{T}}}) @ ColorVectorSpace ~/.julia/packages/ColorVectorSpace/tLy1N/src/ColorVectorSpace.jl:264
# Methods for ⊗ in package LinearMaps
# ⊗(A, B, Cs...) @ LinearMaps ~/.julia/packages/LinearMaps/GRDXH/src/kronecker.jl:136
# ⊗(k::Integer) @ LinearMaps ~/.julia/packages/LinearMaps/GRDXH/src/kronecker.jl:134
# Methods for ⊗ in package DoubleFloats
# ⊗(x::T, y::T) where T<:Union{Float16, Float32, Float64} @ DoubleFloats ~/.julia/packages/DoubleFloats/iU3tv/src/math/ops/arith.jl:82

# This ignores images, which also exports it's function under tensor. 
@doc (@doc getfield(LinearMaps, :⊗))
⊗(k::Integer) = LinearMaps.⊗(k)
⊗(A,B,Cs...) = LinearMaps.⊗(A,B,Cs...)
@doc (@doc getfield(DoubleFloats, :⊗))
⊗(x::T, y::T) where T<:Union{Float16, Float32, Float64} = DoubleFloats.⊗(x, y)
export ⊗
push!(overrides, :⊗)
##-Unused overrides
#=
## :order
order = DataFrames.order 

=#
