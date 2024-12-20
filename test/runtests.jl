println("Time to load TechnicalCompute")
@time using TechnicalCompute
using Test, Aqua, JET 
using Suppressor

# Add methods for optional tests and debugging
envargs = get(()->"", ENV, "JULIA_ACTIONS_RUNTEST_ARGS")
foreach(s->push!(ARGS,s), split(envargs,","))

# make sure datadeps always accepts to download datasets... 
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

test_dups = "test_dups" in ARGS 

if "debug" in ARGS
  ENV["JULIA_DEBUG"] = "TechnicalCompute,Main"
end

include("common.jl")

setup_workdir() 

if "aqua_first" in ARGS
  include("aqua-and-jet.jl")
end 

if test_dups 
  @testset "duplicate names and overrides" begin 
    include("test_names_and_methods.jl")
    test_names_in_packages_for_override(TechnicalCompute.packages)
  end
end 

@testset "simple tests" begin 
  include("simple-tests.jl")
end 

include("optimization.jl")

@testset "overrides" begin 
  include("override-tests.jl")
end 

cleanup_workdir() 

if !("aqua_first" in ARGS)
  include("aqua-and-jet.jl")
end

@testset "utility tests" begin 
  include("utility-tests.jl")
end

