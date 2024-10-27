println("Time to load TechnicalCompute")
@time using TechnicalCompute
using Test

# Add methods for optional tests and debugging
envargs = get(()->"", ENV, "JULIA_ACTIONS_RUNTEST_ARGS")
foreach(s->push!(ARGS,s), split(envargs,","))

# make sure datadeps always accepts to download datasets... 
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

test_dups = "test_dups" in ARGS 

if "debug" in ARGS
  ENV["JULIA_DEBUG"] = "TechnicalCompute,Main"
end

if test_dups 
  @testset "duplicate names and overrides" begin 
    include("test_names_and_methods.jl")
    test_names_in_packages_for_override(TechnicalCompute.packages)
  end
end 

workdir = joinpath(tempdir(), "TechnicalCompute")
tempfiles = [] 
_filename(file) = joinpath(workdir, file)

@testset "simple tests" begin 
  include("simple-tests.jl")
end 

@testset "overrides" begin 
  include("override-tests.jl")
end 

include("optimization.jl")

try 
  if isdir(workdir)
    rm(workdir; force=true, recursive=true)
  end
catch 
end 