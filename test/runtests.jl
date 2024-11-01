println("Time to load TechnicalCompute")
@time using TechnicalCompute
using Test, Aqua, JET 

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

@testset "Code quality (Aqua.jl)" begin
  Aqua.test_all(TechnicalCompute;
    undefined_exports = (; broken=true), # too many of these right now... but we still want the report! 
    stale_deps = (; ignore=[:GLMakie]), # ignore GLMakie as a stale dependency since it isn't tested on CI 
    piraces = (; treat_as_own=[DoubleFloat]), # treat DoubleFloat as something we can work with 
  )
end

# @testset "Code linting (JET.jl)" begin
#   JET.test_package(TechnicalCompute; target_defined_modules = false)
# end  