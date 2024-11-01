
@testset "Code quality (Aqua.jl)" begin
  Aqua.test_all(TechnicalCompute;
    undefined_exports = (; broken=true), # too many of these right now... but we still want the report! 
    stale_deps = (; ignore=[:GLMakie]), # ignore GLMakie as a stale dependency since it isn't tested on CI 
    piracies = (; treat_as_own=[DoubleFloat]), # treat DoubleFloat as something we can work with 
  )
end

# @testset "Code linting (JET.jl)" begin
#   JET.test_package(TechnicalCompute; target_defined_modules = false)
# end  