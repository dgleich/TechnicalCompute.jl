
@testset "Code quality (Aqua.jl)" begin
  stale_deps_list = [:GLMakie]
  if Sys.isapple() && Sys.ARCH == :aarch64
    push!(stale_deps_list, :Clp)
  end
  Aqua.test_all(TechnicalCompute;
    undefined_exports = (; broken=true), # too many of these right now... but we still want the report! 
    stale_deps = (; ignore=stale_deps_list), # ignore GLMakie/Clp
    piracies = (; treat_as_own=[DoubleFloat, nnz, permute]), # treat DoubleFloat as something we can work with 
  )
end

# @testset "Code linting (JET.jl)" begin
#   JET.test_package(TechnicalCompute; target_defined_modules = false)
# end  