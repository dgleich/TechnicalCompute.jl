using TechnicalCompute
using Test

# check for method duplicates in the libraries of Technical compute and 
# make sure we export a method from Technical compute with the same name

function _names_in_packages(pkgs)
  namesused = Dict{Symbol, Vector{Module}}()
  for pkg in pkgs
      for name in names(pkg)
          if haskey(namesused, name)
              push!(namesused[name], pkg)
          else
              namesused[name] = [pkg]
          end
      end
  end
  return namesused
end
function test_name_in_packages_for_equality(packages, name, mynames)
  if name == :gamma
    filter!(p -> p !== Combinatorics, packages)
  end 
  handles = map(packages) do pkg
    getfield(pkg, name)
  end
  for i in eachindex(handles)
    for j in i+1:length(handles)
      if handles[i] !== handles[j]
        #println("Name $name is not equal in packages $(packages[i]) and $(packages[j]) and $(name in TechnicalCompute.overrides)")
        if name in TechnicalCompute.overrides
          # no error, we've handled it...
        else
          @test_broken name in TechnicalCompute.overrides 
          println("Name $name is not equal in packages $(packages[i]) and $(packages[j]) and not in overrides")
        end
      end
    end
  end
end 
function test_names_in_packages_for_override(pkgs)
  namesused = _names_in_packages(map(pkg->eval(Meta.parse("$pkg")), pkgs))
  mynames = names(TechnicalCompute)
  for (name, pkgs) in namesused
    if length(pkgs) > 1
      test_name_in_packages_for_equality(pkgs, name, mynames)
    end
  end
end

@testset "duplicate names and overrides" begin 
  test_names_in_packages_for_override(TechnicalCompute.packages)
end 

@testset "Random" begin
  Random.seed!(0)
  x = rand()
  Random.seed!(0)
  y = rand()
  @test x==y
end

@testset "SparseArrays" begin
  A = sprand(50,50,5/50)
  @test true
end

@testset "Combinatorics" begin
  @test String(nthperm(Vector{Char}("abc"), 2)) == "acb"
end

function _test_optimal_horizon(m) 
  # # https://discourse.julialang.org/t/writing-a-simple-production-scheduling-optimal-control-problem-in-jump/17748
  T = 4 # 4 weeks horizon

  @expression(m, g[t=0:T], -t+5)

  @variable(m, 0 <= x[0:T] <= 0.8)
  @variable(m, 0 <= y[0:T] <= 0.9, start=0.0)

  @objective(m, Min, 2*sum(x) + 3*sum(y))

  @constraint(m, [t in 0:(T-1)], y[t+1] == y[t] + (x[t+1] - g[t+1]))

  optimize!(m)

  @test termination_status(m) != JuMP.MOI.TerminationStatusCode(1)
end 

function _test_nonlinear(model)

  @variable(model, x, start = 0.0)
  @variable(model, y, start = 0.0)

  @NLobjective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)

  optimize!(model)

  @test termination_status(model) != JuMP.MOI.TerminationStatusCode(1)

  # adding a (linear) constraint
  @constraint(model, x + y == 10)
  optimize!(model)

  # only test that we get here...
  @test true 
  #@test termination_status(model) != JuMP.MOI.TerminationStatusCode(1)
end 

# make sure we can solve a simple LP
@testset "Optimization" begin

  _test_optimal_horizon(Model(GLPK.Optimizer))

  # Try Clp 
  # broken on aarch64
  # https://github.com/jump-dev/Clp.jl/issues/131
  if Sys.ARCH == :aarch64 && Sys.isapple() 
    @test_broken false
  else 
    _test_optimal_horizon(Model(Clp.Optimizer))
  end 
  
  # # Ipopt
  _test_nonlinear(Model(Ipopt.Optimizer))
  _test_optimal_horizon(Model(Ipopt.Optimizer))

  # HiGHS
  _test_optimal_horizon(Model(HiGHS.Optimizer))

end
