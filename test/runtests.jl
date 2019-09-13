using TechnicalCompute
using Test

@testset "Random" begin
  Random.seed!(0)
  x = rand()
  Rand.seed!(0)
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

# make sure we can solve a simple LP
@testset "Optimization" begin

  # https://discourse.julialang.org/t/writing-a-simple-production-scheduling-optimal-control-problem-in-jump/17748
  m = Model(with_optimizer(GLPK.Optimizer))
  T = 4 # 4 weeks horizon

  @expression(m, g[t=0:T], -t+5)

  @variable(m, 0 <= x[0:T] <= 0.8)
  @variable(m, 0 <= y[0:T] <= 0.9, start=0.0)

  @objective(m, Min, 2*sum(x) + 3*sum(y))

  @constraint(m, [t in 0:(T-1)], y[t+1] == y[t] + (x[t+1] - g[t+1]))

  optimize!(m)

  @test termination_status(m) != JuMP.MathOptInterface.TerminationStatusCode(1)

  # Try Clp
  m = Model(with_optimizer(Clp.Optimizer))
  T = 4 # 4 weeks horizon

  @expression(m, g[t=0:T], -t+5)

  @variable(m, 0 <= x[0:T] <= 0.8)
  @variable(m, 0 <= y[0:T] <= 0.9, start=0.0)

  @objective(m, Min, 2*sum(x) + 3*sum(y))

  @constraint(m, [t in 0:(T-1)], y[t+1] == y[t] + (x[t+1] - g[t+1]))

  optimize!(m)

  @test termination_status(m) != JuMP.MathOptInterface.TerminationStatusCode(1)

  @test true # just make sure we get here
end
