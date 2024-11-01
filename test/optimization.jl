

function _test_optimal_horizon(m) 
  @suppress_out begin 
    # # https://discourse.julialang.org/t/writing-a-simple-production-scheduling-optimal-control-problem-in-jump/17748
    T = 4 # 4 weeks horizon

    @expression(m, g[t=0:T], -t+5)

    @variable(m, 0 <= x[0:T] <= 0.8)
    @variable(m, 0 <= y[0:T] <= 0.9, start=0.0)

    @objective(m, Min, 2*sum(x) + 3*sum(y))

    @constraint(m, [t in 0:(T-1)], y[t+1] == y[t] + (x[t+1] - g[t+1]))

    optimize!(m)
  end 
  @test termination_status(m) != JuMP.MOI.TerminationStatusCode(1)
end 

function _test_nonlinear(model)
  @suppress_out begin 
    @variable(model, x, start = 0.0)
    @variable(model, y, start = 0.0)

    @NLobjective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)

    optimize!(model)

    @test termination_status(model) != JuMP.MOI.TerminationStatusCode(1)

    # adding a (linear) constraint
    @constraint(model, x + y == 10)
    optimize!(model)
  end 
  
  # only test that we get here...
  @test true 
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

  # Tulip
  _test_optimal_horizon(Model(Tulip.Optimizer))

  # Convex and SCS 
  @testset "Convex" begin 
    m = 5;  n = 4
    A = randn(StableRNG(1), m, n); 
    xtrue = rand(StableRNG(2), n) 
    b = A * xtrue + 0.1*rand(StableRNG(3), m)

    x = Convex.Variable(n)
    problem = minimize(sumsquares(A * x - b), [x >= 0])
    solve!(problem, SCS.Optimizer; silent = true)
    @test x.value ≈ A\b    
  end 
end