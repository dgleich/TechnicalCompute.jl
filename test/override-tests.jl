@testset "@variables" begin 
  @variables x y z
  @test typeof(x) == Symbolics.Num
end 

@testset "Axis" begin 
  M = reshape(1:60, 12, 5)
  A = AxisArray(M, .1:.1:1.2, [:a, :b, :c, :d, :e])
  @test A[ArrayAxis{:col}(2)] == M[:,2]
  @test A[ArrayAxis{:col}(:b)] == M[:,2]

  f = Figure()
  @test begin; a = Axis(f[1,1]); return true; end 
end 


@testset "BSpline" begin 
  a = rand(10)
  itp = interpolate(a, BSpline(Constant()))
  @test itp(5.4) == a[5]

  @test begin 
    curve_XI = [
        [
            [1, 2, 3], [EllipticalArc((2.0, 0.0), (-2.0, 0.0), (0.0, 0.0), 2, 1 / 2, 0.0)],
        ],
        [
            [BSpline([(0.0, 0.4), (1.0, 0.2), (0.0, 0.1), (-1.0, 0.2), (0.0, 0.4)])],
        ],
        [
            [4, 5, 6, 7, 4],
        ],
        [
            [BezierCurve(reverse([(-1.0, -3.0), (-1.0, -2.5), (0.0, -2.5), (0.0, -2.0)]))], [CatmullRomSpline(reverse([(0.0, -2.0), (1.0, -3.0), (0.0, -4.0), (-1.0, -3.0)]))],
        ],
        [
            [12, 11, 10, 12],
        ],
        [
            [CircularArc((1.1, -3.0), (1.1, -3.0), (0.0, -3.0), positive=false)],
        ],
    ]
    points_XI = [(-2.0, 0.0), (0.0, 0.0), (2.0, 0.0), (-2.0, -5.0), (2.0, -5.0), (2.0, -1 / 10), (-2.0, -1 / 10), (-1.0, -3.0), (0.0, -4.0), (0.0, -2.3), (-0.5, -3.5), (0.9, -3.0)]
    points_XI_extra = copy(points_XI)
    push!(points_XI_extra, (-1.0, -4.0), (-1.0, -2.0), (0.0, -4.1), (1.0, -4.0), (1.0, -2.0), (1.0, 0.4), (0.0, 0.49), (1.99, -2.0), (-1.99, -2.0), (-1.99, -3.71))
    tri = triangulate(points_XI_extra; boundary_nodes=curve_XI)
    return true
  end 

end 

@testset "Bisection" begin 
end 

@testset "Categorical" begin 
end 

@testset "ComplexVariable" begin 
end

@testset "EllipticalArc" begin 
end

@testset "Fill" begin 
end 

@testset "Filters" begin 
  zi_python = [ 0.99672078, -1.49409147,  1.28412268, -0.45244173,  0.07559489]

  b = [ 0.00327922,  0.01639608,  0.03279216,  0.03279216,  0.01639608,  0.00327922]
  a = [ 1.        , -2.47441617,  2.81100631, -1.70377224,  0.54443269, -0.07231567]

  @test ≈(zi_python, Filters.filt_stepstate(b, a), atol=1e-7)
end 

@testset "Fixed" begin 
  T = Fixed{Int32, 10}
  @test T(1) == 1
  @test T(-1.0) == -1

  @test begin 
    f = Figure()

    Axis(f[1, 1], title = "My column has size Fixed(400)")
    Axis(f[1, 2], title = "My column has size Auto()")

    colsize!(f.layout, 1, FixedSize(400))
    # colsize!(f.layout, 1, 400) would also work
    return true 
  end 
end

@testset "FunctionMap" begin 

end

@testset "Graph" begin 
  g = Graph(sparse(Tridiagonal(ones(4), zeros(5), ones(4))))
  @test collect(edges(g)) == [Edge(1, 2), Edge(2, 3), Edge(3, 4), Edge(4, 5)]
end 

@testset "Length" begin 
  # intentionally empty
end 

@testset "Line" begin 
  # GeometryBasics.Line(Point2f(1.0, 3.0), Point2f(1.0, 4.0))

end