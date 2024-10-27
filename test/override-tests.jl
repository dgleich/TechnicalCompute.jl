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
  F = FunctionMap{Int64,false}(cumsum, 2)
  @test Matrix(F) == [1 0 ; 1 1]
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

@testset "Mesh" begin
end 

@testset "Normal" begin 
  @test begin 
    x = Normal(0, 1)
    rand(x, 10)
    return true
  end 
end 

@testset "Partition" begin 
  @test 1:8 |> Partition(3) |> Map(copy) |> collect == [[1, 2, 3], [4, 5, 6]]
end

@testset "Variable" begin 
  # intentionally empty
  # removed the override 
end 

@testset "Vec" begin 
  @test Vec(Point2f(1.0,2.0)) == [1.0,2.0]
end 

@testset "Vec2" begin 
  @test Vec2(1.0,2.0) == [1.0,2.0]
  @test_throws MethodError Vec2(Point3f(1.0,2.0,4.0)) 
end

@testset "Vec3" begin 
  @test Vec3(1.0,2.0,4.0) == [1.0,2.0,4.0]
  @test_throws MethodError Vec3(Point2f(1.0,2.0)) 
end

@testset "Zeros" begin 
  @test Zeros(3) == [0,0,0]
end

@testset "attributes" begin 
  
end

@testset "center" begin 
  @test center(path_graph(5)) == [3]
  W = ones(5,5)
  W[1,2] = 5
  W[2,1] = 5 
  @test center(path_graph(5), W) == [2]
  @test center(rand(11,11)) == [6.0,6.0]
end 

@testset "centered" begin 
  A = reshape(collect(1:9), 3, 3)
  Ao = centered(A)
  @test Ao[-1,-1] == 1 
end 

@testset "complement" begin 
  @test begin 
    
    complement(DataStructures.IntSet([1,3,5]))
    return true 
  end 

  @test complement(RGB(0.1,0.1,0.1)) == RGB(0.9,0.9,0.9)
  @test complement(0.1) == 0.9 
  @test complement(RGBA(0.1,0.1,0.1,0.1)) == RGBA(0.9,0.9,0.9,0.1)

  g = path_graph(5)
  complement(g) 
  @test Edge(1,3) in edges(complement(g))

  g = path_digraph(5)
  complement(g) 
  @test Edge(1,3) in edges(complement(g))
end 

@testset "constant" begin 
  # test the jump overrides 
  model = Model()
  @variable(model, x) 
  quad = 2.0*x^2 + 3.0
  @test constant(quad) == 3.0 

  affine = 5*x + 2.0 
  @test constant(affine) == 2.0 


end 

@testset "conv" begin 
end 

@testset "crossentropy" begin 
end 

@testset "curvature" begin 
end 

@testset "degree" begin 
  x = Polynomial(5,6)
  @test degree(x) == 6 

  p,q = fromroots(Polynomial, [1,2,3]), fromroots(Polynomial, [2,3,4])
  pq = p // q
  @test degree(pq)==1 

  g = path_graph(5)
  @test degree(g) == [1,2,2,2,1]
  @test degree(g,2) == 2
end 

@testset "density" begin 
  @test density(path_graph(4)) == 0.5

  # test Makie
  p = density(rand(10))
  @test typeof(p)==Makie.FigureAxisPlot
end 

@testset "derivative" begin 
  p,q = fromroots(Polynomial, [1,2,3]), fromroots(Polynomial, [2,3,4])
  pq = p // q

  pp = derivative(p)
  pp2 = derivative(p, 2)
  @test pp2 == derivative(pp)
  @test derivative(derivative(pq)) == derivative(pq,2)

  t = Taylor1([1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333])
  @test derivative(t, 2) == derivative(derivative(t))
end 