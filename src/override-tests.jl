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


