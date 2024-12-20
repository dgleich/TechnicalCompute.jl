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
  itp = interpolate(a, BSpline())
  @test itp(5.5) == 0.5*(a[5] + a[6])

  a = rand(10)
  itp = interpolate(a, BSpline(Constant()))
  @test itp(5.4) == a[5]

  itp = interpolate(a, BSpline(Constant()))
  @test itp(5) == a[5]

  begin 
    @test BSpline([(0.0, 0.4), (1.0, 0.2), (0.0, 0.1), (-1.0, 0.2), (0.0, 0.4)]) == DelaunayTriangulation.BSpline([(0.0, 0.4), (1.0, 0.2), (0.0, 0.1), (-1.0, 0.2), (0.0, 0.4)])
  end 

end 

@testset "Bisection" begin 
  myf(x) = (x-3)*(x-2)
  @test find_zero(myf, (1.0, 2.5), Bisection()) == 2.0

  myf2(x,p) = myf(x)
  p = NonlinearProblem(myf2, 1.5)
  @test solve(p, Bisection(false,false)).u ≈ 2.0
  #@test solve(p, Bisection(;exact_left=false,exact_right=false)).u ≈ 2.0
end 

@testset "EllipticalArc" begin  
  begin 
    @test EllipticalArc(Point(0, 0), 1, 1, 0, 0, 2pi) == Makie.EllipticalArc(Point(0, 0), 1, 1, 0, 0, 2pi)
  end 
  begin 
    @test  EllipticalArc(0, 0, 1, 1, 0, 0, 2pi) == Makie.EllipticalArc(0, 0, 1, 1, 0, 0, 2pi)
    return true
  end 
  begin 
    @test EllipticalArc(0, 1, 0, 1, 1, 1, pi/2, false, true) == Makie.EllipticalArc(0, 1, 0, 1, 1, 1, pi/2, false, true)
    return true
  end 
  begin 
    @test EllipticalArc((2.0, 0.0), (-2.0, 0.0), (0.0, 0.0), 2, 1 / 2, 0.0) == DelaunayTriangulation.EllipticalArc((2.0, 0.0), (-2.0, 0.0), (0.0, 0.0), 2, 1 / 2, 0.0)  
    return true
  end 
end

@testset "Fill" begin 
  @test Fill(5, (2,3)) == 5*ones(2,3)
  @test Fill(5, 2, 3) == 5*ones(2,3)
  f = fill(0, (9,9))
  f[5,5] = 1
  w = centered([1 2 3; 4 5 6; 7 8 9])
  correlation = imfilter(f, w, Fill(0, w))
  @test correlation[5,5] == 5

  A = reshape(1:36, 6, 6)
  @test Fill(5, (2,3)) == 5*ones(2,3)
  B = padarray(A, Fill(0, (2,2), (2,2)))
  @test B[1,1] == 1
  C = padarray(A, Fill(0, [2,2], [2,2]))
  @test B == C
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

@testset "Flat" begin 
  # @test begin 
  #   BFGS(; alphaguess = LineSearches.InitialStatic(),
  #      linesearch = LineSearches.HagerZhang(),
  #      initial_invH = nothing,
  #      initial_stepnorm = nothing,
  #      manifold = Flat())
  #   return true
  # end 

  itp = interpolate(1:7, BSpline(Linear()))
  etp = extrapolate(itp, Flat())

  @test etp(7.5) == 7
end 

@testset "FunctionMap" begin 
  F = FunctionMap{Int64,false}(cumsum, 2)
  @test Matrix(F) == [1 0 ; 1 1]
end

@testset "Graph" begin 
  g = Graph(sparse(Tridiagonal(ones(4), zeros(5), ones(4))))
  @test collect(edges(g)) == [Edge(1, 2), Edge(2, 3), Edge(3, 4), Edge(4, 5)]
end 

@testset "GroupBy" begin 
  begin 
    x = rand(Bool, 10^2)
    y = x .+ randn(10^2)
    @test fit!(GroupBy(Bool, Series(Mean(), Extrema())), zip(x,y)) == fit!(OnlineStats.GroupBy(Bool, Series(Mean(), Extrema())), zip(x,y))
    return true
  end
  
  begin 
    @test isequal([1,2,3,4] |> GroupBy(iseven, Map(last)'(+)) |> foldxl(right), 
          [1,2,3,4] |> Transducers.GroupBy(iseven, Map(last)'(+)) |> foldxl(right))
  end 


end 

@testset "Length" begin 
  @test Length(:mm, 5) == 5mm
  @test Length(SVector(5,6)) == Length(2)
  @test Length([5,6]) == Length([1,2])
  @test Length(StaticArrays.Args((5,6))) == Length(2)
  @test Length(Size(SMatrix{2,2}([1 2;3 4 ]))) == Length(4)
end 

@testset "Line" begin 
  # GeometryBasics.Line(Point2f(1.0, 3.0), Point2f(1.0, 4.0))
end

# @testset "Mesh" begin
# end 

@testset "Normal" begin 
  @test begin 
    x = Normal(0, 1)
    y = Distributions.Normal(0,1)
    @test x==y
    rand(x, 10)
    return true
  end 
end 

@testset "Partition" begin 
  @test 1:8 |> Partition(3) |> Map(copy) |> collect == [[1, 2, 3], [4, 5, 6]]
end

@testset "Sum" begin 
  @test typeof(Sum([1,2,3])) <: ITensors.LazyApply.Applied

  o = Series(Sum())

  # Update with single data point
  fit!(o, 1.0)
  fit!(o, 2.0)
  @test value(o)[1] == 3.0

  o = Series(Sum(Int))
  fit!(o, 1.0)
  fit!(o, 2.0)
  @test value(o)[1] == 3.0
end 

@testset "Trace" begin 
  o = Trace(Mean())
  @test o == OnlineStats.Trace(OnlineStats.Mean())
  fit!(o, 1:100)
  @test  OnlineStats.snapshots(o)[end].μ == 50.5

  o = Trace(Mean(), 2)
  @test o == OnlineStats.Trace(OnlineStats.Mean(), 2)
  fit!(o, 1:10)
  @test  length(OnlineStats.snapshots(o)) == 3

  begin 
    o = Trace([(1,1) => Mean()], 3, 50)
    @test o == OnlineStats.Trace([(1,1) => OnlineStats.Mean()], 3, 50)
    return true
  end 

  begin 
    x = rand(5) 
    @test Trace(x) == ReinforcementLearning.Trace(x)
    return true
  end
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
  @test begin 
    doc = parsexml("""<genus name="Homo">
            <species name="sapiens">Human</species>
        </genus>""")
    n = collect(eachelement(root(doc)))[1]
    @test attributes(n) == EzXML.attributes(n) 
    return true 
  end
  @test begin 
    p = lines(rand(10)).plot
    @test attributes(p) == Makie.attributes(p) 
    return true 
  end 
  @test begin 
    h5open(_filename("test.h5"), "w") do file
        g = create_group(file, "mygroup") # create a group
        g["dset1"] = 3.2                  # create a scalar dataset inside the group
        attributes(g)["Description"] = "This group contains only a single dataset" # an attribute
        @test attributes(g) == HDF5.attributes(g) 
        return true 
    end
  end 
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

@testset "contract" begin 
  g = complete_graph(10)
  v = ones(Float64, ne(g))
  z = zeros(Float64, nv(g))
  n10 = Nonbacktracking(g)
  Graphs.contract!(z, n10, v)
  zprime = contract(n10, v)
  @test z == zprime
  
  @test begin 
    M = make_itensor_mps()
    contract(M)
    return true
  end
end 

@testset "conv" begin 
end 

@testset "crossentropy" begin 
end 

@testset "curvature" begin 
  @test begin 
    x = Convex.Variable(5)
    p = curvature(x.^2)
    return true
  end
  @test begin 
    a  = EllipticalArc((2.0, 0.0), (-2.0, 0.0), (0.0, 0.0), 2, 1 / 2, 0.0)
    curvature(a, 0.5)
    return true
  end 

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

  @test derivative(pq) == Polynomials.derivative(pq) 

  pp = derivative(p)
  pp2 = derivative(p, 2)
  @test pp2 == derivative(pp)
  @test derivative(derivative(pq)) == derivative(pq,2)

  t = Taylor1([1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333])
  @test derivative(t, 2) == derivative(derivative(t))
end 

@testset "differentiate" begin 
  t = Taylor1([1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333])
  @test differentiate(t, 2) == differentiate(differentiate(t))

  # test code below is from DelaunayTriangulation
  #=
  MIT License

  Copyright (c) 2024 Daniel VandenHeuvel <danj.vandenheuvel@gmail.com>

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  =#
  ⪧(a::Vector{Tuple}, b::Vector{Tuple}; kwargs...) = ⪧(collect.(a), collect.(b); kwargs...)
  ⪧(a::Vector{<:Union{Vector, Number}}, b::Vector; kwargs...) = isapprox(a, b; kwargs...)
  ⪧(a, b; kwargs...) = isapprox(collect(collect.(a)), collect(collect.(b)); kwargs...)
  p = rand(2) |> Tuple
  q = rand(2) |> Tuple
  L = LineSegment(p, q)
  for t in LinRange(0, 1, 100)
    der1 = differentiate(L, t)
    h = 1.0e-8
    der2 = (L(t + h) .- L(t - h)) ./ (2h)
    @test der1 ⪧ der2 rtol = 1.0e-5 atol = 1.0e-5
  end
end 

# @testset "dim" begin   
# end 

# @testset "eigs" begin
# end 

@testset "entropy" begin 

  @test begin 
    x = Convex.Variable(3)
    p = entropy(x) 
    return true 
  end 

  @test entropy(Chisq(3)) ≈ 2.05411995
  
  img = [   2  8   2  5  6
  7  9   2  7  3
  7  8   8  1  3
  3  1   3  9  4
  5  7  10  4  6]
  @test entropy(img) ≈   3.223465189601647

end 

@testset "evaluate" begin 
  @test evaluate(Cityblock(), (1.0,0), (0,1.0)) == 2.0
  begin 
    w = [1., 2., 3., 4., 5.]
    b = 2.5
    x = [4., 5., 2., 3., 1.]

    f = LinearDiscriminant(w, b)
    @test length(f) == 5
    @test dof(f) == 6
    @test coef(f) == (b, w)
    @test weights(f) == w
    @test evaluate(f, x) == 39.5
    @test evaluate(f, -reshape(x, 5, 1)) == [-34.5]
  end 
  begin 
    m = 5;  n = 4
    A = randn(StableRNG(1), m, n); 
    xtrue = rand(StableRNG(2), n) 
    b = A * xtrue + 0.1*rand(StableRNG(3), m)

    x = Convex.Variable(n)
    problem = minimize(sumsquares(A * x - b), [x >= 0])
    solve!(problem, SCS.Optimizer; silent = true)
    @test evaluate(x) ≈ A\b
  end
  @testset "TaylorSeries" begin 
    t = Taylor1([1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333])
    @test evaluate(t, 0.5) ≈ 1.6486979166666667
    @test evaluate([t], 0.5) ≈ [1.6486979166666667]

    δx, δy = set_variables("δx δy")
    xx = 1+Taylor1(δx, 5)
    yy = 1+Taylor1(δy, 5)
    @test evaluate(xx, 2, δy) == xx
    @test evaluate(xx, 1, δy) == yy

    @test evaluate([evaluate(xx)]) == [1.0]
    @test evaluate([evaluate(xx)], [0.5, 0.5]) == [1.5]

    @test begin 
      evaluate(xx)
      return true
    end


    @test isnothing(evaluate([t1N, t1N^2], 0.0, v))

  end 
end 

@testset "expand" begin 
  @test begin 
    @variables x y
    z = expand((x + y)^2)
    return true
  end 
  @test begin 
    # example from ITensorMPS.jl test set
    n = 6
    elt = Float64
    s = siteinds("S=1/2", n; conserve_qns=true)
    rng = StableRNG(1234)
    state = random_mps(rng, elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims=4)
    reference = random_mps(rng, elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims=2)
    state_expanded = expand(state, [reference]; alg="orthogonalize")
    return true
  end 
end 

# TODO 
@testset "fit" begin 
  @test fit(Cauchy{Float32}, collect(-4:4)) == Cauchy{Float64}(0.0, 2.0)
  
  @test begin 
    fit(Histogram, rand(100))
    return true 
  end 
end 

@testset "geomean" begin 
  @test geomean([1,2,3]) == StatsBase.geomean([1,2,3])  
  @test geomean(5) == StatsBase.geomean(5)

  x=Convex.Variable(3)
  y=Convex.Variable(3)
  @test isequal(geomean(x), Convex.geomean(x))
  @test isequal(geomean(x,y), Convex.geomean(x,y))

end 

@testset "get_weight" begin 
  t = Triangulation(rand(Point2f, 6))
  @test get_weight(t, 1) == DelaunayTriangulation.get_weight(t, 1)
  z = DelaunayTriangulation.ZeroWeight()
  @test get_weight(z, 1) == DelaunayTriangulation.get_weight(z, 1)

  A = sprand(5, 5, 0.5) |> A -> A + A'
  g = SimpleWeightedGraph(A)
  i,j = first.(findnz(A)[1:2])
  get_weight(g, i, j) == SimpleWeightedGraphs.get_weight(g, 1, 2)
end

@testset "gradient" begin 
  @test gradient(*, 2.0, 3.0, 5.0) == Flux.gradient(*, 2.0, 3.0, 5.0)
  @test gradient(x->x^2, 2.0) == Flux.gradient(x->x^2, 2.0)

  # Interpolations removed their gradient export... 
end

@testset "groupby" begin 
  df = DataFrame(a=repeat([1, 2, 3, 4], outer=[2]),
                        b=repeat([2, 1], outer=[4]),
                        c=1:8);
  @test groupby(df, :a) == DataFrames.groupby(df, :a)

  keyfunc = x->x[1]
  xs = ["face", "foo", "bar", "book", "baz", "zzz"]
  @test collect(groupby(keyfunc, xs)) == collect(IterTools.groupby(keyfunc, xs))
end 

@testset "hamming" begin
  @test hamming((0,1),(1,0)) == Distances.hamming((0,1),(1,0))  
  @test hamming(5) == DSP.hamming(5)
  @test hamming((5,5)) == DSP.hamming((5,5))
end

@testset "has_vertex" begin 
  g = path_graph(5)
  @test has_vertex(g, 1) == true 
  @test has_vertex(g, 6) == false 

  t = triangulate(rand(Point2f, 6))
  @test has_vertex(t, 0) == false 
  @test has_vertex(t, 1) == true

  g = get_graph(t) 
  @test has_vertex(t, 0) == false 
  @test has_vertex(t, 1) == true
end 

@testset "height" begin 
  r = HyperRectangle(1, 2, 3, 4)
  @test height(r) == GeometryBasics.height(r)
  c = Cylinder(Point3f(0,0,0), Point3f(0,0,1), 0.5f0)
  @test height(c) == GeometryBasics.height(c)

  bbox =  BoundingBox((0mm, 1mm), 1mm, 2mm)
  @test width(bbox) == Measures.width(bbox)
end 

@testset "hinge_loss" begin 
  x = Convex.Variable();
  @test isequal( hinge_loss(x), Convex.hinge_loss(x))
  y_true = [1, -1, 1, 1];
  y_pred = [0.1, 0.3, 1, 1.5];
  @test hinge_loss(y_true, y_pred) == Flux.hinge_loss(y_true, y_pred) 
end 

@testset "integrate" begin 
  @test integrate(Polynomial(1, 2)) == Polynomial(1, 3)/3
  @test integrate(Polynomial(1, 2), 0, 1) == Polynomials.integrate(Polynomial(1, 2), 0, 1)
  @test integrate(Polynomial(1, 2), 0.5) == Polynomial(1, 3)/3 + Polynomial([0.5])
  p,q = fromroots(Polynomial, [1,2,3]), fromroots(Polynomial, [2,3,4])
  pq = p // 5
  @test integrate(pq) == Polynomials.integrate(pq)

  t = Taylor1([1.0, 1.0])
  @test integrate(t) == TaylorSeries.integrate(t)
  @test integrate(t, 5) == TaylorSeries.integrate(t, 5)

  tn = TaylorN(1) + TaylorN(2)
  @test integrate(tn, 1) == TaylorSeries.integrate(tn, 1)
  @test integrate(tn, 1, 0.5) == TaylorSeries.integrate(tn, 1, 0.5)
end 

@testset "inverse" begin
  t = Taylor1([0.0, 1.0])
  @test inverse(t) == TaylorSeries.inverse(t)

  @test inverse(tanh) == Symbolics.inverse(tanh) 
  @test inverse(log) == Symbolics.inverse(log)
end

@testset "kldivergence" begin 
  @test kldivergence(LogNormal(0,1), LogNormal(0,1)) == 0.0
  p1 = [1 0; 0 1]
  p2 = Fill(0.5, 2, 2)
  @test kldivergence(p2, p1) ≈ log(2.0)
end

@testset "logsumexp" begin 
  @test logsumexp([1.0, 2.0, 3.0]) ≈ 3.4076059644443806
  #@test_throws  logsumexp(10000.0) == 10000.0 # can't figure out how to detect an error ... 
  begin 
    x = Convex.Variable(3)
    @test isequal(logsumexp(x), Convex.logsumexp(x))
  end
end 

@testset "maximize" begin 
  fmax(x) = -(1.0 - x[1])^2 - 100.0 * (x[2] - x[1]^2)^2
  function g!(G, x)
    G[1] = -1*(-2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1])
    G[2] = -200.0 * (x[2] - x[1]^2)
  end
  function H!(H, x)
    H[1, 1] = -2.0 + 800.0 * x[1]^2 + 400.0 * x[2]
    H[1, 2] = 400.0 * x[1]
    H[2, 1] = 400.0 * x[1]
    H[2, 2] = -200.0
  end 

  begin 
    x0 = [0.0, 0.0]
    soln = maximize(fmax, x0)
    @test Optim.maximizer(soln) ≈ [1.0, 1.0] atol=1e-3
    soln = maximize(fmax, x0, NelderMead())
    @test Optim.maximizer(soln) ≈ [1.0, 1.0] atol=1e-3
    soln = maximize(fmax, x0, NelderMead(), Optim.Options(g_tol = 1e-12))
    @test Optim.maximizer(soln) ≈ [1.0, 1.0] atol=1e-3

    soln = maximize(fmax, g!, x0, LBFGS())
    @test Optim.maximizer(soln) ≈ [1.0, 1.0] 
    soln = maximize(fmax, g!, x0, LBFGS(), Optim.Options(g_tol = 1e-12))
    @test Optim.maximizer(soln) ≈ [1.0, 1.0] 

    # These are broken... 
    #@test_throws UndefVarError soln = maximize(fmax, g!, H!, x0, Newton())
    @test Optim.maximizer(soln) ≈ [1.0, 1.0] 
    #@test_throws UndefVarError soln = maximize(fmax, g!, H!, x0, Newton(), Optim.Options(g_tol = 1e-12))
    @test Optim.maximizer(soln) ≈ [1.0, 1.0] 
  end 

  begin 
    f1(x) = -x^2
    soln = maximize(f1, -1.0, 1.0)
    @test Optim.maximizer(soln) ≈ 0.0 atol=1e-10

    soln = maximize(f1, -1.0, 1.0, Optim.Brent())
    @test Optim.maximizer(soln) ≈ 0.0 atol=1e-10
  end 

  begin 
    x = Convex.Variable(3)
    p = maximize(-norm(x), [x >= 0])
    solve!(p, SCS.Optimizer; silent = true)
    @test evaluate(x) ≈ [0.0, 0.0, 0.0]

    p = maximize(0, x >= 0, x <= 0)
    solve!(p, SCS.Optimizer; silent = true)
    @test evaluate(x) ≈ [0.0, 0.0, 0.0]

    p = maximize(0, [x >= 0, x <= 0])
    solve!(p, SCS.Optimizer; silent = true)
    @test evaluate(x) ≈ [0.0, 0.0, 0.0]
  end 
end 

@testset "metadata" begin 

  df = DataFrame(a=1, b=2);
  metadata!(df, "name", "example", style=:note);
  @test metadata(df, "name") == "example"
  @test metadata(df, "unknown", "MISSING") == "MISSING"

  @test metadata(Downloads.download("https://github.com/JuliaLang/julia-logo-graphics/raw/master/images/julia-logo-color.png")) == ((320, 200), RGBA{N0f8})
  @test metadata(File{format"PNG"}(Downloads.download("https://github.com/JuliaLang/julia-logo-graphics/raw/master/images/julia-logo-color.png"))) == ((320, 200), RGBA{N0f8})
end 

@testset "mode" begin 
  @test mode(Normal(0,1)) == 0.0
  @test mode([1,2,3,3,4,5]) == 3
  @test mode([1,2,3,3,3,4,4,5],4:5) == 4
  @test mode([1,1,2],weights([1,2,4])) == 2
  m = Model()
  @test mode(m) == AUTOMATIC
end 

@testset "nan" begin 
  @test isequal(nan(RGBf), RGBf(NaN, NaN, NaN))
  @test nan(Double64) == NaN
end 

@testset "nnz" begin 
  i = Index(2)
  O1 = onehot(i=>1)
  @test nnz(O1) == nnz(O1.tensor) # it's dense... 

  i = Index([QN(0)=>1, QN(1)=>2], "i");
  A = [1e-9 0.0 0.0;
  0.0 2.0 3.0;
  0.0 1e-10 4.0];
  T = ITensor(A, i', dag(i); tol = 1e-8);
  @test nnz(T) == 4 # why 4?? 

  @test nnz(NDTensors.EmptyStorage()) == 0 
  @test nnz(NDTensors.Diag(rand(10))) == 10

end 

@testset "onehot" begin 
  @test Vector(onehot(1, [1,2,3])) == [1,0,0]
  @test Vector(onehot(4, [1,2,3], 3)) == [0,0,1]

  i = Index(2)
  O1 = onehot(i=>1)
  @test Vector(O1) == [1,0]
  O1 = onehot([i=>1])
  @test Vector(O1) == [1,0]
  O1 = eltype(Vector(onehot(Int, i=>1))) == Int
  O1 = eltype(Vector(onehot(Int, [i=>1]))) == Int

  i = Index(2)
  j = Index(3) 
  @test array(onehot(Matrix{Float64}, i=>1,j=>3)) == [0 0 1; 0 0 0]
  
end 

@testset "params" begin 
  @test params(Normal(0,1)) == (0.0, 1.0)
  @test begin 
    b = @benchmarkable eigen(rand(10, 10));
    params(b) 
    tune!(b);
    r1 = run(b) 
    p0 = params(r1) 
    m1 = median(run(b))
    m2 = median(run(b))
    p1 = params(ratio(m1, m2)) 
    p2 = params(judge(m1, m2))
    return true
  end 
  @test begin 
    suite = BenchmarkGroup()
    params(suite)
    return true 
  end 
end 

@testset "permute" begin 
  A = sparse([1 2; 3 4])
  @test permute(A, [2,1], [1,2]) == sparse([3 4; 1 2])
end 

@testset "pop" begin 
  N = 5
  v = [Index(2, "Site,n=$n") for n in 1:N]
  # This throws an overflow error right now 
  #@test pop(v)[end] == v[4]
  @test_throws StackOverflowError pop(v)
  
  @test pop(@SVector[1, 2, 3]) == SVector(1,2)
  
  s = NDTensors.SmallVectors.SmallVector{10,Int}([1, 2, 3])
  s = pop(s)
  @test s == [1,2]
end 


@testset "popfirst" begin 
  N = 5
  v = [Index(2, "Site,n=$n") for n in 1:N]
  # This throws an overflow error right now 
  @test popfirst(v)[1] == v[2]
  
  @test popfirst(@SVector[1, 2, 3]) == SVector(2, 3)
  
  s = NDTensors.SmallVectors.SmallVector{10,Int}([1, 2, 3])
  s = popfirst(s)
  @test s == [2,3]
end 

@testset "probs" begin 
  @test probs(Multinomial(3, [0.5, 0.25, 0.25])) == [0.5, 0.25, 0.25]
  x = Series(CountMap())
  fit!(x, 1)
  fit!(x, 2)
  fit!(x, 1)
  fit!(x, 2)
  @test probs(x.stats[1]) == [0.5, 0.5]
  @test probs(x.stats[1], 1:3) == [0.5, 0.5, 0.0]
end 

@testset "properties" begin 
  @test collect(properties(1:10)) == Any[(:start,1), (:stop,10)]
  img = ImageMeta(fill(RGB(1,0,0), 3, 2), date=Date(2016, 7, 31), time="high noon")
  @test properties(img)[:date] == Date(2016, 7, 31)
end 

@testset "push" begin 
  @test push(@SVector[1, 2, 3], 4) == SVector(1, 2, 3, 4)
  N = 5
  v = [Index(2, "Site,n=$n") for n in 1:N]
  # This throws an overflow error right now 
  @test_throws StackOverflowError push(v, Index(2, "Site,n=$(N+1)"))
  #@test push(v, Index(2, "Site,n=$(N+1)")) == [Index(2, "Site,n=$n") for n in 1:N+1]

  s = NDTensors.SmallVectors.SmallVector{10,Int}([1, 2, 3])
  s = push(s, 4)
  @test s == [1,2,3,4]
end


@testset "radius" begin 
  g = path_graph(4)
  @test radius(g) == 2
  @test radius(g,weights(g)) == 2

  @test radius(Cylinder(Point3f(0,0,0), Point3f(0,0,1), 0.5f0)) == 0.5f0
  @test radius(Circle(Point2f(0,0), 1.5f0)) == 1.5f0
end 

@testset "reset!" begin 
  A = Accumulator{Int64,Int64}()
  push!(A, 1)
  push!(A, 1)
  push!(A, 2)
  @test reset!(A, 1) == 2

  # TODO, add more reset tests for ReinforcementLearning

  @test begin 
    rng = StableRNG(123)
    env = CartPoleEnv(; rng=rng)
    env′ = StateCachedEnv(env)
    reset!(env)
    reset!(env′)
    return true
  end
  @test begin 
    f = FIRFilter(rand(Float64, rand(16:128)), 1//1)
    reset!(f) 
  end 
end 

@testset "right" begin 

  @test right(Rect2((1.0,1.0),(2.0,2.0))) == 3.0

  @test foldl(right, Take(5), 1:10) == 5

end 

# @testset "rmsd" begin
# end

@testset "rotate!" begin 

  @test rotate!([1.0,1.0],[-1.0,1], 0, 1) == ([-1.0, 1.0], [-1.0,-1.0])

  @test begin 
    t1 = Transformation()
    rotate!(t1, 0.5)
    rotate!(Accum, t1, 0.5)
    return true 
  end 
  # Makie.rotate! for lights seems to be broken 
  # @test begin
  #   lights = Makie.AbstractLight[
  #       RectLight(RGBf(0.5, 0, 0), Point3f(-0.5, -1, 2), Vec3f(3, 0, 0), Vec3f(0, 3, 0)),
  #       RectLight(RGBf(0, 0.5, 0), Rect2f(-1, 1, 1, 3)),
  #       RectLight(RGBf(0, 0, 0.5), Point3f( 1,  0.5, 2), Vec3f(3, 0, 0), Vec3f(0, 3, 0)),
  #       RectLight(RGBf(0.5, 0.5, 0.5), Point3f( 1, -1, 2), Vec3f(3, 0, 0), Vec3f(0, 3, 0), Vec3f(-0.3, 0.3, -1)),
  #   ]
  #   # Test transformations
  #   rotate!(lights[2], Vec3f(3, 0, 0)) # translate to by default
  #   return true
  # end
end 

@testset "sample" begin 
  @test sample([1,1,1],1) == [1]
  @testset "ITensorMPS" begin 
    M = make_itensor_mps()
    M = normalize!(M)
    orthogonalize!(M,1)
    sample(M)
    sample(StableRNG(1234), M)

    N = 6
    sites = [Index(2, "Site,n=$n") for n in 1:N]
    seed = 623
    rng = StableRNG(seed)
    K = random_mps(rng, sites)
    L = outer(K', K)
    result = sample(rng, L)
    @test result ≈ [1, 1, 2, 1, 1, 1]
    sample(L)
    return true 
  end



end 

@testset "sample!" begin 
  rval = zeros(1)
  @test sample!([1,1,1], rval)   == [1.0]

  @test begin 
    M = make_itensor_mps()
    normalize!(M)
    sample!(M)
    sample!(StableRNG(1234), M)
    return true
  end 



end 

@testset "scale" begin 
  @test scale(Normal()) == 1.0

  xs = 1:0.2:5
  A = log.(xs)
  scaled_itp = scale(interpolate(A, BSpline(Linear())), xs)
end 

@testset "scale!" begin 

  @test begin 
    t1 = Transformation()
    scale!(t1, 0.5, 2, 3)
    return true 
  end 
  @test begin
    lights = Makie.AbstractLight[
        RectLight(RGBf(0.5, 0, 0), Point3f(-0.5, -1, 2), Vec3f(3, 0, 0), Vec3f(0, 3, 0)),
        RectLight(RGBf(0, 0.5, 0), Rect2f(-1, 1, 1, 3)),
        RectLight(RGBf(0, 0, 0.5), Point3f( 1,  0.5, 2), Vec3f(3, 0, 0), Vec3f(0, 3, 0)),
        RectLight(RGBf(0.5, 0.5, 0.5), Point3f( 1, -1, 2), Vec3f(3, 0, 0), Vec3f(0, 3, 0), Vec3f(-0.3, 0.3, -1)),
    ]
    # Test transformations
    translate!(lights[2], Vec3f(-1, 1, 2)) # translate to by default
    scale!(lights[2], 3, 1)
    scale!(lights[2], (3, 1))
    scale!(Accum, lights[2], (3, 1))
    return true
  end

  @test begin 
    i = Index([QN(0)=>1, QN(1)=>2], "i");
    A = [1e-9 0.0 0.0;
    0.0 2.0 3.0;
    0.0 1e-10 4.0];
    T = ITensor(A, i', dag(i); tol = 1e-8);
    scale!(T, 0.5)
    return true
  end 

  z = NDTensors.tensor(NDTensors.Diag(rand(elt, 5)), (5, 5))
  D = Diagonal(z) 
  scale!(z, 2.0)
  @test 2*D == Diagonal(z)


end 

@testset "shape" begin 
  d = Gamma()
  @test shape(d) == 1.0

  # JuMP case 
  m = Model()
  @variable(m, x)
  @variable(m, y)
  @variable(m, z)
  @variable(m, w)
  cref = @constraint(m, [x y; z w] in PSDCone())
  c = constraint_object(cref)
  @test shape(c) isa SquareMatrixShape

  cref = @constraint(m, 2x <= 10)
  c = constraint_object(cref)
  @test shape(c) isa ScalarShape
end 

@testset "solve!" begin 
  @test begin 
    A = rand(10,10) |> A -> A*A'
    b = rand(10)
    s = CgSolver(A,b)
    solve!(s, A, b)
    return true
  end 

  @test begin 
    m = 5;  n = 4
    A = randn(StableRNG(1), m, n); 
    xtrue = rand(StableRNG(2), n) 
    b = A * xtrue + 0.1*rand(StableRNG(3), m)

    x = Convex.Variable(n)
    problem = minimize(sumsquares(A * x - b), [x >= 0])
    solve!(problem, SCS.Optimizer; silent = true)
    return true 
  end 


  fx = ZeroProblem(sin, 3)
  problem = init(fx);
  @test solve!(problem) ≈ π

end 

# @testset "spectrogram" begin
# end

@testset "square" begin 
  @test square(Double64(0.5)) == Double64(0.25)
  @test square(Double64(0.5) + 1im) == -0.75 + 1.0im
  @test begin 
    x = Convex.Variable(5)
    f = square(sum(x))
    return true
  end 
end 

@testset "state" begin 
  @test begin 
    s = Index(2, "Site,S=1/2")
    sup = state(s,"Up")
    sup2 = state([s,s],1,"Up") # this just accesses the first element of the vector
    @test sup == sup2 
    return true 
  end
  
  ITensors.state(::StateName"phase", ::SiteType"Qubit"; θ::Real) = [cos(θ), sin(θ)]
  s = siteind("Qubit")
  @test state("phase", s; θ=π / 6) ≈ itensor([cos(π / 6), sin(π / 6)], s)

  @test state(StateName"phase", SiteType"Qubit"; θ=0.0) == [1.0, 0.0]

  @testset "state with old syntax" begin 
    function ITensors.state(::StateName{N}, ::SiteType"MyQudit2", s::Index) where {N}
      n = parse(Int, String(N))
      st = zeros(dim(s))
      st[n + 1] = 1.0
      return st
    end

    s = siteind("Qudit"; dim=5)
    v0 = state(s, "0")
    v1 = state(s, "1")
    v2 = state(s, "2")
    @test v0 == state("0", s)
    @test v1 == state("1", s)
    @test v2 == state("2", s)
  end 
  
  @test begin 
    rng = StableRNG(123)
    env = CartPoleEnv(; rng=rng)
    env′ = StateCachedEnv(env)
    s1 = state(env)
    s2 = state(env′)
    return true
  end 

end 

@testset "statistics" begin 
  solver = CgSolver(rand(10,10) |> A -> A*A', rand(10))
  @test statistics(solver).niter == 0 
  T = triangulate(rand(Point2f, 25))
  @test statistics(T).num_vertices >= 25 
end

# @testset "sfft" begin 
# end 

# @testset "top" begin 
# end

# @testset "transform" begin
# end

@testset "trim" begin 
  @test collect(trim([1,100,2,3,4];count=1)) == [2,3,4]
  @test begin 
    p =  @benchmark rand() samples=1 evals=5 seconds=0.01
    trim(p)
    trim(p,0.25)
    return true 
  end 
end 

@testset "trim!" begin 

  @test trim!([1,100,2,3,4];count=1) == [2,3,4]
  @test begin 
    f = Figure();
    trim!(f.layout)
    return true
  end 
end 

@testset "truncate!" begin 
  p = Polynomial([1, eps(1.0), 3, 4])
  truncate!(p)
  @test p == Polynomial([1, 0, 3, 4])

  @test begin 
    M = make_itensor_mps()
    truncate!(M)
    return true 
  end 

  @test begin 
    truncate!([1.0])
    return true 
  end 
end 

@testset "update!" begin 
  @test begin 
    h = MutableBinaryMinHeap{Int}()
    i = push!(h, 5)
    update!(h, i, 1.0)
    return true
  end  

  @testset "JumpProcesses" begin 
    DJ = DifferentialEquations.JumpProcesses

    minpriority = 2.0^exponent(1e-12)
    maxpriority = 2.0^exponent(1e12)
    priorities = [1e-13, 0.99 * minpriority, minpriority, 1.01e-4, 1e-4, 5.0, 0.0, 1e10]

    mingid = exponent(minpriority)   # = -40
    ptog = priority -> DJ.priortogid(priority, mingid)
    t = DJ.PriorityTable(ptog, priorities, minpriority, maxpriority)

    grpcnt = DJ.numgroups(t)
    push!(priorities, maxpriority * 0.99)
    DJ.insert!(t, length(priorities), priorities[end])

    push!(priorities, maxpriority * 0.99999)
    DJ.insert!(t, length(priorities), priorities[end])

    numsmall = length(t.groups[2].pids)
    push!(priorities, minpriority * 0.6)
    DJ.insert!(t, length(priorities), priorities[end])

    push!(priorities, maxpriority)
    DJ.insert!(t, length(priorities), priorities[end])

    # test updating
    update!(t, 5, priorities[5], 2 * priorities[5])   # group 29
    priorities[5] *= 2
    @test t.groups[29].numpids == 1
    @test t.groups[30].numpids == 1
  end
    
  @test begin 
    prog = ProgressUnknown(desc="Total length of characters read:")
    total_length_characters = 0
    for val in ["aaa" , "bb", "c", "d"]
      total_length_characters += length(val)
      update!(prog, total_length_characters)
      if val == "c"
        finish!(prog)
        break
      end
    end
    return true 
  end 
  @test begin 
    prog = ProgressUnknown(desc="Total length of characters read:")
    total_length_characters = 0
    for val in ["aaa" , "bb", "c", "d"]
      total_length_characters += length(val)
      update!(prog, total_length_characters)
      if val == "c"
        finish!(prog)
        break
      end
    end
    return true 
  end 
  @test begin 
    # from update! docstring
    # https://github.com/FluxML/Flux.jl/blob/c86580b34edd979dd57899defd39a39f20e84462/test/train.jl#L91
    m = Chain(Dense(2=>3, tanh), Dense(3=>1), only)
    x = rand(Float32, 2)
    y1 = m(x)  # before
    gs = Flux.Zygote.gradient(marg -> marg(x), m)
    s = Flux.setup(Adam(), m)
    update!(s, m, gs[1])
    return true 

  end 
  @test begin 
    t = Taylor1([1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333])
    update!(t, 0.5)
    return true
  end 
end 

@testset "value" begin 
  @testset "JuMP" begin 
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @objective(model, Min, (x - p)^2)
    optimize!(model)
    @test value(x) ≈ 1.0 rtol=1e-6
    @test value(x;result=1) ≈ 1.0 rtol=1e-6
  end 

  @testset "OnlineStats" begin 
    o = Series(Mean(), Variance())

    # Update with single data point
    fit!(o, 1.0)
    @test value(o)[1] == 1.0
  end 
end 

@testset "volume" begin 
  h = HyperRectangle(Vec(1.0, 2.0), Vec(3.0, 4.0))
  @test volume(h) == 3.0*4.0

  cube = Rect(Vec3f(-0.5), Vec3f(1))
  cube_faces = decompose(TriangleFace{Int}, faces(cube))
  cube_vertices = decompose(Point{3,Float32}, cube)
  mesh = Mesh(cube_vertices, cube_faces)
  @test volume(mesh) ≈ 1
  @test volume(cube) ≈ 1

  tri = Triangle(Point3f(0,0,0), Point3f(1,0,0), Point3f(0,1,0))
  @test volume(tri) == 0.0
  
  r = LinRange(-1, 1, 100)
  cube = [(x.^2 + y.^2 + z.^2) for x = r, y = r, z = r]
  cube_with_holes = cube .* (cube .> 1.4)
  f = volume(cube_with_holes, algorithm = :iso, isorange = 0.05, isovalue = 1.7); 
  return true 

  @test_throws "Not implemented" volume()
end 


@testset "weights" begin 
  @test begin
    a = rand(10)
    A = rand(5,5)
    w = weights(a)
    W = weights(A)
    return true
  end 

  A = rand(5,5) |> A -> A + A'
  g = SimpleWeightedGraph(A)
  @test weights(g) == A
end

@testset "width" begin 
  h = HyperRectangle(Vec(1.0, 2.0), Vec(3.0, 4.0))
  @test width(h) == 3.0 
  bbox =  BoundingBox((0mm, 1mm), 1mm, 2mm)
  @test width(bbox) == 1mm
end 

@testset "write_to_file" begin 
  @test begin 
    m = Model(Tulip.Optimizer)
    T = 4
    @expression(m, g[t=0:T], -t+5)

    @variable(m, 0 <= x[0:T] <= 0.8)
    @variable(m, 0 <= y[0:T] <= 0.9, start=0.0)

    @objective(m, Min, 2*sum(x) + 3*sum(y))

    @constraint(m, [t in 0:(T-1)], y[t+1] == y[t] + (x[t+1] - g[t+1]))
    write_to_file(m, _filename("test-jump.mps"))
    return true
  end
  @test begin 
    m = 5;  n = 4
    A = randn(StableRNG(1), m, n); 
    xtrue = rand(StableRNG(2), n) 
    b = A * xtrue + 0.1*rand(StableRNG(3), m)

    x = Convex.Variable(n)
    problem = minimize(sumsquares(A * x - b), [x >= 0])
    write_to_file(problem, _filename("test-convex.sdpa"))
  end 
end 

@testset "delta" begin 
  @test array(δ(Index(5, "x"))) == ones(5)
  @test array(δ(Index(5, "x"), Index(5, "y"))) == Matrix(1.0I,5,5)
end 

@testset "directsum" begin 

  a = rand(3,3)
  @test Matrix(LinearMap(a)^(⊕(2))) == kron(a,Matrix(1.0I,3,3)) + kron(Matrix(1.0I,3,3),a)

  a = rand(3,3)
  b = rand(4,4)
  @test Matrix(a ⊕ b) == kron(a,Matrix(1.0I,4,4)) + kron(Matrix(1.0I,3,3),b)

  @test begin 
    x = Index(1, "x")
    i1 = Index(2, "i1")
    j1 = Index(3, "j1")
    i2 = Index(4, "i2")
    j2 = Index(5, "j2")

    A1 = random_itensor(x, i1)
    A2 = random_itensor(x, i2)
    S, s = (A1 => i1) ⊕ (A2 => i2)
    return true
  end

  x = Index(2, "x")
  i1 = Index(3, "i1")
  j1 = Index(4, "j1")
  i2 = Index(5, "i2")
  j2 = Index(6, "j2")

  A1 = random_itensor(x, i1)
  A2 = random_itensor(x, i2)
  S, s = directsum(A1 => i1, A2 => i2)

  @test 5.0 ⊕ 5.0 == 10.0
end 

@testset "tensor" begin 
  a = rand(10)
  b = rand(10)
  @test vec(Matrix(b ⊗ a)) == vec(a*b')

  @test vec(Matrix(LinearMap(a)^(⊗(2)))) == vec(a*a')

  r = colorant"red"
  b = colorant"blue"
  @test typeof(r ⊗ b) <: RGBRGB

  @test 5.0 ⊗ 5.0 == 25.0
end 

@testset "hadamard" begin 
  i = Index(2, "i")
  A = random_itensor(i', i)
  B = random_itensor(i', i)
  C = A ⊙ B
  @test array(A) .* array(B) ≈ array(C)

  r = colorant"red"
  b = colorant"blue"
  @test r ⊙ b == colorant"black"

  A = rand(3,3,3)
  B = rand(3,3,3)
  @test A .* B ≈ A ⊙ B

end 

@testset "Text" begin 
  @test typeof(text(rand(Point2f), text="Hello, World!").plot) <: Makie.Text 
end 

@testset "tanpi" begin 
  x = DoubleFloat(rand()) 
  y = rand(Float64)
  @test tanpi(x) ≈ tan(pi*x)
  @test tanpi(y) ≈ tan(pi*y)
end 

@testset "axes" begin 
  @test axes([1,2,3])[1] == 1:3
end 
