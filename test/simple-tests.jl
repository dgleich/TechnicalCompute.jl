

@testset "SparseArrays" begin
  A = sprand(50,50,5/50)
  @test true
end

@testset "Random" begin
  Random.seed!(0)
  x = rand()
  Random.seed!(0)
  y = rand()
  @test x==y
end

@testset "LinearAlgebra" begin
  A = rand(50,50)
  @test begin; vals,vecs = eigen(A); return true; end 
end

@testset "DataStructures" begin
  q = Queue{Int}()
  enqueue!(q, 1)
  @test dequeue!(q) == 1
end

@testset "OrderedCollections" begin
  d = OrderedDict{Int,Int}()
  d[1] = 2
  @test d[1] == 2
end

@testset "GeometryBasics" begin
  p = Point2f(1,2)
  @test p[1] == 1.0 
end

@testset "Distributions" begin
  d = Normal(0,1)
  @test pdf(d, 0) ≈ 1/sqrt(2*pi)
end

@testset "StatsBase" begin
  x = [1,2,3,4,5]
  @test begin wsample(StableRNG(1), x, Weights([0.99, 0.0025, 0.0025, 0.0025, 0.0025])) == 1
  end
  @test begin; bins = [0,1,7]; obs = [0.5, 1.5, 1.5, 2.5]; fit(Histogram, obs, bins); return true; end 
end

@testset "KernelDensity" begin
  y = randn(10) 
  x = [0.01*y; 0.01.*y .+ 1] 
  kd = kde(x)
  @test pdf(kd, 0) ≈ pdf(kd, 1) rtol=1e-2
end

@testset "Statistics" begin
  @test isapprox(std(randn(StableRNG(100), 10000)), 1; atol=1e-2)
  x = [1,2,3,4,5]
  @test median(x) == 3
end

@testset "MultivariateStats" begin
  C = randn(10,4)
  PC = pcacov(C'*C, zeros(size(C,2)))
  @test PC.prinvars ≈ sort(eigvals(C'*C), rev=true)
  @test begin; pca = fit(PCA, rand(10,4)); return true; end 
end

@testset "Flux" begin 
  function run_test() 
    actual(x) = 4x + 2
    x_train, x_test = hcat(0:5...), hcat(6:10...)
    y_train, y_test = actual.(x_train), actual.(x_test)
    predict = Dense(1 => 1)
    loss(model, x, y) = mean(abs2.(model(x) .- y));
    opt = Descent()
    data = [(x_train, y_train)]
    Flux.train!(loss, predict, data, opt)
    for epoch in 1:200
      Flux.train!(loss, predict, data, opt)
    end
    @test predict(x_test) ≈ y_test rtol=1e-2
  end 
end 

@testset "MLDatasets" begin 
  dataset = Iris() 
  @test size(dataset.features)  == (150, 4)
end

@testset "ReinforcementLearning" begin 
  @test begin
    run(
           RandomPolicy(),
           CartPoleEnv(),
           StopAfterNSteps(1_000),
           TotalRewardPerEpisode()
       )
    return true
  end 
end

@testset "NMF" begin 
  @test begin 
    A = rand(50, 50)
    A .= abs.(A)
    W, H = NMF.randinit(A, 3)
    NMF.solve!(NMF.MultUpdate{Float64}(obj=:mse,maxiter=100), A, W, H);
    return true
  end 
end

@testset "RDatasets" begin 
  @test begin 
    iris = dataset("datasets", "iris")
    @test size(iris) == (150, 5)
    return true
  end 
end

@testset "OnlineStats" begin 
end

@testset "Clustering" begin 
  @test begin 
    x = rand(10, 2)
    kmeans(x, 2)
    return true
  end 
end

@testset "Distances" begin 
  @test begin 
    x = rand(10)
    y = rand(10)
    @test euclidean(x,y) ≈ sqrt(sum((x .- y).^2))
    return true
  end 
end

@testset "NearestNeighbors" begin 
  @test begin 
    x = rand(10, 2)
    knn = KDTree(x)
    @test sum(knn.data[1]) + sum(knn.data[2]) ≈ sum(x)
    return true
  end 
end

@testset "Images" begin 
  @test begin 
    img = rand(10, 10)
    img = Gray.(img)
    @test size(img) == (10, 10)
    return true
  end 
end

@testset "FFTW" begin 
  @test begin 
    n = 20
    x = rand(n)
    fft(x)
    @test real(sum(fft(fft(x))) / sum(x)) ≈ n
    return true
  end 
end

@testset "TestImages" begin 
  @test begin 
    img = testimage("cameraman")
    @test size(img) == (512, 512)
    return true
  end 
end

@testset "DataFrames" begin 
  @test begin 
    df = DataFrame(A = 1:3, B = ["a", "b", "c"])
    @test size(df) == (3, 2)
    df[!, :C] = [1.0, 2.0, 3.0]
    @test size(df) == (3, 3)
    return true
  end 
end

@testset "Tables" begin 
  T =  CSV.read(IOBuffer("a,b\n1,2\n3,4"), rowtable)
  @test T == [(a=1,b=2), (a=3,b=4)]
  @test columntable(T) == (a = [1,3], b = [2,4])
end 

@testset "JSON" begin
  @test json([2,3]) == "[2,3]"
  @test JSON.parse("{\"title\":\"Matrix\",\"values\":[2,3,4]}") == Dict("title" => "Matrix", "values" => [2,3,4])
end

@testset "CSV" begin
  @test CSV.read(IOBuffer("a,b\n1,2\n3,4"), DataFrame) == DataFrame(a=[1,3], b=[2,4])
  #@test CSV.read(IOBuffer("a,b\n1,2\n3,4"), NamedTuple) 
end

@testset "TOML" begin
  @test TOML.parse("a = 1") == Dict("a" => 1)
end

@testset "YAML" begin 
  @test YAML.load("a: 1") == Dict("a" => 1)
end

@testset "DelimitedFiles" begin 
  @test [5.0,6.0] == vec(readdlm(IOBuffer("5.0\n6.0"), ' ', Float64))
end

@testset "Serde" begin 
end

@testset "EzXML" begin 
end 

@testset "FileIO" begin 
  img = load(HTTP.URI("https://github.com/JuliaLang/julia-logo-graphics/raw/master/images/julia-logo-color.png"));
  @test size(img) == (200, 320)
end 

@testset "HTTP" begin 
  @test HTTP.get("https://www.google.com") |> x -> x.status == 200
end

@testset "JLD2" begin 
  D = Dict("hello" => "world", "foo" => :bar)
  save(_filename("example.jld2"), D)
  D2 = load(_filename("example.jld2"))
  @test D2 == D 
end 

@testset "HDF5" begin 
  D = OrderedDict("z"=>1, "a"=>2, "g/f"=>3, "g/b"=>4)
  save(_filename("example.h5"), D)
  D2 = load(_filename("example.h5"); dict=OrderedDict())
  @test D2 == D
end

@testset "MAT" begin 
  D = OrderedDict("z"=>1, "a"=>2, "x"=>[1.0,2.0])
  matwrite(_filename("example.mat"), D)
  D2 = matread(_filename("example.mat"))
  @test D2 == D
end

@testset "BSON" begin 
  D = OrderedDict("z"=>1, "a"=>2, "x"=>[1.0,2.0])
  save(_filename("example.bson"), D)
  D2 = load(_filename("example.bson"))
  @test D2 == D
end 

@testset "NIfTI" begin 
  @test begin
    brain = niread(Makie.assetpath("brain.nii.gz")).raw
    return true
  end
end

@testset "CodecBzip2" begin 
end

@testset "CodecLz4" begin 
end

@testset "CodecXz" begin 
end

@testset "CodecZLib" begin 
end

@testset "CodecZstd" begin 
end

@testset "ZipFile" begin 
end

@testset "TranscodingStreams" begin 
end

@testset "GraphIO" begin 
  @test begin 
    testdatadir = joinpath(pkgdir(GraphIO), "test", "testdata")
    g = loadgraph(joinpath(testdatadir, "kinship.net"), GraphIO.NET.NETFormat()) 
    return true
  end 
end

@testset "BenchmarkTools" begin 
  @btime sum(rand(1000)); 
end

@testset "StableRNG" begin 
  x = rand(StableRNG(1), 10)
  @test rand(StableRNG(1), 10) == x
end 

@testset "ProgressMeter" begin 
  p = Progress(100)
  for i in 1:100
    update!(p, i)
  end
  @test true
end 

@testset "Printf" begin 
  @test @sprintf("%d", 1) == "1"
end

@testset "Measures" begin 
  x = 1mm; y = 1cm; 
  @test x + y == 11mm
end

@testset "Unitful" begin 
  @test 1u"s" == Second(1)
  @test 1u"minute" == Minute(1)
  @test 1u"hr" == Hour(1)
  @test 1u"d" == Day(1)
  @test_broken 1u"yr" == Year(1) 
  @test 1u"kg" * 1u"m/s^2" == 1u"N"
  @test uconvert(u"°C", 212u"°F") == 100u"°C"
  @test mod(1u"hr" + 24u"minute", 35u"s") == 0u"s"
end 

@testset "Colors" begin
  @test colorant"white" == RGB(1.0, 1.0, 1.0)
end

@testset "ColorVectorSpace" begin
  @test RGB(1.0, 1.0, 1.0) + RGB(0.0, 0.0, 0.0) == RGB(1.0, 1.0, 1.0)
end

@testset "ColorSchemes" begin
  @test colorschemes[:Purples_5] == ColorSchemes.Purples_5 
  #@test ColorSchemes.viridis(0.5) == RGB(0.282623, 0.140926, 0.457517)
end

@testset "Dates" begin
  @test Dates.DateTime(2021, 1, 1) == Date(2021, 1, 1)
end

@testset "HypertextLiteral" begin
  s = "Me & You"
  @test string(@htl("<p>$s</p>")) == "<p>Me &amp; You</p>"
end

@testset "Transducers" begin
  @test 1:3 |> Map(x -> 2x) |> collect == [2,4,6]
end 

@testset "ThreadsX" begin 
  @test ThreadsX.sum(y for x in 1:10 if isodd(x) for y in 1:x^2) == sum(y for x in 1:10 if isodd(x) for y in 1:x^2)
end

@testset "IterTools" begin 
  @test collect(distinct([1,1,2,1,2,4,1,2,3,4])) == [1,2,4,3]
end

@testset "StaticArrays" begin 
  a = SVector(1,2,3)
  b = SVector(4,5,6)
  @test a + b == SVector(5,7,9)
end

@testset "IndirectArrays" begin 
  a = [1,2,3]
  b = [4,5,6]
  ia = IndirectArray(a, b)
  @test ia[1] == 4
end

@testset "OffsetArrays" begin 
  a = OffsetArray(1:10, -5:4)
  @test a[-5] == 1
  @test a[4] == 10
  
end

@testset "KahanSummation" begin
  @test sum_kbn([1.0, 1.0, 1.0]) == 3.0
end

@testset "FillArrays" begin 
  x = Ones(5)
  y = Zeros(5)
  @test 5x+y == 5*ones(5) 
end 

@testset "TiledIteration" begin 
  A = reshape(1:16, 4, 4)
  @test collect(TileIterator((1:4, 1:4), (2,2))) == [(1:2, 1:2) (1:2, 3:4); (3:4, 1:2) (3:4, 3:4)]
end 

@testset "AxisArrays" begin 
  M = reshape(1:60, 12, 5)
  A = AxisArray(M, .1:.1:1.2, [:a, :b, :c, :d, :e])
  @test A[ArrayAxis{:col}(2)] == M[:,2]
  @test A[ArrayAxis{:col}(:b)] == M[:,2]

  f = Figure()
  @test begin; a = Axis(f[1,1]); return true; end 
end

@testset "DSP" begin 
  @test shiftsignal([1,2,3,4], 2) == [0,0,1,2]
end

@testset "CairoMakie / Makie" begin 
  @test begin; brain = load(assetpath("brain.stl")); mesh(brain); return true; end 
end

@testset "Observables" begin 
  x = Observable(1)
  y = Observable(2)
  z = Observable(3)
  onany(x, y) do x, y
    z[] = x + y
  end
  y[] = 4
  @test z[] == 5
end

@testset "LaTeXStrings" begin 
  @test begin; x = L"\alpha"; return true; end 
end

@testset "Latexify" begin 
end 

@testset "Arpack" begin 
  A = sprand(StableRNG(1), 50,50,10/50)
  Avals,Avecs = eigen(Matrix(A))
  k = 5 
  vals,vecs = eigs(A, nev=k, which=:LM)
  @test abs.(vals) ≈ abs.(sort(Avals, by=abs,rev=true)[1:k])
end 

@testset "LinearMaps" begin 
  # for some reason this test fails on windows
  if !Sys.iswindows()
    B = LinearMap(cumsum, 10)-I
    @test B*ones(10) == 0:9
    B = LinearMap(cumsum, reverse∘cumsum∘reverse, 10)+I
    @test B'*ones(10) == 11:-1:2
  else
    @test_broken false 
    @test_broken false 
  end 
end 

@testset "Krylov" begin 
  A = SymTridiagonal(2*ones(100), -1*ones(99))
  b = ones(100)
  (x,stats) = cg(A,b) 
  @test x ≈ A\b 
end 

@testset "MatrixMarket" begin 
end

@testset "SuiteSparseMatrixCollection" begin 
  #@test begin; A = load("HB/1138_bus.mtx"); return true; end 
end

@testset "MeshIO" begin 
  @test begin 
    mesh = load(Makie.assetpath("cat.obj"))
    return true
  end 
end

# This is exported by Meshes... 
@testset "DelaunayTriangulation" begin
  points = rand(StableRNG(1), 2, 100)
  tri = triangulate(points)
  @test true
end

@testset "DoubleFloats" begin 
  @test Double64(1.0) + Double64(1.0) == Double64(2.0)
end

@testset "MultiFloats" begin 
  @test Float64x4(1.0) + Float64x4(eps(1.0)/2) - Float64x4(1.0) ≈ Float64x4(eps(1.0)/2)
end

@testset "Polynomials" begin 
  @test degree(Polynomial([1, 0, 3, 4])) == 3
end 

@testset "SpecialFunctions" begin 
  @test besselj(1, 0.5) ≈ 0.24226845767487388
end

@testset "Roots" begin 
  #@test find_zero(x -> x^2 - 2, 0, Roots.Newton()) ≈ sqrt(2)
  @test find_zero(sin, 3) ≈ π
end

@testset "TaylorSeries" begin 
  @test Taylor1([1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333])(1.5) ≈ exp(1.5) atol=1e-1
end

@testset "FastTransforms" begin 
  c = range(0, 1, length=8192)
  x = leg2cheb(c, normcheb=true)
  y = cheb2leg(x, normcheb=true)
  @test y ≈ c
end

@testset "Interpolations" begin 
  xs = 1:0.2:5
  A = log.(xs)
  interp_linear = linear_interpolation(xs, A)
  @test interp_linear(3) ≈ log(3)
  @test interp_linear(2) ≈ log(2)
end 

@testset "Graphs" begin 
  @test degree(path_graph(5)) == [1, 2, 2, 2, 1]
  @test degree(path_graph(5), 4:5) == [2, 1]
  @test degree(path_graph(5), 1) == 1
end 

@testset "SimpleWeightedGraphs" begin 
  g = SimpleWeightedGraph(3)
  add_edge!(g, 1, 2, 0.5)
  add_edge!(g, 2, 3, 0.8)
  add_edge!(g, 1, 3, 2.0)
  @test  get_weight(g, 1, 2) == 0.5
  @test enumerate_paths(dijkstra_shortest_paths(g, 1), 3) == [1,2,3]
  add_edge!(g, 1, 2, 1.6)
  @test  enumerate_paths(dijkstra_shortest_paths(g, 1), 3) == [1,3] 
end 

@testset "Metis" begin 
  # TODO, improve this test. 
  T = smallgraph(:tutte)
  p = Metis.partition(T, 3)
  @test extrema(p) == (1, 3)
end

@testset "Combinatorics" begin
  @test String(nthperm(Vector{Char}("abc"), 2)) == "acb"
end

## Optimization Methods tested in optimization.jl

@testset "OptimTestProblems" begin 
end 

@testset "Optim" begin 
end

@testset "NonlinearSolve" begin 
end

@testset "LsqFit" begin 
end

@testset "Tulip" begin 
end

@testset "ForwardDiff" begin 
  f(x::Vector) = sin(x[1]) + prod(x[2:end]);  # returns a scalar
  x = vcat(pi/4, 2:4)
  @test ForwardDiff.gradient(f, x) == [  0.7071067811865476
    12.0
    8.0
    6.0]
  @test ForwardDiff.hessian(f, x) ≈ [ -0.7071067811865476  0.0  0.0  0.0
  0.0       0.0  4.0  3.0
  0.0       4.0  0.0  2.0
  0.0       3.0  2.0  0.0]
end 

@testset "Symbolics" begin 
  @test begin 
    @variables x y 
    f = x^2 + y^2
    dfdx = (Differential(x))(f) 
    return simplify(expand_derivatives(dfdx) == 2x)
  end 
end

@testset "DifferentialEquations" begin 
  # https://docs.sciml.ai/DiffEqDocs/stable/examples/classical_physics/
  C₁ = 5.730

  #Setup
  u₀ = 1.0
  tspan = (0.0, 1.0)

  #Define the problem
  radioactivedecay(u, p, t) = -C₁ * u

  #Pass to solver
  prob = ODEProblem(radioactivedecay, u₀, tspan)
  sol = solve(prob, Tsit5())

  #Plot
  # p = plot(sol, linewidth = 2, 
  #     axis=(title = "Carbon-14 half-life", xlabel = "Time in thousands of years", ylabel = "Percentage left"),
  #     label = "Numerical Solution")
  # lines!(p.axis, sol.t, t -> exp(-C₁ * t), linewidth = 3, linestyle = :dash, label = "Analytical Solution")

  @test map(t -> exp(-C₁ * t), sol.t) ≈ sol.u rtol=1e-3 

  p = plot(sol, linewidth = 2, 
      axis=(title = "Carbon-14 half-life", xlabel = "Time in thousands of years", ylabel = "Percentage left"),
      label = "Numerical Solution")
end
