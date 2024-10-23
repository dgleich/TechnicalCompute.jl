

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

@testset "JSON" begin
  @test json([2,3]) == "[2,3]"
  @test JSON.parse("{\"title\":\"Matrix\",\"values\":[2,3,4]}") == Dict("title" => "Matrix", "values" => [2,3,4])
end

@testset "ProgressMeter" begin 
  p = Progress(100)
  for i in 1:100
    update!(p, i)
  end
  @test true
end 

@testset "Makie" begin 
  @test begin; brain = load(assetpath("brain.stl")); mesh(brain); return true; end 
end 

@testset "Arpack" begin 
  A = sprand(StableRNG(1), 50,50,10/50)
  Avals,Avecs = eigen(Matrix(A))
  k = 5 
  vals,vecs = eigs(A, nev=k, which=:LM)
  @test abs.(vals) ≈ abs.(sort(Avals, by=abs,rev=true)[1:k])
end 

# This is exported by Meshes... 
@testset "DelaunayTriangulation" begin
  points = rand(StableRNG(1), 2, 100)
  tri = triangulate(points)
  @test true
end

@testset "Polynomials" begin 
  @test degree(Polynomial([1, 0, 3, 4])) == 3
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
