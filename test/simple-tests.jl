

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
  vals,vecs = eigen(A)
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

@testset "Statistics" begin
  @test isapprox(std(randn(StableRNG(100), 10000)), 1; atol=1e-2)
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

@testset "Combinatorics" begin
  @test String(nthperm(Vector{Char}("abc"), 2)) == "acb"
end

@testset "Makie" begin 
  @test begin; brain = load(assetpath("brain.stl")); mesh(brain); return true; end 
end 

@testset "Graphs" begin 
  @test degree(path_graph(5)) == [1, 2, 2, 2, 1]
  @test degree(path_graph(5), 4:5) == [2, 1]
  @test degree(path_graph(5), 1) == 1
end 

@testset "MultivariateStats"  begin 
  @test begin; pca = fit(PCA, rand(10,4)); return true; end 

end 

@testset "Polynomials" begin 
  @test degree(Polynomial([1, 0, 3, 4])) == 3
end 

@testset "ProgressMeter" begin 
  p = Progress(100)
  for i in 1:100
    update!(p, i)
  end
  @test true
end 

@testset "StatsBase" begin 
  bins = [0,1,7]
  obs = [0.5, 1.5, 1.5, 2.5]
  @test begin; fit(Histogram, obs, bins); return true; end 
end 

@testset "Symbolics" begin 
  @test begin 
    @variables x y 
    f = x^2 + y^2
    dfdx = (Differential(x))(f) 
    return simplify(expand_derivatives(dfdx) == 2x)
  end 
end 