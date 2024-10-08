

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