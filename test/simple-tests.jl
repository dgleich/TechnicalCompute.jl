

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


@testset "Combinatorics" begin
  @test String(nthperm(Vector{Char}("abc"), 2)) == "acb"
end

@testset "ProgressMeter" begin 
  p = Progress(100)
  for i in 1:100
    update!(p, i)
  end
  @test true
end 

