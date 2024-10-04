# check for method duplicates in the libraries of Technical compute and 
# make sure we export a method from Technical compute with the same name

function _names_in_packages(pkgs)
  namesused = Dict{Symbol, Vector{Module}}()
  for pkg in pkgs
    for name in names(pkg)
      if haskey(namesused, name)
        push!(namesused[name], pkg)
      else
        namesused[name] = [pkg]
      end
    end
  end
  return namesused
end
function test_name_in_packages_for_equality(packages, name, overlaps)
  if name == :gamma
    filter!(p -> p !== Combinatorics, packages)
  end 
  handles = map(packages) do pkg
    getfield(pkg, name)
  end
  #overriden_names = TechnicalCompute.overrides
  overriden_names = Set{Symbol}()
  for i in eachindex(handles)
    for j in i+1:length(handles)
      if handles[i] !== handles[j]
        #println("Name $name is not equal in packages $(packages[i]) and $(packages[j]) and $(name in TechnicalCompute.overrides)")
        pkgpair = (packages[i], packages[j])
        pkgpair = repr(pkgpair[1]) < repr(pkgpair[2]) ? pkgpair : (pkgpair[2], pkgpair[1])
        if haskey(overlaps, pkgpair)
          overlaps[pkgpair] += 1
        else
          overlaps[pkgpair] = 1
        end
        if name in overriden_names
          # no error, we've handled it...
        else
          @test_broken name in overriden_names 
          println("Name $name is not equal in packages $(packages[i]) and $(packages[j]) and not in overrides")
        end
      end
    end
  end
end 
function test_names_in_packages_for_override(pkgs)
  namesused = _names_in_packages(map(pkg->eval(Meta.parse("$pkg")), pkgs))
  overlaps = Dict{Tuple{Module,Module},Int}()
  for (name, pkgs) in namesused
    if length(pkgs) > 1
      test_name_in_packages_for_equality(pkgs, name, overlaps)
    end
  end
  pairs = collect(keys(overlaps))
  sort!(pairs, by = x -> overlaps[x], rev=true)
  for pair in pairs
    println("$(pair[1]) and $(pair[2]) have $(overlaps[pair]) overlapping names")
  end
end

@testset "duplicate names and overrides" begin 

  test_names_in_packages_for_override(TechnicalCompute.packages)
end