# check for method duplicates in the libraries of Technical compute and 
# make sure we export a method from Technical compute with the same name

function _names_in_packages(pkgs)
  namesused = Dict{Symbol, Vector{Module}}()
  for pkg in pkgs
    for name in names(pkg)
      if name == :gamma && pkg == Combinatorics
        continue # this one doesn't really exist! 
      end 
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
  overriden_names = TechnicalCompute.overrides
  #overriden_names = Set{Symbol}()
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
          println("Name $name is not equal in packages $(packages[i]) and $(packages[j]) and is in overrides")
          @test name in overriden_names
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

function _find_duplicate_methods(name, pkgs)
  handles = map(pkgs) do pkg
    getfield(pkg, name)
  end
  handles = unique(handles) # remove duplicates... 
  allmethods = methods.(handles)
  #filter!(ms -> !isempty(ms), allmethods) # we can't actaully ignore these... 
  if length(allmethods) > 1
    return (name, packages=pkgs, handles, methods=allmethods)
  else
    return nothing
  end
end 
#= 
allduplicates = map(collect(pairs(namesused))) do (name, pkgs)
    println("name is $name")
    _find_duplicate_methods(name, pkgs)
  end 
  # remove nothing entries
  filter!(x -> x !== nothing, allduplicates)
  # sort by name
  sort!(allduplicates, by = x -> x.name)

  for dup in allduplicates
    println(io)
    println(io, "Showing duplicate methods for $(dup.name) in packages $(dup.packages)")
    for h in unique(dup.handles) 
      println(io, "Methods for $(h) in package $(typeof(h).name.module)")
      for method in methods(h)
        println(io, method)
      end
    end
  end
  =#


function report_on_all_methods_for_all_duplicate_names(;packages=TechnicalCompute.packages, io=stdout)
  # This method is slightly complicated because we need to handle a number of special 
  # cases and then we want to report in sorted order.
  namesused = _names_in_packages(map(pkg->eval(Meta.parse("$pkg")), packages))
  namesused = sort(collect(namesused), by = x -> x[1])
  
  for (name, pkgs) in namesused
    if length(pkgs) > 1
      handles = map(pkgs) do pkg
        getfield(pkg, name)
      end
      handles = unique(handles)
      # Filter out empty methods
      # TODO, add a note to flag any empty methods... 
      allmethods = methods.(handles)
      for ms in allmethods
        if length(ms) == 0 
          println(stderr, "Warning: empty method for $name ")
          println(stderr, ms)
        end 
      end
      #filter!(ms -> !isempty(ms), allmethods)
      if length(allmethods) > 1
        println(io)
        println(io, "Showing duplicate methods for $name in packages $(pkgs)")
        for h in unique(handles) 
          println(io, "Methods for $(h) in package $(typeof(h).name.module)")
          for method in methods(h)
            println(io, method)
          end
        end
      end
    end
  end
end 


function _read_overrides(filename)
  overrides = Dict{Symbol, Vector{String}}()
  name = nothing 
  overrides_lines = String[] 
  aftername = false 
  open(filename, "r") do io 
    lines = readlines(io)
    for line in lines 
      if startswith(line, "## :")
        # we start a new symbol 
        if name !== nothing
          overrides[name] = copy(overrides_lines)
        end
        # parse a symbol from a string 
        expr = Meta.parse(line[4:end])
        if expr isa QuoteNode
          name = expr.value
        else
          name = expr.args[1].args[1]
        end 
        empty!(overrides_lines)
        aftername = true 
      elseif startswith(line, "##-")
        continue # this is just an extra comment we use for structural pieces... 
      elseif startswith(line, "#=")
        continue # this is just an extra comment we use for structural pieces... 
      elseif startswith(line, "=#")
        continue # this is just an extra comment we use for structural pieces... 
      elseif aftername && startswith(line, "#")
        continue 
      else
        aftername = false 
        push!(overrides_lines, line)
      end
    end
  end 
  # handle the last item
  if name !== nothing
    overrides[name] = copy(overrides_lines)
  end
  return overrides 
end 

function _write_overrides(;io=stdout, namesused, overrides)
  namesused = sort(collect(namesused), by = x -> x[1])
  println(io, "const overrides = Set{Symbol}()")
  println(io, "") 
  usedoverrides = Set{Symbol}()
  
  for (name, pkgs) in namesused
    if length(pkgs) > 1
      handles = map(pkgs) do pkg
        getfield(pkg, name)
      end
      handles = unique(handles)
      # Filter out empty methods
      # TODO, add a note to flag any empty methods... 
      allmethods = methods.(handles)
      #filter!(ms -> !isempty(ms), allmethods)
      if length(allmethods) > 1
        println(io, "## :$name")
        println(io, "# Showing duplicate methods for $name in packages $(pkgs)")
        for h in unique(handles) 
          println(io, "# Methods for $(h) in package $(typeof(h).name.module)")
          for method in methods(h)
            println(io, "# ", method)
          end
        end
        if name in keys(overrides)
          println.(io, overrides[name]) # add the previous overrides... 
          # remove the name from the list of names used...
          push!(usedoverrides, name)
        end
      end
    end
  end

  # Now print out any remaining overrides...
  println(io, "##-Unused overrides")
  remaining_overrides = setdiff(keys(overrides), usedoverrides)
  sort!(collect(remaining_overrides))
  for name in remaining_overrides
    println(io, "#=")
    println(io, "## :$name")
    println.(io, overrides[name])
    println(io, "=#")
  end
end

# Often I use this as follows... 
# include("test/test_names_and_methods.jl"); merge_overrides_file()
function merge_overrides_file(;filename="src/overrides.jl", packages = TechnicalCompute.packages)
  # first, make a copy of overrides file with the current timestamp and a backup
  # copy of the original file
  timestamp = Dates.format(now(), "yyyy-mm-dd-HHMMSS")
  backupfilename = replace(filename, ".jl" => "-$timestamp.backup")  
  cp(filename, backupfilename) # TODO, how to tell if this failed? 

  # Now we need to read in the the file and the list of overrides... 
  overrides = _read_overrides(filename)

  # Now write out any changes... 
  namesused = _names_in_packages(map(pkg->eval(Meta.parse("$pkg")), packages))
  open(filename, "w") do io 
    _write_overrides(;io=io, namesused, overrides)
  end
end 
  