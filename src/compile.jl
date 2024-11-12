## based on ITensors.jl or ITensorsMPS.jl compile
# https://github.com/ITensor/ITensorMPS.jl/blob/main/ext/ITensorMPSPackageCompilerExt/compile.jl and
# https://github.com/ITensor/ITensors.jl/blob/main/src/packagecompile/compile.jl 

default_compile_dir() = joinpath(homedir(), ".julia", "sysimages")

default_compile_filename() = "sys_technicalcompute.so"

#default_compile_path() = joinpath(default_compile_dir(), default_compile_filename())

import PackageCompiler
"""
    compile(; [dir=$(default_compile_dir()), filename=$(default_compile_filename())])

Use PackageCompiler to compile the TechnicalCompute package into a system image. This 
makes it much faster to use is if you are commonly using this package. 
"""
function compile(; dir=default_compile_dir(), filename=default_compile_filename(), _coverage_only::Bool=false)
  path = joinpath(dir, filename)
  if !isdir(dir)
    println("""The directory "$dir" doesn't exist yet, creating it now.""")
    println()
    mkdir(dir)
  end
  println(
    """Creating the system image "$path" containing the compiled version of TechnicalCompute. This may take a few minutes.""",
  )
  _coverage_only && return # return early if we are just running tests...
  PackageCompiler.create_sysimage(
    ["TechnicalCompute"]; # should we also add all the dependencies? 
    sysimage_path=path,
    #precompile_execution_file=joinpath(@__DIR__, "precompile", "precompile-for-sysimage.jl"),
  )
end