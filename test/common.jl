
const workdir = joinpath(tempdir(), "TechnicalCompute")
const tempfiles = String[] 
_filename(file) = begin; fn = joinpath(workdir, file); push!(tempfiles, fn); fn; end

function setup_workdir() 
  try 
    if isdir(workdir)
      println("Workdir: $workdir ALREADY EXISTS")
    else 
      println("Creating workdir: $workdir")
      mkdir(workdir)
    end
  catch 
  end 
end 

function cleanup_workdir()
  try 
    if isdir(workdir)
      println("Cleaning up workdir: $workdir")
      println("Files: ", tempfiles)
      rm(workdir; force=true, recursive=true)
    end
  catch 
  end 
end 

function make_itensor_mps()
  d = 2
  N = 5
  A = randn(d,d,d,d,d)
  sites = siteinds(d,N)
  cutoff = 1E-8
  maxdim = 10
  M = MPS(A,sites;cutoff=cutoff,maxdim=maxdim)
end 