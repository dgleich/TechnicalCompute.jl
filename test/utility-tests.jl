## Test utilities like compile and the preferences.
@test_throws ArgumentError TechnicalCompute.set_makie_backend("PyPlot")
@test begin 
  TechnicalCompute.set_makie_backend("GLMakie")
  return true
end 

@test begin 
  TechnicalCompute.set_show_banner(false)
  return true
end 
@test begin 
  TechnicalCompute.set_makie_load_glmakie(false)
  return true
end 

TechnicalCompute.compile(; _coverage_only=true)
