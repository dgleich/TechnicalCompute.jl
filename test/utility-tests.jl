## Test utilities like compile and the preferences.
@test_throws ArgumentError TechnicalCompute.set_make_backend("PyPlot")
@test begin 
  set_make_backend("GLMakie")
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

if Sys.isunix() && "test_compile" in ARGS
  @test begin 
    TechnicalCompute.compile()
  end 
end 