# We don't automatically detect conflicts with Base
# Aqua.jl will report conflicts with base as undefined exports

# No point in doing this as Base already exports it... 
# const Text = Makie.Text
# export Text

# Julia 1.10 includes tanpi, but many functions haven't added it yet.
# this is type piracy, but it fixes a broader problem. 
import Base.tanpi
tanpi(x::DoubleFloat) = DoubleFloats.tanpi(x) 
export tanpi

# data is deprecated, so we don't report on it.
import Base.axes
export axes 