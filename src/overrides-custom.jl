
# We don't automatically detect conflicts with Base
# Aqua.jl will report conflicts with base as undefined exports

# Julia 1.10 includes tanpi, but many functions haven't added it yet.
import Base.tanpi
tanpi(x::DoubleFloat) = DoubleFloats.tanpi(x) 
export tanpi


