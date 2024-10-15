Notes on duplicate resolution
-----------------------------

**All methods can be called by fully qualifying their names!** 

both JuMP and Symbolics.jl export a macro @variables to help declare variables. 
I decided to give this one to Symbolics as it was more prominently featured in 
their documenation. 

This is merely to establish a dialect that allows one to avoid that,
where it can be resolved with more refinement over the namespace. 

Categorial is duplicated between distributions and Makie
in Makie, Categorical is used to indicate a categorical colormap, so this was renamed CategoricalColormap. 

ImageFiltering.Fill represents the value to be used when the filter is applied to the border of the image. The default value is zero. So we change the name to FillValue
FillArrays.Fill represents a matrix with a single value. The default value is zero. So we change the name to FillArray

Fixed is duplicated between FixedPointNumbers and Makie
but in Makie, this is used to indicate a fixed size partition of a GridLayout, so we change it to FixedSize

Graph is used in DelaunayTriangulation and Graphs.jl
We keep the meaning from Graphs.jl

Normal is duplicated between Distributions and GeometryBasics
but in GeometryBasics, this is used to indicate a normal vector, so we change it to NormalVector

Partition is duplicated between Transducers.jl and Combinatorics.jl 
but the use in Combinatorics seems to be restricted to young diagrams, which is fairly specialized.

Zeros is duplicated between FillArrays and JuMP
We keep the meaning of the Zeros from FillArrays here as Jump is a very specialized use. 

Where there were duplicates, we gave preference to the DSP functions vs. the NNLib functions included by Flux. 
- conv
- spectrogram
- stft
