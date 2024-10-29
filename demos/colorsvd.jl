##
using TechnicalCompute

# load the coffee image 
img = testimage("coffee")

function make_svd_approximation(img) 
  Xs = vcat(eachslice(permutedims(channelview(img), (2, 3, 1)), dims=3)...)
  Us, Ss, Vs = svd(Xs)
  approx_function(k::Int) = begin 
    Xsr = Us[:, 1:k] * Diagonal(Ss[1:k]) * Vs[:, 1:k]'
    err = norm(Xs - Xsr)/norm(Xs) 
    Yr = Xsr[1:size(img,1),:]
    Cbr = Xsr[(size(img,1)+1):(2*size(img,1)),:]
    Crr = Xsr[(2*size(img,1)+1):(3*size(img,1)),:]
    Tr = cat(Yr, Cbr, Crr; dims=3)

    Tr = permutedims(Tr, (3, 1, 2))

    @show sizeof(Tr)
    @show sizeof(eltype(img))

    println("using ", length(Us[:,1:k]) + length(Vs[:,1:k]), " values to approximate the image with $(100err)% error")
    return colorview(eltype(img).name.wrapper, Tr)
  end 
  return approx_function
end 
imageapprox = make_svd_approximation(img)
imageapprox(9)

##
import REPL.TerminalMenus
menu = REPL.TerminalMenus.RadioMenu(repr.(1:20), pagesize=30)
curchoice = Ref{Int}(1) # default choice
while true
    # `request` displays the menu and returns the index after the
    #   user has selected a choice
    @show curchoice[]
    println("Choose your approximation rank:")
    choice = REPL.TerminalMenus.request(menu; cursor=curchoice)
    if choice != -1
        println("Looking at rank ", choice)
        display(imageapprox(choice))
    else
        println("Menu canceled.")
        break
    end
end
