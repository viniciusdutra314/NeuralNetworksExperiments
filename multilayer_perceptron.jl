using MLDatasets

struct Network{T<:AbstractFloat} 
    layers::Vector{Int}
    b_vals::Vector{Array{T,1}}
    w_vals::Vector{Array{T,2}}
end 
σ(x)=1/(1+exp(-x))


function Network(layers::Vector{Int})
    b_vals = [randn(Float64, layer_size) for layer_size in layers[2:end]]
    w_vals = [randn(Float64, layer_size, prev) for (prev, layer_size) in zip(layers[1:end-1], layers[2:end])]
    return Network(layers, b_vals, w_vals)
end

function activation(net::Network, a::Vector{Float64})
    a_copy=copy(a)
    for (b,w) in zip(net.b_vals, net.w_vals)
        a_copy = σ.(w*a_copy+b)
    end
    return a_copy
end


dataset=MNIST(Tx=Float64).features
img=dataset[:,:,1]
a=vec(img)
N=size(a,1)
net=Network([N,N,N,10])


