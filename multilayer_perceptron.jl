using MLDatasets
using LinearAlgebra 
include("dual_numbers.jl")

struct Network{T<:Number} 
    layers::Vector{Int}
    b_vals::Vector{Array{T,1}}
    w_vals::Vector{Array{T,2}}
end 

σ(x) = 1 / (1 + exp(-x))

function Network(layers::Vector{Int})
    b_vals = [randn(Float64, layer_size) for layer_size in layers[2:end]]
    w_vals = [randn(Float64, layers[i+1], layers[i]) for i in 1:length(layers)-1]
    return Network{Float64}(layers, b_vals, w_vals)
end

function activation(net::Network{T}, a::Vector{S}) where {T<:Number, S<:Number}
    a_copy = copy(a) 
    for (b, w) in zip(net.b_vals, net.w_vals)
        a_copy = σ.(w * a_copy + b)
    end
    return a_copy 
end

dataset = MNIST(Tx=Float64)
N = 28 * 28
net = Network([N, N, N, 10]) 

img = dataset.features[:,:,1]
label = dataset.targets[1]
a_float = vec(img) 
pixel_to_diff = 100
a_dual = convert(Vector{DualNumber{Float64}}, a_float)

a_dual[pixel_to_diff] = DualNumber(a_float[pixel_to_diff], 1.0)


output_dual = activation(net, a_dual)
println(output_dual)