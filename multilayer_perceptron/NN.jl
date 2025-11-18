include("dual_numbers.jl")
using MLDatasets
using LinearAlgebra
using Base.Iterators: partition


σ(x) = 1 / (1 + exp(-x))
ReLU(x) = max(0, x)
SoftMax(x) = exp.(x) / sum(exp.(x))
ϕ(x) = ReLU(x)


struct Network{T<:Number}
    layers::Vector{Int}
    b_vals::Vector{Array{T,1}}
    w_vals::Vector{Array{T,2}}
end

function Network(layers::Vector{Int})
    b_vals = [randn(Float64, layer_size) for layer_size in layers[2:end]]
    w_vals = [randn(Float64, layers[i+1], layers[i]) for i in 1:length(layers)-1]
    return Network{Float64}(layers, b_vals, w_vals)
end

function activation(net::Network{T}, a::Vector{S}) where {T<:Number,S<:Number}
    a_copy = convert(Vector{promote_type(T, S)}, a)
    for (b, w) in zip(net.b_vals, net.w_vals)
        a_copy = ϕ.(w * a_copy + b)
    end
    return a_copy
end
function cost_cross_entropy(output::Vector{T}, target_label::Int) where {T<:Number}
    target_vector = zeros(T, length(output))
    target_vector[target_label+1] = one(T)
    return -sum(target_vector .* log.(SoftMax(output)))
end

function promote_net_to_dual(net::Network{Float64})
    dual_b_vals = [convert(Array{DualNumber{Float64}}, b) for b in net.b_vals]
    dual_w_vals = [convert(Array{DualNumber{Float64}}, w) for w in net.w_vals]
    return Network{DualNumber{Float64}}(net.layers, dual_b_vals, dual_w_vals)
end


function check_accuracy(net::Network{Float64}, dataset)
    correct = 0
    total = length(dataset.targets)
    for idx in 1:total
        img = dataset.features[:, :, idx]
        label = dataset.targets[idx]
        a_float = vec(img)
        output = activation(net, a_float)
        predicted_label = argmax(output) - 1
        if predicted_label == label
            correct += 1
        end
    end
    return correct / total
end