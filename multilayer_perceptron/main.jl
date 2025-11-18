include("NN.jl")
using MLDatasets: MNIST

function train_on_example!(net::Network{Float64}, dataset; idx::Int, nsteps::Int, η::Float64)
    a = vec(dataset.features[:, :, idx])
    label = dataset.targets[idx]
    dual_net = promote_net_to_dual(net)
    for _ in 1:nsteps
        grads_w = [zeros(size(w)) for w in dual_net.w_vals]
        grads_b = [zeros(length(b)) for b in dual_net.b_vals]

        for l in 1:length(dual_net.b_vals)
            for i in 1:size(dual_net.w_vals[l], 1), j in 1:size(dual_net.w_vals[l], 2)
                temp_w = dual_net.w_vals[l][i, j]
                dual_net.w_vals[l][i, j] = DualNumber(temp_w.real, 1.0)
                cost_dual = cost_cross_entropy(activation(dual_net, a), label)
                grads_w[l][i, j] = cost_dual.dual
                dual_net.w_vals[l][i, j] = temp_w
            end
            
            for i in 1:length(dual_net.b_vals[l])
                temp_b = dual_net.b_vals[l][i]
                dual_net.b_vals[l][i] = DualNumber(temp_b.real, 1.0)
                cost_dual = cost_cross_entropy(activation(dual_net, a), label)
                grads_b[l][i] = cost_dual.dual
                dual_net.b_vals[l][i] = temp_b
            end
        end

        for l in 1:length(dual_net.w_vals)
            dual_net.w_vals[l] .-= η .* grads_w[l]
            dual_net.b_vals[l] .-= η .* grads_b[l]
        end
    end
    dual_to_real(x) = x.real

    for l in 1:length(net.w_vals)
        net.w_vals[l] .= dual_to_real.(dual_net.w_vals[l])
        net.b_vals[l] .= dual_to_real.(dual_net.b_vals[l])
    end
    return nothing
end


dataset = MNIST(Tx=Float64)
N = 28 * 28
net = Network([N, 16, 16, 10])
idx = 1
a = vec(dataset.features[:, :, idx])
label = dataset.targets[idx]

println("Single-example demo: idx=$idx label=$label")
println("Before: predicted=", argmax(activation(net, a)) - 1, 
        " cost=", 
        cost_cross_entropy(activation(net, a), label))
train_on_example!(net, dataset; idx=idx, nsteps=30, η=1e-3)
println("After:  predicted=", 
        argmax(activation(net, a)) - 1, 
        " cost=", 
        cost_cross_entropy(activation(net, a), label))
