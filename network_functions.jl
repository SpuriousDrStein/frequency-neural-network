using CuArrays
using LinearAlgebra

function FrequencyLayer(input_size, output_size, batch_size; Lmin=1, magnitude_delim=10)
    a = [CuArrays.rand(output_size, input_size) for _ in 1:batch_size]
    b = [CuArrays.rand(output_size, input_size) for _ in 1:batch_size]
    c = [CuArrays.rand(output_size) for _ in 1:batch_size]

    N = CuArray([i/magnitude_delim for i in 1:input_size]) # constant
    L = [1. for _ in 1:batch_size] # usualy this is the period of the to be appriximated function

    dlocaly_da = [CuArrays.zeros(input_size) for _ in 1:batch_size]
    dlocaly_db = [CuArrays.zeros(input_size) for _ in 1:batch_size]
    # dlocaly_dL = [0. for _ in 1:batch_size]
    dlocaly_dlocalX = [CuArrays.zeros(input_size, output_size) for _ in 1:batch_size]

    function forward!(x)
        y = [CuArrays.zeros(output_size) for i in 1:batch_size]

        for i in 1:batch_size
            l = π/(L[i]+Lmin)

            y[i] = a[i] * sin.(x[i] .* N .* l) .+ b[i] * cos.(x[i] .* N .* l) .+ c[i]

            dlocaly_da[i] = sin.(x[i] .* N .* l)
            dlocaly_db[i] = cos.(x[i] .* N .* l)
            # dlocaly_dL[i] = sum(a[i] * (cos.(x[i] .* N .* l) .* x[i] .* N .* (-π/(L[i]+Lmin^2))) .+ b[i] * (sin.(x[i] .* N .* l) .* x[i] .* N .* (π/(L[i]+Lmin^2)))) / length(x[i])


            dlocaly_dlocalX[i] = a[i]' .* (cos.(x[i] .* N .* l) .* N .* l) .- b[i]' .* (sin.(x[i] .* N .* l) .* N .* l)
        end

        # println("dlocaly_dL batch mean: ", (dlocaly_dL |> sum) / batch_size)

        return y
    end

    function backward!(Δ, lr)
        dpred_dx = [CuArrays.zeros(input_size) for i in 1:batch_size]

        for i in 1:batch_size

            a[i] .+= lr .* (Δ[i] * dlocaly_da[i]')
            b[i] .+= lr .* (Δ[i] * dlocaly_db[i]')
            c[i] .+= lr .* Δ[i]

            dpred_dx[i] .= dlocaly_dlocalX[i] * Δ[i]
        end

        return dpred_dx
    end

    return forward!, backward!
end

function ActivationLayer(in_out_size, batch_size, actf, d_actf)
    dlocaly_dlocalX = [CuArrays.zeros(in_out_size, in_out_size) for _ in 1:batch_size]

    function forward!(x)
        y = [CuArrays.zeros(in_out_size) for i in 1:batch_size]

        for i in 1:batch_size
            dlocaly_dlocalX[i] .= d_actf(x[i])
            y[i] = actf(x[i])
        end
        return y
    end

    function backward!(Δ, lr)
        dprediction_dlocalX = [similar(Δ[i]) for i in 1:batch_size]

        for i in 1:batch_size
            dprediction_dlocalX[i] = dlocaly_dlocalX[i] * Δ[i]
        end
        return dprediction_dlocalX
    end

    return forward!, backward!
end

function feed_forward!(x, forward_layer_functions::AbstractArray)
    for ff_layer in forward_layer_functions
        x = ff_layer(x)
    end
    return x
end

function feed_backwards!(d_pred, fb_net::AbstractArray; lr=0.006)
    for bl in reverse(fb_net)
        d_pred = bl(d_pred, lr)
    end
    return d_pred
end

mse(x::CuArray, y::CuArray) = sum((x .- y).^2) / length(x)
Δmse(x::CuArray, y::CuArray) = 2 .* (x .- y) ./ length(x)

sigmoid(A::CuArray) = 1 ./ (1 .+ exp.(.-A))
Δsigmoid(A::CuArray) = CuArray{Float32}(I, length(A), length(A)) .* (sigmoid(A) .* (1 .- sigmoid(A)))

relu(A::CuArray) = map(a->a * (a>0), A)
Δrelu(A::CuArray) =  CuArray{Float32}(I, length(A), length(A)) .* map(a->(a>0), A)
