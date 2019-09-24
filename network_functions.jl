function FrequencyLayer(input_size, output_size)
    p = rand(output_size, input_size)   # point
    f = rand(output_size, input_size)   # frequency
    o = rand(output_size)               # offset
    w = rand(output_size, input_size)   # weight
    b = rand(output_size)               # bias
    ϵ = 1
    dlocaly_df = zeros(output_size, input_size)
    dlocaly_dw = zeros(output_size, input_size)
    dlocaly_dp = zeros(output_size, input_size)
    dlocaly_do = zeros(output_size)
    dlocaly_dlocalX = zeros(output_size, input_size)

    function forward!(X)
        out = [0. for _ in 1:output_size]
        for i in 1:output_size
            dlocaly_do[i] = 0.

            for j in 1:input_size
                pdiv = (p[i, j] ^ 2 + ϵ)

                out[i] += (sin(f[i,j] * X[j] + o[i]) / pdiv) * w[i,j] + b[i]

                dlocaly_df[i,j] = (w[i,j] * cos(X[j] * f[i,j] + o[i]) * X[j]) / pdiv
                dlocaly_dw[i,j] = sin(X[j] * f[i,j] + o[i]) / pdiv
                dlocaly_dp[i,j] = (-w[i,j] * sin(X[j] * f[i,j] + o[i]) * 2 * p[i,j]) / (pdiv ^ 2)
                dlocaly_dlocalX[i,j] = (cos(X[j] * f[i,j] * o[i]) * f[i,j] * w[i,j]) / pdiv
                dlocaly_do[i] += (w[i,j] * cos(X[j] * f[i,j])) / pdiv
            end
        end
        return out
    end

    function backward!(Δ, lr)
        dpred_dx = zeros(input_size)
        for i in 1:output_size
            b[i] += lr * Δ[i]
            o[i] += lr * Δ[i] * dlocaly_do[i]

            for j in 1:input_size
                w[i,j] += lr * Δ[i] * dlocaly_dw[i,j]
                p[i,j] += lr * Δ[i] * dlocaly_dp[i,j]
                f[i,j] += lr * Δ[i] * dlocaly_df[i,j]

                dpred_dx[j] += lr * dlocaly_dlocalX[i,j]
            end
        end
        return dpred_dx
    end

    return forward!, backward!
end

function BatchNormLayer(in_out_size, batch_size)
    α = rand(in_out_size, batch_size)
    β = rand(1, batch_size)
    ϵ = 0.00001 # to prevent devision by zero

    dlocaly_dα = zeros(in_out_size, batch_size)
    dlocaly_dlocalX = zeros(in_out_size, batch_size)

    function forward!(X)
        N = length(X)
        μ = sum(X) / N
        σ = sum((X .- μ) .^ 2) / N

        bn = (X .- μ) ./ sqrt(σ + ϵ)
        y = α .* bn .+ β

        # println("μ: ", sqrt(σ + ϵ))
        # println("σ: ", sqrt(σ + ϵ))
        # println("sqrt(σ + ϵ): ", sqrt(σ + ϵ))

        dlocaly_dlocalX .= α .* sqrt(σ + ϵ)
        dlocaly_dα .= bn

        return y
    end

    function backward!(dprediction_dlocaly, lr)
        α .+= lr .* (dprediction_dlocaly .* dlocaly_dα)
        β .+= lr .* (sum(dprediction_dlocaly; dims=1) ./ in_out_size)

        return dprediction_dlocaly .* dlocaly_dlocalX
    end

    return forward!, backward!
end

function ActivationLayer(in_out_size, batch_size, actf, d_actf)
    dlocaly_dlocalX = zeros(in_out_size, batch_size)

    function forward!(X)
        dlocaly_dlocalX .= d_actf(X)
        return actf(X)
    end

    function backward!(Δ, lr)
        return Δ .* dlocaly_dlocalX
    end

    return forward!, backward!
end

function feed_forward!(X, ff_net::Array)
    for l in ff_net
        X = l(X)
    end
    return X
end

function feed_backwards!(d_pred, fb_net::Array; lr=0.006)
    for bl in reverse(fb_net)
        d_pred = bl(d_pred, lr)
    end
    return d_pred
end

relu(A::Array) = map(a->a * (a>0), A)
d_relu(A::Array) = map(a->(a>0), A)

sigmoid(A::Array) = 1 ./ (1 .+ exp.(.-A))
d_sigmoid(A::Array) = sigmoid(A) .* (1 .- sigmoid(A))
