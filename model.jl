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

                dlocaly_df[i,j] = (w[i,j] * cos(X[j] * f[i, j] + o[i]) * X[j]) / pdiv
                dlocaly_dw[i,j] = sin(X[j] * f[i,j] + o[i]) / pdiv
                dlocaly_dp[i,j] = (-w[i,j] * sin(X[j] * f[i,j] + o[i]) * 2 * p[i,j]) / (pdiv ^ 2)
                dlocaly_do[i] += (w[i,j] * cos(X[j] * f[i,j])) / pdiv
            end
        end
        return out
    end

    function backward!(Δ, lr)
        println(Δ |> size)

        for i in 1:output_size
            b[i] += lr * Δ[i]
            o[i] += lr * Δ[i] * dlocaly_do[i]

            for j in 1:input_size
                w[i,j] += lr * Δ[i] * dlocaly_dw[i,j]
                p[i,j] += lr * Δ[i] * dlocaly_dp[i,j]
                f[i,j] += lr * Δ[i] * dlocaly_df[i,j]
            end
        end
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
