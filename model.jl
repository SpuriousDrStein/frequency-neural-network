using Plots
using OpenAIGym



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
                dlocaly_dw[i,j] = sin(x[j] * f[i,j] + o[i]) / pdiv
                dlocaly_dp[i,j] = (-w[i,j] * sin(X[j] * f[i,j] + o[i]) * 2 * p[i,j]) / (pdiv ^ 2)
                dlocaly_do[i] += (w[i,j] * cos(X[j] * f[i,j])) / pdiv
            end
        end
        return out
    end

    function backward!(dprediction_dlocaly, lr)
        for i in 1:output_size
            b[i] += lr * dprediction_dlocaly[i]
            o[i] += lr * dprediction_dlocaly[i] * dlocaly_do[i]

            for j in 1:input_size
                w[i,j] += lr * dprediction_dlocaly[i] * dlocaly_dw[i,j]
                p[i,j] += lr * dprediction_dlocaly[i] * dlocaly_dp[i,j]
                f[i,j] += lr * dprediction_dlocaly[i] * dlocaly_df[i,j]
            end
        end
    end

    return forward!, backward!
end




env = OpenAIGym.GymEnv(:CartPole, :v1)

INPUT_SIZE = length(env.state)
OUTPUT_SIZE = length(env.actions)
TRAIN_EPISODES = 300
ENV_STEPS = 200
LEARNING_RATE = 0.04


l1, dl1 = FrequencyLayer(INPUT_SIZE, OUTPUT_SIZE)

y = l1(env.state |> Array)

dl1(y, LEARNING_RATE)


l1.dlocaly_df |> println
l1.dlocaly_do |> println
l1.dlocaly_dp |> println
dl1.dlocaly_dw |> println


train_metric = [NaN for _ in 1:TRAIN_EPISODES]; for e in 1:TRAIN_EPISODES

    s = reset!(env) |> Array
    for es in 1:ENV_STEPS
        y = ff_net
        r, s = step!(env, env.actions[y[argmax(y)]])



        if env.done
            break
        end
    end
end

scatter([o for o in p[:, 1]])
