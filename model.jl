using Plots
using OpenAIGym



function FreqLayer(input_size, output_size)
    p = rand(output_size, input_size)
    freq = rand(output_size, input_size)
    off = rand(output_size)
    w = rand(output_size, input_size)
    b = rand(output_size)
    ϵ = 1
    dlocaly_dfreq = zeros()
    dlocaly_doff = zeros()
    dlocaly_dw = zeros()
    dlocaly_db = zeros()
    dlocaly_dp = zeros()

    function forward!(t, X, freq, off, w, b, p)
        out = [0. for _ in 1:output_size]
        for i in 1:output_size
            for j in 1:input_size
                out[i] += (sin(t * freq[i, j] * X[j] + off[i]) / (p[i, j] ^ 2 + ϵ)) * w[i, j] + b[i]
            end
        end
        return out
    end

    function backward!(dprediction_dlocaly, lr)

    end

    return forward!, backward!
end




env = OpenAIGym.GymEnv(:CartPole, :v1)

INPUT_SIZE = length(env.state)
OUTPUT_SIZE = length(env.actions)
TRAIN_EPISODES = 300
ENV_STEPS = 200


l1 = FreqLayer(INPUT_SIZE, OUTPUT_SIZE)


metric = [NaN for _ in 1:TRAIN_EPISODES]; for e in 1:TRAIN_EPISODES

end

scatter([o for o in p[:, 1]])
