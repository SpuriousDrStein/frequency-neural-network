include("network_functions.jl")

using OpenAIGym
using Random
using Plots

env = OpenAIGym.GymEnv(:CartPole, :v1)

INPUT_SIZE = length(env.state)
OUTPUT_SIZE = length(env.actions)
TRAIN_EPISODES = 10000
TEST_EPISODES = 50
MAX_ENV_STEPS = 200
PARALLEL_NETS = 10
LEARNING_RATE = 0.04
γ = 0.9 # future reward discount
Random.seed!(1307)


l1, dl1 = FrequencyLayer(INPUT_SIZE, OUTPUT_SIZE, 1)
# l2, dl2 = ActivationLayer(OUTPUT_SIZE, 1, sigmoid, Δsigmoid)

ff_net = [l1]
fb_net = [dl1]


train_metric = [NaN for _ in 1:TRAIN_EPISODES]; for e in 1:TRAIN_EPISODES

    reward_buffers = [[] for _ in 1:PARALLEL_NETS]
    for pn in 1:PARALLEL_NETS
        s = reset!(env)
        for es in 1:MAX_ENV_STEPS
            y = feed_forward!(s |> Array, ff_net)

            a = env.actions[argmax(y)]
            r, s_ = step!(env, a)

            append!(reward_buffers[pn], [[r, s, a]])

            if env.done
                break
            end
        end
    end

    # discount reward for future rewards in each iteration
    sums = [0. for _ in eachindex(reward_buffers)]
    for rb in eachindex(reward_buffers)
        for rsa_i in 1:length(reward_buffers[rb])-1
            discounted_future_reward = sum([(γ ^ (i-rsa_i+1)) * reward_buffers[rb][i][1] for i in rsa_i+1:length(reward_buffers[rb])])

            reward_buffers[rb][rsa_i][1] *= discounted_future_reward
            sums[rb] += reward_buffers[rb][rsa_i][1]
        end
    end

    best_net = reward_buffers[argmax(sums)]

    println("episode: $e -- best net total reward: $(maximum(sums))")
    train_metric[e] = maximum(sums)

    for (_, s, a) in best_net
        a_ = feed_forward!(s, ff_net)
        loss = mse(a_, a)
        d_loss = Δmse(a_, a)
        feed_backwards!(d_loss, fb_net)
    end
end


plot(train_metric)


# testing
for e in 1:TEST_EPISODES
    s = reset!(env)
    for es in 1:MAX_ENV_STEPS
        y = feed_forward!(s |> Array, ff_net)

        a = env.actions[argmax(y)]
        r, s_ = step!(env, a)

        render(env)

        if env.done
            break
        end
    end
end; OpenAIGym.close(env)
