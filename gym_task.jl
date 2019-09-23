using OpenAIGym

env = OpenAIGym.GymEnv(:CartPole, :v1)

INPUT_SIZE = length(env.state)
OUTPUT_SIZE = length(env.actions)
TRAIN_EPISODES = 300
ENV_STEPS = 200
PARALLEL_NETS = 10
LEARNING_RATE = 0.04
γ = 0.9 # future reward discount

l1, dl1 = FrequencyLayer(INPUT_SIZE, OUTPUT_SIZE)


ff_net = [l1]

train_metric = [NaN for _ in 1:TRAIN_EPISODES]; for e in 1:TRAIN_EPISODES

    reward_buffers = [[] for _ in 1:PARALLEL_NETS]
    for pn in 1:PARALLEL_NETS
        s = reset!(env)
        for es in 1:ENV_STEPS
            y = feed_forward!(ff_net, s |> Array)

            a = env.actions[y[argmax(y)]]
            r, s_ = step!(env, a)

            println("sate: $s_")

            append!(reward_buffers[pn], [[r, s, a]])

            if env.done
                break
            end
        end
    end

    # discount reward for future rewards in each iteration
    sums = [0 for _ in eachindex(reward_buffers)]
    for rb in eachindex(reward_buffers)
        for rsa_i in eachindex(reward_buffers[rb])-1
            discouned_future_reward = sum([(γ ^ (i-rsa_i+1)) * reward_buffers[rb][i][1] for i in rsa_i+1:length(reward_buffers[rb])])
            reward_buffers[rb][rsa_i][1] += discouned_future_reward

            sums[rb] += reward_buffers[rb][rsa_i][1]
        end
    end

    best_net = reward_buffers[argmax(sums)]

    for (_, s, a) in best_net
        a_ = ff_net(s)
        loss = mse(a_, a)
        d_loss = d_mse(a_, a)
        fb_net(d_loss)
    end
end
