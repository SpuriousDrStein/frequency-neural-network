include("model.jl")

using Plots
using MLDatasets
using Random

# ::: DATA
train_x, train_y = MNIST.traindata()
train_x = permutedims(Array{Float32}(train_x), [3,2,1])
train_x = reshape(train_x, (length(train_x[:, 1, 1]), length(train_x[1, :, 1]) * length(train_x[1, :, 1])))
get_MNIST_batch(total_x, BS) = begin
    ri = rand(1:length(total_x[:, 1])-BS)
    return total_x[ri:ri+BS-1, :]'
end

# ::: PARAMS
INPUT_SIZE = length(train_x[1, :, :])
LATENT_SIZE = 70
BATCH_SIZE = 1
LR = 0.04
TRAIN_EPISODES = 1000
Random.seed!(1301)


l1, dl1 = FrequencyLayer(INPUT_SIZE, 500)
l2, dl2 = FrequencyLayer(500, 300)
l3, dl3 = FrequencyLayer(300, 150)
l4, dl4 = FrequencyLayer(150, LATENT_SIZE)
encoder_net_f, encoder_net_b = [l1, l2, l3, l4], [dl1, dl2, dl3, dl4]

l5, dl5 = FrequencyLayer(LATENT_SIZE, 150)
l6, dl6 = FrequencyLayer(150, 300)
l7, dl7 = FrequencyLayer(300, 500)
l8, dl8 = FrequencyLayer(500, INPUT_SIZE)
decoder_net_f, decoder_net_b = [l5, l6, l7, l8], [dl5, dl6, dl7, dl8]


# loss function + derivative
mse(x, y) = sum((x.-y).^2) / length(x)
Δmse(x, y) = 2 .* (x .- y) ./ length(x)



# ::: TRAINING
runtime_plotting = true
plot_intervall = 15
train_metric = [NaN for _ in 1:TRAIN_EPISODES]; println("---"); @time for e in 1:TRAIN_EPISODES

    curr_batch = get_MNIST_batch(train_x, BATCH_SIZE)

    z = feed_forward!(curr_batch, encoder_net_f)
    y = feed_forward!(z, decoder_net_f)

    loss = mse(curr_batch, y)
    dloss = Δmse(curr_batch, y)

    println(dloss |> size, " ", dloss .|> isnothing |> any)

    dz = feed_backwards!(dloss, decoder_net_b, lr=LR)
    _ = feed_backwards!(dz, encoder_net_b, lr=LR)

    println("episode $e : $loss = ", loss)
    train_metric[e] = loss

    if runtime_plotting && e % plot_intervall == 0
        p = plot(train_metric)
        display(p)
    end
end


plot(train_metric, xlabel="training iterations", label="MSE (lr=$LR, batch size=$BATCH_SIZE)")
