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
TRAIN_EPISODES = 3000
Random.seed!(1301)


l1, dl1 = FrequencyLayer(INPUT_SIZE, 500)
l1a, dl1a = ActivationLayer(500, BATCH_SIZE, sigmoid, d_sigmoid)
# l1bn, dl1bn = BatchNormLayer(500, BATCH_SIZE)
l2, dl2 = FrequencyLayer(500, 300)
l2a, dl2a = ActivationLayer(300, BATCH_SIZE, sigmoid, d_sigmoid)
# l2bn, dl2bn = BatchNormLayer(300, BATCH_SIZE)
l3, dl3 = FrequencyLayer(300, 150)
l3a, dl3a = ActivationLayer(150, BATCH_SIZE, sigmoid, d_sigmoid)
# l3bn, dl3bn = BatchNormLayer(150, BATCH_SIZE)
l4, dl4 = FrequencyLayer(150, LATENT_SIZE)
l4a, dl4a = ActivationLayer(LATENT_SIZE, BATCH_SIZE, sigmoid, d_sigmoid)
# l4bn, dl4bn = BatchNormLayer(LATENT_SIZE, BATCH_SIZE)
# full -> encoder_net_f, encoder_net_b = [l1, l1a, l1bn, l2, l2a, l2bn, l3, l3a, l3bn, l4, l4a, l4bn], [dl1, dl1a, dl1bn, dl2, dl2a, dl2bn, dl3, dl3a, dl3bn, dl4, dl4a, dl4bn]
encoder_net_f, encoder_net_b = [l1, l1a, l2, l2a, l3, l3a, l4, l4a], [dl1, dl1a, dl2, dl2a, dl3, dl3a, dl4, dl4a]

l5, dl5 = FrequencyLayer(LATENT_SIZE, 150)
l5a, dl5a = ActivationLayer(150, BATCH_SIZE, sigmoid, d_sigmoid)
# l5bn, dl5bn = BatchNormLayer(150, BATCH_SIZE)
l6, dl6 = FrequencyLayer(150, 300)
l6a, dl6a = ActivationLayer(300, BATCH_SIZE, sigmoid, d_sigmoid)
# l6bn, dl6bn = BatchNormLayer(300, BATCH_SIZE)
l7, dl7 = FrequencyLayer(300, 500)
l7a, dl7a = ActivationLayer(500, BATCH_SIZE, sigmoid, d_sigmoid)
# l7bn, dl7bn = BatchNormLayer(500, BATCH_SIZE)
l8, dl8 = FrequencyLayer(500, INPUT_SIZE)
l8a, dl8a = ActivationLayer(INPUT_SIZE, BATCH_SIZE, sigmoid, d_sigmoid)
# l8bn, dl8bn = BatchNormLayer(INPUT_SIZE, BATCH_SIZE)
# full -> decoder_net_f, decoder_net_b = [l5, l5a, l5bn, l6, l6a, l6bn, l7, l7a, l7bn, l8, l8a, l8bn], [dl5, dl5a, dl5bn, dl6, dl6a, dl6bn, dl7, dl7a, dl7bn, dl8, dl8a, dl8bn]
decoder_net_f, decoder_net_b = [l5, l5a, l6, l6a, l7, l7a, l8, l8a], [dl5, dl5a, dl6, dl6a, dl7, dl7a, dl8, dl8a]


# loss function + derivative
mse(x, y) = sum((x.-y).^2) / length(x)
Δmse(x, y) = 2 .* (x .- y) ./ length(x)




# ::: TRAINING
runtime_plotting = true
plot_intervall = 5
train_metric = [NaN for _ in 1:TRAIN_EPISODES]; println("---"); @time for e in 1:TRAIN_EPISODES

    curr_batch = get_MNIST_batch(train_x, BATCH_SIZE)

    # println("layer 8bn Δ magnitude: ", sum(l8bn.dlocaly_dlocalX)/length(l8bn.dlocaly_dlocalX))
    # println("layer 8 Δ magnitude: ", sum(l8.dlocaly_dlocalX)/length(l8.dlocaly_dlocalX))
    # println("layer 7bn Δ magnitude: ", sum(l7bn.dlocaly_dlocalX)/length(l7bn.dlocaly_dlocalX))
    # println("")

    z = feed_forward!(curr_batch, encoder_net_f)
    y = feed_forward!(z, decoder_net_f)



    loss = mse(curr_batch, y)
    dloss = Δmse(curr_batch, y)


    dz = feed_backwards!(dloss, decoder_net_b, lr=LR)
    _ = feed_backwards!(dz, encoder_net_b, lr=LR)

    println("episode $e : loss = $loss")
    train_metric[e] = loss

    if runtime_plotting && e % plot_intervall == 0
        p = plot(train_metric)
        display(p)
    end
end



plot(train_metric, xlabel="training iterations", label="MSE (lr=$LR, batch size=$BATCH_SIZE)")
