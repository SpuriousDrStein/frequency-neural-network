include("network_functions.jl")

using Plots
using MLDatasets
using Random
using CuArrays
using Images
# set random seed
Random.seed!(1301)


# ::: DATA
train_x, _ = MNIST.traindata()
train_x = permutedims(Array{Float32}(train_x), [3,2,1])
train_x = reshape(train_x, (length(train_x[:, 1, 1]), length(train_x[1, :, 1]) * length(train_x[1, :, 1])))
get_MNIST_input_batch(total_x, BS) = begin
    return [CuArray(total_x[rand(1:length(total_x[:, 1])), :]) for _ in 1:BS]
end


# ::: PARAMS
INPUT_SIZE = length(train_x[1, :, :])
LATENT_SIZE = 70
BATCH_SIZE = 1
LR = 0.04
TRAIN_EPISODES = 10000


l1, dl1 = FrequencyLayer(INPUT_SIZE, 500, BATCH_SIZE)
l1a, dl1a = ActivationLayer(500, BATCH_SIZE, sigmoid, Δsigmoid)
# l1bn, dl1bn = BatchNormLayer(500, BATCH_SIZE)
l2, dl2 = FrequencyLayer(500, 300, BATCH_SIZE)
l2a, dl2a = ActivationLayer(300, BATCH_SIZE, sigmoid, Δsigmoid)
# l2bn, dl2bn = BatchNormLayer(300, BATCH_SIZE)
l3, dl3 = FrequencyLayer(300, 150, BATCH_SIZE)
l3a, dl3a = ActivationLayer(150, BATCH_SIZE, sigmoid, Δsigmoid)
# l3bn, dl3bn = BatchNormLayer(150, BATCH_SIZE)
l4, dl4 = FrequencyLayer(150, LATENT_SIZE, BATCH_SIZE)
l4a, dl4a = ActivationLayer(LATENT_SIZE, BATCH_SIZE, sigmoid, Δsigmoid)
# l4bn, dl4bn = BatchNormLayer(LATENT_SIZE, BATCH_SIZE)
# full -> encoder_net_f, encoder_net_b = [l1, l1a, l1bn, l2, l2a, l2bn, l3, l3a, l3bn, l4, l4a, l4bn], [dl1, dl1a, dl1bn, dl2, dl2a, dl2bn, dl3, dl3a, dl3bn, dl4, dl4a, dl4bn]
encoder_net_f, encoder_net_b = [l1, l1a, l2, l2a, l3, l3a, l4, l4a], [dl1, dl1a, dl2, dl2a, dl3, dl3a, dl4, dl4a]
# nothing -> encoder_net_f, encoder_net_b = [l1, l2, l3, l4], [dl1, dl2, dl3, dl4]


l5, dl5 = FrequencyLayer(LATENT_SIZE, 150, BATCH_SIZE)
l5a, dl5a = ActivationLayer(150, BATCH_SIZE, sigmoid, Δsigmoid)
# l5bn, dl5bn = BatchNormLayer(150, BATCH_SIZE)
l6, dl6 = FrequencyLayer(150, 300, BATCH_SIZE)
l6a, dl6a = ActivationLayer(300, BATCH_SIZE, sigmoid, Δsigmoid)
# l6bn, dl6bn = BatchNormLayer(300, BATCH_SIZE)
l7, dl7 = FrequencyLayer(300, 500, BATCH_SIZE)
l7a, dl7a = ActivationLayer(500, BATCH_SIZE, sigmoid, Δsigmoid)
# l7bn, dl7bn = BatchNormLayer(500, BATCH_SIZE)
l8, dl8 = FrequencyLayer(500, INPUT_SIZE, BATCH_SIZE)
l8a, dl8a = ActivationLayer(INPUT_SIZE, BATCH_SIZE, sigmoid, Δsigmoid)
# l8bn, dl8bn = BatchNormLayer(INPUT_SIZE, BATCH_SIZE)
# full -> decoder_net_f, decoder_net_b = [l5, l5a, l5bn, l6, l6a, l6bn, l7, l7a, l7bn, l8, l8a, l8bn], [dl5, dl5a, dl5bn, dl6, dl6a, dl6bn, dl7, dl7a, dl7bn, dl8, dl8a, dl8bn]
decoder_net_f, decoder_net_b = [l5, l5a, l6, l6a, l7, l7a, l8, l8a], [dl5, dl5a, dl6, dl6a, dl7, dl7a, dl8, dl8a]
# final layer activation -> decoder_net_f, decoder_net_b = [l5, l6, l7, l8, l8a], [dl5, dl6, dl7, dl8, dl8a]





a = [CuArrays.rand(INPUT_SIZE) .* 0.5]
b = [CuArrays.rand(INPUT_SIZE) .* 0.5]

(a[1] .- b[1]) |> sum |> println

aa = feed_forward!(feed_forward!(a, encoder_net_f), decoder_net_f)
bb = feed_forward!(feed_forward!(b, encoder_net_f), decoder_net_f)

(aa[1] .- bb[1]) |> sum |> println

colorview(Gray, reshape(aa[1] |> Array, (28, 28)))
colorview(Gray, reshape(bb[1] |> Array, (28, 28)))















# training data
train_data_x = [get_MNIST_input_batch(train_x, BATCH_SIZE) for _ in 1:TRAIN_EPISODES]

# ::: TRAINING
runtime_plotting = true
plot_intervall = 10
FS_train_metric = [NaN for _ in 1:TRAIN_EPISODES]; println("---"); for e in 1:TRAIN_EPISODES

    z = feed_forward!(train_data_x[e], encoder_net_f)
    y = feed_forward!(z, decoder_net_f)


    loss = mse.(train_data_x[e], y)
    dloss = Δmse.(train_data_x[e], y)


    dz = feed_backwards!(dloss, decoder_net_b, lr=LR)
    _ = feed_backwards!(dz, encoder_net_b, lr=LR)

    println("episode $e : loss = $(sum(loss)/length(loss))")
    FS_train_metric[e] = sum(loss)/length(loss)

    if runtime_plotting && e % plot_intervall == 0
        p = plot(FS_train_metric)
        display(p)
    end
end



plot(train_metric, xlabel="training iterations", label="FS: MSE (lr=$LR, batch size=$BATCH_SIZE)")




# save test image for comparison
generate_images(number_of_images; path="/Users/BBM2H16AHO/Documents/programming/auto-encoder/images/dense_BN_relu") = begin
    for n in 1:number_of_images
        all_test_image_ref = get_MNIST_input_batch(train_x, BATCH_SIZE)
        all_test_images = feed_forward!(feed_forward!(all_test_image_ref, encoder_net_f), decoder_net_f)

        random_index = rand(1:BATCH_SIZE)
        test_image_ref = all_test_image_ref[random_index] |> Array
        test_image = all_test_images[random_index] |> Array


        test_image_ref_r = reshape(test_image_ref, (28, 28))
        test_image_r = reshape(test_image, (28, 28))
        save("$path/$(n)_reference.png", colorview(Gray, test_image_ref_r))
        save("$path/$(n)_reconstruction.png", colorview(Gray, test_image_r))
    end
end

generate_images(10, path="/Users/BBM2H16AHO/Documents/programming/frequency-neural-network/images")
