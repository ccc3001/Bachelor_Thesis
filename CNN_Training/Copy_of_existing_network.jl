using Flux, CUDA, Statistics, ProgressMeter
using HDF5
using ImageMetadata
using JSON
#Load labels and images
@info("load training data...")

test_x =[]
test_y =[]


for i in readdir("h5")
    j = i[1:(length(i)-3)]
    open(pwd() * "\\json\\" * j * ".json", "r") do f
        global dict
        dicttxt = readline(f)  # file information to string
        dict= JSON.parse(dicttxt)  # parse and transform data  
    end

    push!(test_y , dict["img_datatype_array"])
    push!(test_y , dict["img_datatype_array"])
    push!(test_y , dict["img_datatype_array"])



    c = h5open(pwd() * "\\h5\\" * i, "r") do file
        read(file)    
    end
    print(c["1"])
    push!(test_x,reshape(c["1"],(1,1536,1536)))
    push!(test_x,reshape(c["50"],(1,1536,1536)))
    push!(test_x,reshape(c["100"],(1,1536,1536)))

end
test_set = [(test_x,test_y)]
train_set = (test_x, test_y)


# Bundle images together with labels and group into minibatchess
#function make_minibatch(test_x, test_y, idxs)
#    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
#    for i in 1:length(idxs)
#        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
#    end
#    Y_batch = onehotbatch(Y[idxs], 0:9)
#    return (X_batch, Y_batch)
#end

#@info("creating mini batches...")
#batch_size = 128
#mb_idxs = partition(1:length(train_imgs), batch_size)
#train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

# Prepare test set as one giant minibatch:
#test_imgs = 
#test_labels = 
#test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))

# CNN model 
model = Chain(
    #First convolution
    Conv((3,3), 1=>16,leakyrelu),
    x-> maxpool(x,(2,2),stride = 2 ),
    
    #Second convolution 
    Conv((3,3), 16=>16, leakyrelu),
    x-> maxpool(x,(2,2),stride = 2),#TODO: Jana fragen ob das sin ergibt -> könnte sein das es maxpool(2,2) ist wenn ich 0.5 als wert haben darf 

    #Third convolution 
    Conv((3,3), 16=>16, leakyrelu),
    x-> maxpool(x,(2,2),stride = 2),

    #Fourth convolution
    Conv((3,3), 16=>16, leakyrelu),
    x-> maxpool(x,(2,2), stride =2),

    #Fifth convolution TODO: check if GlobalMeanPool is the same as global average pooling 
    Conv((3,3), 16=>16, leakyrelu),
    x-> GlobalMeanPool(x),
    
    #Fully connected layer 
    Dense(16=>5, softmax)

    )|> gpu #moves the model to GPU if available

#set model.bias and model.weight -> not needed since it is initialized automatically 

# Load model and datasets onto GPU, if enabled
train_set = gpu.(train_set)
test_set = gpu.(test_set)
model = gpu(model)

# Make sure our model is nicely precompiled before starting our training loop
model(train_set[1][1])

# `loss()` calculates the crossentropy loss between our prediction `y_hat`
# (calculated from `model(x)`) and the ground truth `y`.  We augment the data
# a bit, adding gaussian random noise to our image to make it more robust.
function loss(x, y)
    # We augment `x` a little bit here, adding in random noise
    x_aug = x .+ 0.1f0*gpu(randn(eltype(x), size(x)))

    y_hat = model(x_aug)
    return crossentropy(y_hat, y)
end
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# Train our model with the given training set using the ADAM optimizer and
# printing out performance against the test set as we go.
opt = ADAM(0.001)

@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in 1:100
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)

    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    
    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
        BSON.@save "mnist_conv.bson" model epoch_idx acc
        best_acc = acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 10
        @warn(" -> We're calling this converged.")
        break
    end
end
