using Flux, CUDA, Statistics, ProgressMeter

#Load labels and images
@info("load training data...")
train_labels =
train_imgs = 


# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

@info("creating mini batches...")
batch_size = 128
mb_idxs = partition(1:length(train_imgs), batch_size)
train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

# Prepare test set as one giant minibatch:
test_imgs = 
test_labels = 
test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))

# CNN model 
model = Chain(
    #First convolution
    Conv((3,3), 1=>16, pad(0,0),leakyrelu),
    x-> maxpool(x,(6,6), stride = 2 ),
    
    #Second convolution TODO: check padding because of 253x253
    Conv((3,3), 16=>16, pad(1,1), leakyrelu),
    x-> maxpool(x,(7,7),stride = 2),

    #Third convolution
    Conv((3,3), 16=>16, pad(1,1), leakyrelu),
    x-> maxpool(x,(6,6),stride = 2 ),

    #Fourth convolution
    Conv((3,3), 16=>16, pad(1,1), leakyrelu),
    x-> maxpool(x,(6,6),stride = 2),

    #Fifth convolution TODO: check if GlobalMeanPool is the same as global average pooling 
    Conv((3,3), 16=>16, pad(1,1), leakyrelu),
    x-> GlobalMeanPool(x,(28,28)),
    
    #Fully connected layer 
    Dense(16=>5, softmax)

    )|> gpu #moves the model to GPU if available

#set model.bias and model.weight


#initiate loss function
loss(model, x, y) = mean(abs2.(model(x) .- y));#standard gradient descent

#add optimizer
optimizer

#create data batches for training (not sure how to do it rn )



#training loop
@ showprogress for epoch in 1:200
    train!(loss, predict, data , optimizer)

#TODO: testing of data is needed 