using Flux, CUDA, Statistics, ProgressMeter

#Load labels and images
@info("load training data...")

for i in readdir()
    if endswith(i,".h5") 
        train_imgs =[]
    if endswith(i,"json")
        image_metadata=JSON.parse(i)
        model_settings = get(image_metadata, "img_datatype_array", "")
        train_labels =[]
     
end

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
    x-> maxpool(x,(2,2),stride = 2 ),
    
    #Second convolution 
    Conv((3,3), 16=>16, pad(0,0), leakyrelu),
    x-> maxpool(x,(3,3),stride = 2),#TODO: Jana fragen ob das sin ergibt -> könnte sein das es maxpool(2,2) ist wenn ich 0.5 als wert haben darf 

    #Third convolution 
    Conv((3,3), 16=>16, pad(0,0), leakyrelu),
    x-> maxpool(x,(2,2),stride = 2),

    #Fourth convolution
    Conv((3,3), 16=>16, pad(0,0), leakyrelu),
    x-> maxpool(x,(2,2), stride =2),

    #Fifth convolution TODO: check if GlobalMeanPool is the same as global average pooling 
    Conv((3,3), 16=>16, pad(0,0), leakyrelu),
    x-> GlobalMeanPool(x),
    
    #Fully connected layer 
    Dense(16=>5, softmax)

    )|> gpu #moves the model to GPU if available

#set model.bias and model.weight -> not needed since it is initialized automatically 


#initiate loss function
loss(model, x, y) = mean(abs2.(model(x) .- y));#standard gradient descent

#add optimizer
learning_rate = 0.01
optimizer = Descent(learning_rate)

#create data batches for training (not sure how to do it rn )
train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)





#training loop
epochs=200
ps = Flux.params(model)

@ showprogress for epoch in 1:epochs
    for (x,y) in train_set
        
        device= gpu 
        x,y = device(x), device(y) #load data to device 
        gs = gradient(()-> loss(model, x , y),ps)   # compute gradient w.r.t parameters
        Flux.Optimise.update!(opt, ps, gs)    # update parameters
    end
    current_loss = loss(model())
end

            #train!(loss, predict, data , optimizer)

#TODO: testing of data is needed 