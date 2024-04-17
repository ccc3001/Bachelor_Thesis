using Flux, CUDA, Statistics, ProgressMeter
using HDF5
using ImageMetadata
using JSON
using Flux
using JLD2
using OneHotArrays
#Load labels and images
@info("load training data...")

global H5_Dict = Dict{Int,String}
global n = 0
global scan_types =[0,0,0,0,0]
global scan_pos = [[],[],[],[],[]]

for (root, dirs, files) in walkdir(pwd())

    for file in files

        if endswith(file, ".h5")
            global scan_pos
            global n = n + 1
            tmp_Dict = Dict(n => file)#need to find a better way to store where the file is located 
            global H5_Dict = merge!(H5_Dict,tmp_Dict)
            x = h5open(file, "r") do h5
                read(h5, string(111))
            end
            #stores location and number of type n scans 
            global scan_types = scan_types.+x
            push!(scan_pos[ onecold(x,[1,2,3,4,5])],n) 
            
        end
    end
    
end

function split_data(scan_types, scan_pos, percent_test_size )
    training_data = []
    test_data = []
    for i in 1:5
        
        for j in 1:round(scan_types[i]*percent_test_size)
            rand_nr = Int(rand(1:(scan_types[i]+1-j)))
            push!(training_data , scan_pos[i][rand_nr])
            deleteat!(scan_pos[i], rand_nr)
        end
        append!(test_data, scan_pos[i]) 
    end

    return(training_data, test_data)
end


#TODO: find a way to combine all the mini batches 
function make_minibatch(batch_size, dict, splitted_data_array)
    dict_size = length(dict)
    X_batch=[]
    Y_batch=[]
    for i in 1:batch_size
            rand_nr=splitted_data_array[rand(1:length(splitted_data_array))]
            X =  h5open(dict[rand_nr], "r") do file
                read(file, string(rand([1,15,29,43,57,70,83,97,110])))
            end
            Y =  h5open(dict[rand_nr], "r") do file
                read(file, string(111))
            end
        
        # chance to add gaussian noise
        if rand() >= 0.9
            X = add_gauss(X,0.1)
        end
        #chance to rotate image
        if rand() >=0.25

            rotmatrix = recenter(RotMatrix((pi/2)*rand(1:3)), center(X))
            X = warp(X , rotmatrix)
        end
        #TODO: chance to erase parts of the image 
        if rand() >=0.98
            #still needs to be done 
        end
        #push the x and y values on the arrays
        push!(X_batch,X)
        push!(Y_batch,Y)

    end

    #TODO: Normalize batch 

    return (X_batch, Y_batch)
end

training_data , test_data = split_data(scan_types,scan_pos, 0.1)
#print(training_data,test_data)


# Bundle images together with labels and group into minibatchess
#function make_minibatch(test_x, test_y, idxs)
#    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
#    for i in 1:length(idxs)
#        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
#    end
#    Y_batch = onehotbatch(Y[idxs], 0:9)
#    return (X_batch, Y_batch)
#end

@info("creating mini batches...")
batch_size = 128
mb_idxs = 200  #partition(1:length(train_imgs), batch_size)
train_set = [make_minibatch(batch_size,H5_Dict,training_data) for i in mb_idxs]    #[make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]
@info("training set currated")
## Prepare test set as one giant minibatch:
test_set = [make_minibatch(batch_size,H5_Dict,test_data) for i in mb_idxs] 
# CNN model 
model = Chain(
    #First convolution
    Conv((3,3), 1=>16,leakyrelu),
    x-> maxpool(x,(2,2),stride = 2 ),
    
    #Second convolution 
    Conv((3,3), 16=>16, leakyrelu),
    x-> maxpool(x,(2,2),stride = 2),

    #Third convolution 
    Conv((3,3), 16=>16, leakyrelu),
    x-> maxpool(x,(2,2),stride = 2),

    #Fourth convolution
    Conv((3,3), 16=>16, leakyrelu),
    x-> maxpool(x,(2,2), stride =2),

    #Fifth convolution 
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
    #x_aug = x .+ 0.1x0*gpu(randn(eltype(x), size(x)))

    y_hat = model(x)#if random noise is added x need to be changed to x_aug
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
    train_set = make_minibatch(batch_size= 128 , dict, splitted_data_array)
    Flux.train!(loss, params(model), train_set, opt)

    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    
    # If our accuracy is good enough, quit out.
    if acc >= 0.95
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
        BSON.@save "motion_artefact_detection.bson" model epoch_idx acc
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




###### Training Pipeline ######
y1hat = model(example) # to check weather the system works 
sum(softmax(y1hat);dims=1)
#TODO: still written for mnist
@show hcat(Flux.onecold(y1hat, 0:9),Flux.onecold(y1,0:9))

using Statistics: mean  # standard library

function loss_and_accuracy(model, data::MNIST=test_data)
    (x,y) = only(loader(data; batchsize=length(data)))  # make one big batch
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    (; loss, acc, split=data.split)  # return a NamedTuple
end


@show loss_and_accuracy(lenet); 


#===== TRAINING =====#

# Let's collect some hyper-parameters in a NamedTuple, just to write them in one place.
# Global variables are fine -- we won't access this from inside any fast loops.

settings = (;
    eta = 3e-4,     # learning rate
    lambda = 1e-2,  # for weight decay
    batchsize = 128,
    epochs = 10,
)
train_log = []

# Initialise the storage needed for the optimiser:

opt_rule = OptimiserChain(WeightDecay(settings.lambda), Adam(settings.eta))
opt_state = Flux.setup(opt_rule, lenet);

for epoch in 1:settings.epochs
    # @time will show a much longer time for the first epoch, due to compilation
    @time for (x,y) in loader(batchsize=settings.batchsize)
        grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), lenet)
        Flux.update!(opt_state, lenet, grads[1])
    end

    # Logging & saving, but not on every epoch
    if epoch % 2 == 1
        loss, acc, _ = loss_and_accuracy(lenet)
        test_loss, test_acc, _ = loss_and_accuracy(lenet, test_data)
        @info "logging:" epoch acc test_acc
        nt = (; epoch, loss, acc, test_loss, test_acc)  # make a NamedTuple
        push!(train_log, nt)
    end
    if epoch % 5 == 0
        JLD2.jldsave(filename; lenet_state = Flux.state(lenet) |> cpu)
        println("saved to ", filename, " after ", epoch, " epochs")
    end
end

@show train_log;

# We can re-run the quick sanity-check of predictions:
y1hat = lenet(x1)
@show hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))
