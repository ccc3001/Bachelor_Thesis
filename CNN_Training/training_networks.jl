#librarys
using Flux, CUDA, Statistics, ProgressBars
using HDF5
using ImageMetadata
using JSON
using Flux
using JLD2
using OneHotArrays
using BSON
using Images
using Flux.Data: DataLoader
using Noise
using ImageTransformations
using CoordinateTransformations
using Rotations
using Plots

#global variables 
global H5_Dict = Dict{Int,String}
global n = 0
global scan_types =[0,0,0,0,0]
global scan_pos = [[],[],[],[],[]]
global number_of_minibatches = 50
output_size = 1


##Load labels and images
#scans the folder that test_cnn.jl is in, for files with the ending .h5 and checks what the type of the scan is 
# it changes the global H5_Dict variable wich stors the location of all h5 files, 
#it changes the scan_types var which saves how many scans of which type the are stored in H5_Dict
#it changes the scan_pos var which saves which positions of the H5_Dict save which type of scan 
@info("creating dictionary of .h5 files...")
filter_for_radius = true
for (root, dirs, files) in walkdir(pwd())

    for file in files
        
        if endswith(file, ".h5")
            global scan_pos
            global n = n + 1
            if filter_for_radius
                y = h5open(file, "r") do h5
                    read(h5, string(112))
                end
                if y == 1 
                    x = h5open(file, "r") do h5
                        read(h5, string(111))
                    end
                    tmp_Dict = Dict(n => root*"\\"*file)
                    global H5_Dict = merge!(H5_Dict,tmp_Dict)
                    global scan_types = scan_types.+x
                    push!(scan_pos[ onecold(x,[1,2,3,4,5])],n) 
                end
            else
                x = h5open(file, "r") do h5
                    read(h5, string(111))
                end
                tmp_Dict = Dict(n => root*"\\"*file)
                global H5_Dict = merge!(H5_Dict,tmp_Dict)
                global scan_types = scan_types.+x
                push!(scan_pos[ onecold(x,[1,2,3,4,5])],n)     
            end 
           
            
        end
    end
    
end



#calculates the xlogy which is needed for the crossentropy loss 
function xlogy(x, y)
    result = x * log(y)
    ifelse(iszero(x), zero(result), result)
end


#splitts the scan_pos variable in two sets which can then be used to generate a test and training set
# hte percent_test_size variable sets the percentage size of the training set compared to all the scans
function split_data(scan_types, scan_pos, percent_test_size )
    training_data = []
    test_data = []
    test_scan_types=[[],[],[],[],[]]
    training_scan_types=[[],[],[],[],[]]
    counter_a=1
    counter_b=1
    for i in 1:5
        
        for j in 1:round(scan_types[i]*percent_test_size)
            rand_nr = Int(rand(1:(scan_types[i]+1-j)))
            push!(test_data , scan_pos[i][rand_nr])
            deleteat!(scan_pos[i], rand_nr)
            push!(training_scan_types[i],counter_a)
            counter_a +=1
        end
        append!(training_data, scan_pos[i])
        for k in 1:length(scan_pos[i])
            push!(test_scan_types[i],counter_b)
            counter_b += 1
        end 
        
    end
    return training_data, test_data , training_scan_types, test_scan_types
end

# this function is used to create augmented test or training data sets
# it is modular and therefore the application of rotation, noise, image snippets, number of repertitions can be set
# also the output size of the network can be changed 
function make_test_batch(dict, splitted_data_array, output_size = 5,rotation = false, noise = false,use_image_snipets= false, nr_of_reps = 1, rot_val = 10, shuffle_bool =true)
    
    #image layers that can be extracted from the h5 dataset
    image_layers = [1,15,29,43,57,70,83,97,110]
    
    len_im_layers = length(image_layers)
    dict_size= length(dict)
    splitted_array_size = length(splitted_data_array)
    @info("the test image size equals: $(splitted_array_size)")

    #initalizes 2 empty arrays with the needed size for the input and output data  
    X_batch=zeros(512,512,1,splitted_array_size*len_im_layers*nr_of_reps)
    if output_size == 5 
        Y_batch=zeros(5,splitted_array_size*len_im_layers*nr_of_reps)

    elseif output_size ==1
        Y_batch=zeros(1,splitted_array_size*len_im_layers*nr_of_reps)
        
    elseif output_size ==2
        Y_batch=zeros(2,splitted_array_size*len_im_layers*nr_of_reps)  
    else 
        print("error outputsize not supported")
        return 0 
    end
    

    # generates the augmented data and saves it into the initalized arrays 
    for i in 1:splitted_array_size
        
            for j in 1:len_im_layers
                for tmp in 1:nr_of_reps
                    k = image_layers[j]
                    h5open(dict[splitted_data_array[i]],"r" ) do file
                        Y = read(file, string(111))
                        if output_size == 5
                            Y_batch[:,((i-1)*len_im_layers*nr_of_reps+(j-1)*nr_of_reps+nr_of_reps)]=Float32.(Y)
                        elseif output_size == 1
                            Y_batch[((i-1)*len_im_layers*nr_of_reps+(j-1)*nr_of_reps+tmp)]=onecold(Y,[0,0,0,1,1])
                        elseif output_size == 2
                            Y_batch[:,((i-1)*len_im_layers*nr_of_reps+(j-1)*nr_of_reps+nr_of_reps)]=onehot(onecold(Y,[0,0,0,1,1]),[0,1])

                        else
                            print("error outputsize not supported")
                            return 0 
                        end 
                    end
                end
                h5open(dict[splitted_data_array[i]],"r" ) do file
                    k = image_layers[j]
                    X_tmp = read(file, string(k))
                    for tmp in 1:nr_of_reps
                        X=X_tmp

                        #rotation of the image 
                        if rotation

                            ## this part can be used if we want to rotate the image by a costum value from 0 to 359 degrees 
                            #rot = rand(0:rot_val)
                            #trfm = recenter(RotMatrix(rot*pi/360), [768.5,768.5])
                            #X=warp(X, trfm)
                            #la = 1536
                            #alpha = mod(rot*pi/360,2*pi/4)
                            #lb = la/(sin(alpha)+cos(alpha))
                            #a =Int.(ceil.((1536-lb)/2))
                            #X=X[(1+a+5):(1536-a-5),(1+a+5):(1536-a-5)]
                             #flipping of the image 
                            

                            # flipps the image horizontal or vertical randomly 
                            k = rand()
                            if k > 0.66666
                                reverse(X,dims=1)
                            elseif k >0.33333
                                reverse(X,dims=2)
                            end   

                            #rotation of the image by 90, 180 or 270 randomly 
                            l = rand()
                            if l >=0.25
                                if l >= 0.75
                                    X= rotr90(X) 
                                elseif l >= 0.5
                                    X= rotr90(X)
                                    X= rotr90(X)
                                else 
                                    X= rotr90(X)
                                    X= rotr90(X)
                                    X= rotr90(X)
                                end
                            end
                        end


                        # this part cuts image snipets from the original image and cropps them to the needed inupt size of the network  
                        if use_image_snipets
                            image_size = size(X)
                
                            # since the main information of the image need to be keept the center image of 640x640 pixels needs to be included in the snippet
                            #crop_to_size sets the size of the image snippet 
                            crop_to_size = rand(640:image_size[1])
                            
                            #sets the maximum value the snippet can be moved horizontal or verticaly this is just given if the radius is positive
                            movement_radius = min(Int(floor.((crop_to_size-640)/2)),Int.(floor.((image_size[1]-crop_to_size)/2 )))
                            if movement_radius > 0
    	                        horizontal_movement = rand(-movement_radius:movement_radius)
	                            vertical_movement = rand(-movement_radius:movement_radius)
                            else
	                            horizontal_movement = 0
	                            vertical_movement = 0 
                            end
                            X = X[Int.(image_size[1]/2-floor.(crop_to_size/2)+horizontal_movement+1):Int.(image_size[1]/2+floor.(crop_to_size/2)+horizontal_movement),
                            Int.(image_size[1]/2-floor.(crop_to_size/2)+vertical_movement+1):Int.(image_size[1]/2+floor.(crop_to_size/2)+vertical_movement)]
                            

                            
                        end
                        #resizes the image to the input size of the network 
                        X= imresize(X,(512,512))

                        #adds gaussian noise to the image 
                        if noise && rand() >= 0.01
                            X = add_gauss(X,0.01,0.0)
                        end

                        
                        X_batch[:,:,1,((i-1)*len_im_layers*nr_of_reps+(j-1)*nr_of_reps+tmp)]=Float32.(X)
                    end    
                end
                
            end 
    end 
    X_batch = reshape(X_batch,(512,512,1,splitted_array_size*len_im_layers*nr_of_reps))# this was changed for making the test batch gpu viable
    Y_batch = reshape(Y_batch,(output_size,splitted_array_size*len_im_layers*nr_of_reps))# this was changed for making the test batch gpu viable
    println(size(X_batch),size(Y_batch))

    # shuffels the data,seperates it in batches of 16 and loads it to the gpu if possible 
    return DataLoader((X_batch,Y_batch)|> gpu,batchsize=16,shuffle=shuffle_bool)
end



# calculates the weightes loss of the network regarding the given x and y values 
function weighted_loss(x,y)
    
    y_ = model(x)
    if size(y,1) ==1
        number_of_classes = 2
        weights_for_class_i= [0 0]
        number_of_examples_in_class_i = [(size(y,2)-sum(y)) sum(y)]

        if number_of_examples_in_class_i[1] == 0 || number_of_examples_in_class_i[2] == 0
            loss_var = mean(((y_.-y).^2))
        else
            #calculates the weights for y = 0 and 1 and generates a vector that can be applied to the loss function 
            weights_for_class_i = size(y,2)./(number_of_examples_in_class_i'.*number_of_classes)
            y0= weights_for_class_i[1]*(0 .== y)
            y1 =weights_for_class_i[2]*y
            
            y01 = y0-(-y1)
            y01 = y01
            #loss_var = mean(y01.*((y_.-y).^2))
            #loss_var =mean(-y01.*(y .* log.(y_)+(1 .- y ).*log.(1 .- y_)))
            
            loss_var = mean(-y01 .* (xlogy.(y,y_)+xlogy.((1 .- y),(1 .- y_))))
            if isinf(loss_var) 
                loss_var = 9999
            end
            if isnan(loss_var) || isinf(loss_var) || isinf(-loss_var)
                println(y)
                println(y_)
                println(xlogy.(y,y_))
                println(xlogy.((1 .- y),(1 .- y_)))
            end

        end

        #if isnan(loss_var)
        #    y_addapted = y_ |> cpu
        #    println("test 1")
        #    for i in 1:length(y_addapted)
        #        print("test_1_1")
        #        if y_addapted[i] == 0
        #            println("test 2 ")
        #            y_addapted[i] = 0.000000001
        #        elseif y_addapted[i] == 1
        #            println("test 2_1 ")
        #            y_addapted[i] = 0.9999999
        #            
        #        end 
        #    end
        #    y_addapted = y_addapted |> gpu
        #    t = (y .* log.(y_addapted)+(1 .- y ).*log.(1 .- y_addapted))|>cpu
        #    println(-y01 .* t)
        #    loss_var =mean(-y01 .* t)
        #    return loss_var
        #else
        return loss_var     #-sum(y01.* xlogy.(y,y_))/sum((number_of_examples_in_class_i'.*number_of_classes))
        #end
    else
        number_of_classes= size(y,1)
        #weights_for_class_i=[size(y,2)]
        
        number_of_examples_in_class_i = sum(y;dims=2)
        weights_for_class_i = length(y)./(number_of_examples_in_class_i'.*number_of_classes)
        loss_var = Flux.mean(.-sum(weights_for_class_i' .* xlogy.(y,y_)))#/sum((number_of_examples_in_class_i'.*number_of_classes))
        #println(loss_var)
        return loss_var #; dims = number_of_classes
        #return Flux.crossentropy(y,y;dims = number_of_classes)    
    end
end


function f1_score_precission_recall_accuracy(model, test_log, test_model_data ,output_size = 5, info_feature = false ,test_set_bool = false)
    
    
    y_=[]
    y =[]
    for (x_tmp,y_tmp) in test_model_data
        result = model(x_tmp)
        result = result |> cpu
        CUDA.@allowscalar append!(y_,result)
        CUDA.@allowscalar append!(y,y_tmp)
    end
    if test_set_bool
        global array
        global scan_type_probability
        y__rounded = round.(y_)
        correct_scans = Float32.(y__rounded .== y)
        correct_detected_scantypes=array .* correct_scans'
        for i in 1:5
            append!(scan_type_probability[i],sum(correct_detected_scantypes.==i)/sum(array.==i))
        end
    end
    f1_matrix = zeros(output_size,output_size)
    #y_ = zeros(1,length(x))

    if size(y,2) ==1
        #println(y)
        #println(y_)
        number_of_classes = 2
        weights_for_class_i= [0 0]
        number_of_examples_in_class_i = [(size(y,1)-sum(y)) sum(y)]
        weights_for_class_i = size(y,1)./(number_of_examples_in_class_i'.*number_of_classes)
        y0= weights_for_class_i[1]*(0 .== y)
        y1 =weights_for_class_i[2]*y
        y01 = y0-(-y1)
        #println(y01)
        #loss_var = Flux.mean(y01.*(y_.-y).^2)#-sum(y01.* xlogy.(y,y_))/sum((number_of_examples_in_class_i'.*number_of_classes))
        #loss_var =mean(-y01.*(y .* log.(y_)+(1 .- y ).*log.(1 .- y_)))#sigmoid_fast(
        loss_var = mean(-y01.*(xlogy.(y,y_)+xlogy.(1 .- y,1 .- y_)))
    else
        number_of_classes= size(y,1)
        #weights_for_class_i=[size(y,2)]
        
        number_of_examples_in_class_i = sum(y;dims=2)
        weights_for_class_i = length(y)./(number_of_examples_in_class_i'.*number_of_classes)
        loss_var = mean(.-sum(weights_for_class_i' .* xlogy.(y,y_)))#/sum((number_of_examples_in_class_i'.*number_of_classes))
    end
    #Flux.crossentropy(y_,y;dims = output_size)
    
    if size(y_,2)>=2
        y = Int.(onecold(y))
        y_=Int.(onecold(y_)) 
        for i in 1:length(y_)
            f1_matrix[y[i],y_[i]]=f1_matrix[y[i],y_[i]]+ 1
        end
        precision = zeros(output_size,1)
        recall = zeros(output_size,1)
        F1_Score = zeros(output_size,1)
        accuracy_array =zeros(output_size,1)
        accuracy=0
        for i in 1:output_size
            recall[i]= f1_matrix[i,i]/sum(f1_matrix[:,i])
            precision[i] = f1_matrix[i,i]/sum(f1_matrix[i,:])
            F1_Score[i] =2*precision[i]*recall[i]/(precision[i]+recall[i])
            accuracy_array[i] =f1_matrix[i,i]/(sum(f1_matrix[i,:])+sum(f1_matrix[:,i])-f1_matrix[i,i])
            accuracy+= f1_matrix[i,i]/length(y_)
            
        end
        #p_e= ((true_positives+false_positives)*(true_negatives+false_negatives)+(true_positives+false_negatives)*(false_positives+true_negatives) )/(true_negatives +true_positives+false_positives+false_negatives)^2
        cohens_kappa = 0#(accuracy-p_e)/(1-p_e) 
        confusion_matrix = 0 # [true_positives false_negatives; false_positives false_negatives]
        push!(test_log[1],F1_Score)
        push!(test_log[2],precision)
        push!(test_log[3], recall)
        push!(test_log[4],accuracy)
        push!(test_log[5], loss_var)
        push!(test_log[6], cohens_kappa)
        push!(test_log[7], confusion_matrix)
        
        if info_feature
            println("\n")
            @info("\nTest accuracy: $(test_log[4][length(test_log[4])] ) Test loss:$(test_log[5][length(test_log[5])])")
            @info("\n F1-Score:$(test_log[1][length(test_log[1])])")
            @info("\n Precission:$(test_log[2][length(test_log[2])])")
            @info("\n Recall:$(test_log[3][length(test_log[3])])")
            @info("\n cohens_kappa:$(test_log[6][length(test_log[6])])")
            @info("\n confusion matrix: $(test_log[7][length(test_log[7])])")
            #@info("\n scan_type_probability: $(scan_type_probability[length(scan_type_probability[1])][:])")
        end
        return test_log 

    else
        y_=round.(y_)
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        for i in 1:length(y) 
            if y[i] == 1 && y_[i] == 1
                true_positives +=1 
            elseif y[i] == 0 && y_[i] == 1
                false_positives +=1 
            elseif y[i] == 1 && y_[i] == 0
                false_negatives +=1 
            else 
                true_negatives +=1
            end
        end
        print("\n"*string(true_positives) *" "* string(false_negatives) * " "* string(false_positives)*" "*string(true_negatives)*" "*string(length(y)))
        precision =true_positives/(true_positives+false_positives)
        recall= true_positives/(true_positives+false_negatives)
        F1_Score= 2*precision*recall/(precision+recall)
        accuracy = sum( y .== y_)/length(y)
        
        p_e= ((true_positives+false_negatives)*(true_positives+false_positives)+(true_negatives+false_positives)*(true_negatives+false_negatives) )/(true_negatives +true_positives+false_positives+false_negatives)^2
        cohens_kappa = (accuracy-p_e)/(1-p_e)
        confusion_matrix = [true_positives false_negatives; false_positives true_negatives]
        push!(test_log[1],F1_Score)
        push!(test_log[2],precision)
        push!(test_log[3], recall)
        push!(test_log[4],accuracy)
        push!(test_log[5], loss_var)
        push!(test_log[6], cohens_kappa)
        push!(test_log[7], confusion_matrix)

        if info_feature
            @info("\nTest accuracy: $(test_log[4][length(test_log[4])] ) Test loss:$(test_log[5][length(test_log[5])])")
            @info("\n F1-Score:$(test_log[1][length(test_log[1])])")
            @info("\n Precission:$(test_log[2][length(test_log[2])])")
            @info("\n Recall:$(test_log[3][length(test_log[3])])")
            @info("\n cohens_kappa:$(test_log[6][length(test_log[6])])")
            @info("\n confusion matrix: $(test_log[7][length(test_log[7])])")
            #@info("\n scan_type_probability: $(scan_type_probability[:][length(scan_type_probability[1])])")
        end

        return test_log
    end
    
end

function make_minibatch(size_train_set, dict, splitted_data_array,scan_types,output_size = 5)
    dict_size = length(dict)
    X_batch=zeros(512,512,1,size_train_set)
    if output_size == 5 
        Y_batch=zeros(5,size_train_set)
    elseif output_size == 1
        Y_batch=zeros(1,size_train_set) 
    elseif output_size == 2
        Y_batch=zeros(2,size_train_set) 
    else 
        print("error outputsize not supported")
        return 0 
    end
 
    for i in 1:size_train_set
            custom_average_element_types = false
            if custom_average_element_types
                element_type =rand([1,2,3,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5])#,
                rand_nr= scan_types[element_type][rand(1:length(scan_types[element_type]))]
                rand_nr = splitted_data_array[rand_nr]
            else
                rand_nr=splitted_data_array[rand(1:length(splitted_data_array))]
            end
                X =  h5open(dict[rand_nr], "r") do file
                read(file, string(rand([1,15,29,43,57,70,83,97,110])))
            end
            #X = imresize(X,(512,512))
            X = X[257:768,257:768]
            Y=  h5open(dict[rand_nr], "r") do file
                read(file, string(111))
            end
            if output_size == 5
                Y_batch[:,i]= Y
            elseif output_size == 1
                Y_batch[i]= onecold(Y,[0,0,0,1,1])
            elseif output_size == 2
                Y_batch[:,i]= onehot(onecold(Y,[0,0,0,1,1]),[0,1])
            else
                print("error outputsize not supported")
                return 0 
            end
        # chance to add gaussian noise
        if rand() >= 0.25
            X = add_gauss(X,0.01,0.0)
        end
        k= rand()
        if k >=0.25
            if k >= 0.75
                X = reverse(X, dims=1) 
            elseif k >= 0.5
                X =reverse(X, dims=2)
            else 
                X =reverse(X, dims=1)
                X =reverse(X, dims=2)
            end
        
        end
        X_batch[:,:,1,i]=Float32.(X)
    end
#TODO: find a way to combine all the mini batches 
    #TODO: Normalize batch not sure if this is needed 
    return DataLoader((X_batch, Y_batch)|>gpu, batchsize=16  , shuffle=true)
end

function save_models(test_log,training_log)
    
    loss_plot = plot(1:length(test_log[5]),[test_log[5],training_log[5]], label = ["test loss" "training loss"],title="Loss Plot")
    xlabel!(loss_plot,"epoch")
    ylabel!(loss_plot,"loss")
    savefig(loss_plot,"$(model_name)\\loss_png.png")

    accuracy_plot = plot(1:length(test_log[4]),[test_log[4],training_log[4]],label = ["test accuracy" "training accuracy"],title="Accuracy Plot")
    xlabel!(accuracy_plot,"epoch")
    ylabel!(accuracy_plot,"accuracy")
    savefig(accuracy_plot,"$(model_name)\\accuracy_png.png")

    cohens_kappa_plot = plot(1:length(test_log[6]),[test_log[6],training_log[6]],label = ["test cohen's kappa" "training cohen's kappa"],title="Cohen's Kappa Plot")
    xlabel!(cohens_kappa_plot,"epoch")
    ylabel!(cohens_kappa_plot,"cohen's kappa")
    savefig(cohens_kappa_plot,"$(model_name)\\cohens_kappa_png.png")

    f1_score_plot = plot(1:length(test_log[1]),[test_log[1],training_log[1]],label = ["test F1-Score" "training F1_score"],title="F1-Score Plot")
    xlabel!(f1_score_plot,"epoch")
    ylabel!(f1_score_plot,"F1-score")
    savefig(f1_score_plot,"$(model_name)\\f1_score_png.png")
    
    println("f1:$(test_log[1])")
    println("prec:$(test_log[2])")
    println("rec:$(test_log[3])")
    println("acc:$(test_log[4])")
    println("loss:$(test_log[5])")
    println("cohens_kappa:$(test_log[6])")
    println("confusion matrix:$(test_log[7])")
    return 1 
end 




training_data, test_data , test_scan_types, training_scan_types =  split_data(scan_types,scan_pos,0.25)

###creating array with actuall scan types for the test set so that i can check which scans are detected the best 
global array = zeros(1,length(test_data)*9)
global counter = 0
for i in 1:5
    for j in 1:length(test_scan_types[i])
        for k in 1:9
            global counter= counter + 1  
            global array[counter] = i
        end
    end
end



#### First model 

global test_log =[[],[],[],[],[],[],[]]
global training_log = [[],[],[],[],[],[],[]]
global scan_type_probability = [[],[],[],[],[]]
opt = Adam(0.001)
model_name = "final_run_6_2"
model = Chain(
    #First convolution
    BatchNorm(1),
    Conv((3,3), 1=>16,leakyrelu),
    x-> maxpool(x,(2,2),stride = 2 ),
    #Second convolution 
    Conv((3,3), 16=>16,leakyrelu),
    x-> maxpool(x,(2,2),stride = 2 ),

    #Third convolution 
    Conv((3,3), 16=>16,leakyrelu),
    x-> maxpool(x,(2,2),stride = 2),

    #Fourth convolution
    Conv((3,3), 16=>16,leakyrelu),
    x-> maxpool(x,(2,2),stride = 2 ),

    #Fifth convolution 
    Conv((3,3), 16=>16,leakyrelu),
    #AlphaDropout(0.5),
    Flux.GlobalMeanPool(),
    Flux.flatten,
    #Fully connected layer
    Dense(16=>1),
    sigmoid_fast
    #TODO: softmax needs to be added to last dense layer

    )|> gpu  
model=gpu(model)
ps = Flux.params(model)
global best_acc = 0 
loss(x, y) = Flux.Losses.mse(model(x), y)
epochs = 200
#test_model= make_test_batch(H5_Dict,test_data,1)
test_model= make_test_batch(H5_Dict,test_data,1,false, false,false,1,10,false)
#train_model = make_test_batch(H5_Dict, training_data, 1,false, false,false,1,10)
#train_model=make_test_batch(H5_Dict,training_data,1)
for i in 1:epochs
    global best_acc
    @info("Epoch: $(i)")
    #if !@isdefined best_acc 
    #    best_acc = 0 
    #    @info("best_acc set to zero")
    #end
    @info("creating training model ...")
    train_model = make_test_batch(H5_Dict, training_data, 1,true, true,true,1,10)
    @info("training model ...")
    Flux.testmode!(model , false)
    Flux.train!(weighted_loss,ps,train_model,opt)
    Flux.testmode!(model , true)
    @info("training done")

    @info("calculating accuracy loss and f1 score of test set ... ")
    global training_log = f1_score_precission_recall_accuracy(model,training_log, train_model, 1 , false )
    global test_log =f1_score_precission_recall_accuracy(model, test_log, test_model, 1  , true,true )
    acc = test_log[4][length(test_log[4])]
    if acc >=0.99
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
        if !isdir(model_name)
            mkdir(model_name)
        end
        BSON.@save "$(model_name)\\epoch_$(i)_acc_$(acc).bson" model
        best_acc = acc 
        
    end
    #weight decay ...
    #opt.eta = opt.eta*0.94 #for Adam
    #opt.os[1].eta = opt.os[1].eta*0.89 # 0.94 is if we want a * 0.3 every 20 epochs atm its *0.3 every 10 epochs 
end 

save_models(test_log,training_log)


JLD2.save_object("$(model_name)\\traininglog_epoch_model_$(model_name)_best_acc_$(best_acc).jld2",test_log)
JLD2.save_object("$(model_name)\\scan_type_probability.jld2",scan_type_probability)

