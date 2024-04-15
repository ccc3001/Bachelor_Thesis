using ScancoFiles
using Images
using Flux
using CUDA
using Scratch
using ImageMetadata
using JSON
using ProgressBars
using HDF5
using OpenCV
using Noise
using OneHotArrays
using Rotations
using CoordinateTransformations
using ImageTransformations




for (root, dirs, files) in walkdir(pwd())
    for file in files
        ## checks for .ISQ files and creates .h5 files to get the filesize down 
        if endswith(file , ".ISQ")
            img = loadISQ(file)
            h5open(joinpath(pwd(),  file[1:(length(file)-4)] * ".h5") ,"w") do h5
                for i in [1,15,29,43,57,70,83,97,110]
                    image =Float32.((ImageAxes.data(img[:,:,i])).-img["dataRange"][1])/(img["dataRange"][2]-img["dataRange"][1])
                    h5[string(i)]=image
                end
                img_datatype_array=[0,0,0,0,0]
                if Int(img["dataType"]) <= 5
                    img_datatype_array[Int(img["dataType"])] = 1;
                end
                h5[string(111)]= img_datatype_array
            end
            rm(joinpath(pwd(),file))
        end
        
    end
    
end


##goes through the foldern structure and creates a dictionary with all the location of  h5 files
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
###function that splits the data in a training and test set
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


### function to create minibatches (working)
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
