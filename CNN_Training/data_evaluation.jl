using JLD2
using Plots
using EasyFit
using Statistics
using Glob
using BSON: @load
using Flux, CUDA
using HDF5
using OneHotArrays
using Images
using Flux.Data: DataLoader
## calculating the mean and standard deviation of the networks

#global H5_Dict = Dict{Int,String}
#global n = 0
#global scan_types =[0,0,0,0,0]
#global scan_pos = [[],[],[],[],[]]

#Y_batch = zeros(1,448)
#X_batch = zeros(512,512,1,448)
#Y_actual_type = zeros(1,448)
#global i_tmp = 0
#filter_for_radius = true
#for (root, dirs, files) in walkdir(pwd())
#
#    for file in files
#        
#        if endswith(file, ".h5")
#            global scan_pos
#            global n = n + 1
#            if filter_for_radius
#                y = h5open(file, "r") do h5
#                    read(h5, string(112))
#                end
#                if y == 1 
#                    x = h5open(file, "r") do h5
#                        read(h5, string(111))
#                    end
#                    x_X = h5open(file, "r") do h5
#                        read(h5, string(1))
#                    end
#
#                    global i_tmp = i_tmp+1
#                    X_batch[:,:,1,i_tmp] = imresize(x_X,(512,512))
#                    Y_batch[i_tmp] = onecold(x,[0,0,0,1,1])
#                    Y_actual_type[i_tmp] = onecold(x,[1,2,3,4,5])
#                    tmp_Dict = Dict(n => root*"\\"*file)
#                    global H5_Dict = merge!(H5_Dict,tmp_Dict)
#                    global scan_types = scan_types.+x
#                    push!(scan_pos[ onecold(x,[1,2,3,4,5])],n) 
#                end
#            end
#        end
#    end
#    
#end
#println("end")
#test_model = DataLoader((X_batch,Y_batch)|>gpu,batchsize=16,shuffle=false)


#println(Y_actual_type)


for (root, dirs, files) in walkdir(pwd())
    for dir in dirs
        i = readdir(glob"*.jld2", dir)
        println("test")
        for file in i
            if endswith(file,".jld2") 
                Dict= load(file)
                open("evaluated_data.txt", "w") do file1
                    test = Dict[:"single_stored_object"]
                    if length(test) == 7
                        #println(file)
                        write(file1,"$(file)\n")
                        f1,prec,rec,acc,loss,cohens_kappa,confusion_matrix = test
                        for (i,j) in [(f1,"f1"),(prec,"prec"),(rec,"rec"),(acc,"acc"),(loss,"loss"),(cohens_kappa,"cohens_kappa")]
                            #println(j)
                            mean_val = mean(i[(length(i)-5):length(i)])
                            std_dev_val = std(i[(length(i)-5):length(i)])
                            write(file1,"$(j)\n")
                            write(file1,"$(mean_val)\n")
                            write(file1,"$(std_dev_val)\n")
                            #println(mean_val)
                            #println(std_dev_val)
                        end
    
                        #j = readdir(glob"*.bson", dir)
                        #println(j[length(j)-1])
                        #@load j[length(j)-1] model
                        #model = model 
                        #println(model[2].weight)
                        #model = model |>gpu
                        
                        #y_=[]
                        #y =[]
                        #for (x_tmp,y_tmp) in test_model
                        #    
                        #    result = model(x_tmp)
    
                        #    result = result |> cpu
                        #    println(result)
                        #    CUDA.@allowscalar append!(y_,result)
                        #    CUDA.@allowscalar append!(y,y_tmp)
                        #end
                        #y_=round.(y_)
                        #println(y_)
                        #println(y)
                        #correctnes =Float32.(y_ .== y)
                        #println(Y_actual_type .* correctnes')
    
    
                    elseif length(test) == 5   
                        #println(file)
                        write(file1, "$(file)\n")
                        a,b,c,d,e = test
                        for i in [a,b,c,d,e]
                            mean_val = mean(i[(length(i)-5):length(i)])
                            std_dev_val = std(i[(length(i)-5):length(i)])
                            write(file1,"$(mean_val)\n")
                            write(file1,"$(std_dev_val)\n")
                            #println(mean_val)
                            #println(std_dev_val)
                        end     
                    end
                end

                 
            end

            
            
        end
    end
end