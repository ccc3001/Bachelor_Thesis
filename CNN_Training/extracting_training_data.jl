using HDF5
using ImageMetadata
using JSON


#reading json dict 
#dict2 = Dict()
#open("write_read.json", "r") do f
#    global dict2
#    dicttxt = readall(f)  # file information to string
#    dict2=JSON.parse(dicttxt)  # parse and transform data
#end

#reading h5 file 
#c = h5open("mydata.h5", "r") do file
#    read(file, "A")
#end

#defining variables 
#test_x =[]
#test_y =[]


#for i in readdir("h5")
#    j = i[1:(length(i)-3)]
#    open(pwd() * "\\json\\" * j * ".json", "r") do f
#        global dict
#        dicttxt = readline(f)  # file information to string
#        dict= JSON.parse(dicttxt)  # parse and transform data  
#    end
#
#    push!(test_y , dict["img_datatype_array"])
#    push!(test_y , dict["img_datatype_array"])
#    push!(test_y , dict["img_datatype_array"])
#
#
#
#   c = h5open(pwd() * "\\h5\\" * i, "r") do file
#        read(file)    
#    end
#    push!(test_x, c["1"])
#    push!(test_x, c["50"])
#    push!(test_x, c["100"])
#
#end

### new more efficient methode 
#create a dictionary with all the isq files and the exact slices i want to read from them 


ISQ_Dict = Dict{Int,String}
n = 1
for (root, dirs, files) in walkdir(pwd())
    for file in files
        ## checks for .ISQ files and creates .h5 files to get the filesize down 
        if endswith(file , ".ISQ")
            img = loadISQ(file)
            h5open(pwd() * file[1:(length(file)-4)] * ".h5" ,"w") do h5
                for i in [1,15,29,43,57,70,83,97,110]
                    image =ImageAxes.data(img[:,:,i])
                    h5[string(i)]=image
                end
            img_datatype_array=[0,0,0,0,0]
            if Int(img["dataType"]) <= 5
                img_datatype_array[Int(img["dataType"])] = 1;
            end
                h5[string(0)=img_datatype_array]
            end
            rm(pwd()*file)
        end
    
end


##goes through the foldern structure and creates a dictionary with all the location of  h5 files
for (root, dirs, files) in walkdir(pwd())
        if endswith(file, ".h5")
            a = Dict(n => pwd() * file)
            n = n + 1
            ISQ_Dict = merge!(ISQ_Dict,a)
        end
    end
    
end

print(ISQ_Dict)

### function to load the 
# i should do preprocessing first