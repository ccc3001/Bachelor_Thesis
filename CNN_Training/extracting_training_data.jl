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
test_x =[]
test_y =[]


for i in readdir("h5")
    j = i[1:(length(a)-3)]
    open(pwd() * "\\json\\" * j * "json", "r") do f
        global dict
        dicttxt = readall(f)  # file information to string
        dict=JSON.parse(dicttxt)  # parse and transform data  
    end

    push!(test_y , dict["img_datatype_array"])

    c = h5open(pwd() * "\\h5\\" * i, "r") do file
        read(file, "A")    
    end

    push!(test_x, c)

end