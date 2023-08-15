#### Datapipeline ####

using ScancoFiles
using Images
using Flux
using Scratch
using ImageMetadata
using JSON
using ProgressBars
using HDF5
using OpenCV

  ### Trial Arrea ###
#print(pwd())
#cd("$(homedir())/Documents/TUHH/Bachelorarbeit/data")
#print(readdir())
#if !("jsons" in readdir())
#    mkdir("jsons") 
#end
#print("\n\n")


#load a dict from a JSON File
#dict2 = Dict()
#open("jsons\\C0027977_mv_ISQ.json", "r") do f
#    global dict2
#    print("test")
#    dict2=JSON.parse(f)  # parse and transform data
#end
#print(dict2["version"],dict2["physdim"])


### Actual Code ###


##preprocessing ##
if !isdir(joinpath(joinpath(pwd(),"extracted_data"), "h5")) 
    mkpath(joinpath(joinpath(pwd(),"extracted_data"), "h5"))
end
if !isdir(joinpath(joinpath(pwd(),"extracted_data"), "json"))
    mkpath(joinpath(joinpath(pwd(),"extracted_data"), "json"))
end


for i in ProgressBar(readdir())
   #save all .ISQ files in the directory as json and remove the patient name
    print(i,"\n")
    if endswith(i,".ISQ")
        ISQname = i[1:(length(i)-4)]
        img = loadISQ(i)
        img_datatype_array=[0,0,0,0,0]
        if Int(img["dataType"]) <= 5
            img_datatype_array[Int(img["dataType"])] = 1;
        else
            Exception("dataType Exceeds value Range ")
        end
        image_data = ImageMeta(img[:,:,:])
        #print(dataType, index,img["patientName"] )
        Dict_1 = Dict("version" => img["version"],"scannerID" => img["scannerID"],"numberOfProjections" => img["numberOfProjections"],"scannerType" => img["scannerType"],"intensity" => img["intensity"]
        ,"dataRange" => img["dataRange"],"pixdim" => img["pixdim"],"reconstructionAlg" => img["reconstructionAlg"],"sliceThickness" => img["sliceThickness"],"sliceIncrement" => img["sliceIncrement"]
        ,"measurementIndex" => img["measurementIndex"],"referenceLine" => img["referenceLine"],"site" => img["site"],"dataOffset" => img["dataOffset"],"numberOfSamples" => img["numberOfSamples"]
        ,"sampleTime" => img["sampleTime"],"muScaling" => img["muScaling"],"physdim" => img["physdim"],"numBlocks" => img["numBlocks"],"energy" => img["energy"],"dataType" => img["dataType"]
        ,"headerSize" => img["headerSize"],"scanDistance" => img["scanDistance"],"startPosition" => img["startPosition"],"numBytes" => img["numBytes"],"patientIndex" => img["patientIndex"],"img_datatype_array" => img_datatype_array
        )#,"data" => image_data
        # save the image in a json dictionary 
        stringdata = JSON.json(Dict_1)
        open(joinpath(joinpath(joinpath(pwd() , "extracted_data"), "json") , ISQname * "_ISQ" * ".json"), "w") do f
            write(f, stringdata)
        h5open(joinpath(joinpath(joinpath(pwd() , "extracted_data"), "h5") , ISQname * "_ISQ" * ".h5"),"w") do h5
            for i in 1:img["pixdim"][3]
                image =ImageAxes.data(img[:,:,i])
                h5[string(i)]=image
            end
            #h5open(pwd() * "\\extracted_data"* "\\h5\\" * ISQname *"_snipped"*"_ISQ" * ".h5","w") do h5
            #    for i in 1:110
            #        image =Float32.(ImageAxes.data(img[257:1280,257:1280,i]))/(img["dataRange"][2])
            #        h5[string(i)]=image
            #    end
            #end
        end

    end

    #save all .RSQ files in the directory as json and remove the patient name
    elseif endswith(i,".RSQ")
        RSQname = i[1:(length(i)-4)]
        img = loadRSQ(i)
        img_datatype_array=[0,0,0,0,0]
        if Int(img["dataType"]) <= 5
            img_datatype_array[Int(img["dataType"])] = 1;
        else
            Exception("dataType Exceeds value Range ")
        end
    
        image_data = ImageMeta(img[:,:,:])
        Dict_2 = Dict("patientIndex" => img["patientIndex"],"detectorOffset" => img["detectorOffset"],"detectorSourceDist" => img["detectorSourceDist"],"detectorCenterDist" => img["detectorCenterDist"]
        ,"scannerID" => img["scannerID"],"detectorHeight" => img["detectorHeight"],"version" => img["version"],"numBytes" => img["numBytes"],"dataPixelR" => img["dataPixelR"]
        ,"detectorWidth" => img["detectorWidth"],"intensity" => img["intensity"],"dataRange" => img["dataRange"],"dataRecord" => img["dataRecord"],"integrationTime" => img["integrationTime"]
        ,"sliceThickness" => img["sliceThickness"],"ioRecord" => img["ioRecord"],"pixdim" => img["pixdim"],"numberOfProjections" => img["numberOfProjections"],"sliceIncrement" => img["sliceIncrement"]
        ,"fanAngle" => img["fanAngle"],"dataOffset" => img["dataOffset"],"numberOfSamples" => img["numberOfSamples"],"muScaling" => img["muScaling"],"angleRange" => img["angleRange"]
        ,"physdim" => img["physdim"],"numBlocks" => img["numBlocks"],"energy" => img["energy"],"centerPixel" => img["centerPixel"],"dataType" => img["dataType"],"headerSize" => img["headerSize"]
        ,"orbitStartOffset" => img["orbitStartOffset"],"startPosition" => img["startPosition"],"calibrationScans" => img["calibrationScans"],"darkRecord" => img["darkRecord"],"img_datatype_array" => img_datatype_array)
        # save the image in a json dictionary
        stringdata = JSON.json(Dict_2)
        open(joinpath(joinpath(joinpath(pwd() , "extracted_data"), "json") , RSQname * "_RSQ" * ".json"), "w") do f
            write(f, stringdata)
        end
        h5open(joinpath(joinpath(joinpath(pwd() , "extracted_data"), "h5") , RSQname * "_RSQ" * ".h5"),"w") do h5
            for i in 1:img["pixdim"][3]
                image = ImageAxes.data(img[:,:,i])
                h5[string(i)]=image
            end
        end
#        h5open(pwd() * "\\h5\\" * RSQname *"_snipped" * "_RSQ" * ".h5","w") do h5
#            for i in 1:124
#                image = Gray.(Float32.(ImageAxes.data(img[:,:,i]))//(img["dataRange"][2]))
#                h5[string(i)]=image
#            end
#        end
    end

end