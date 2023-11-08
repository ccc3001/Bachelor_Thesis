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




struct CBAM{T}
    input_size::T
end

function (m::CBAM)(x)
      
    Channel_Attention_Block=Chain(
        Parallel((A,B)->A.+B,
        Chain(Flux.GlobalMaxPool(),x->Flux.flatten(x)),
        Chain(Flux.GlobalMeanPool(),x->Flux.flatten(x)))
        ,x->sigmoid_fast(x))    
    Channel_Attention=Chain(SkipConnection(
        Channel_Attention_Block,
        (A,B)->reshape(A,(1,1,m.input_size,1)).*B)
        ,BatchNorm(m.input_size))
    

    Spatial_Attention_Block = Chain(
        Parallel((A,B) -> cat(A,B;dims=3),
        x->meanpool(x,(7,7),pad = 3,stride=1),
        x->maxpool(x,(7,7),pad = 3, stride =1)),
        Conv((7,7), m.input_size*2=>m.input_size, sigmoid_fast;pad = 3,stride=1),
        x->sigmoid_fast(x),
        BatchNorm(m.input_size))#Not sure if the batch normalization part makes sence 
    Spatial_Attention = Chain(SkipConnection(Spatial_Attention_Block,(A,B)->A.*B))

    model = Chain(SkipConnection(Chain(Channel_Attention,Spatial_Attention),(A,B)->A.+B))

    return model(x)
end


struct Inception_module{T}
    input_size::T
end
function (m::Inception_module)(x)
    model= Chain(
        x->maxpool(x,(3,3),stride=2),
        
        Parallel((A,B,C,D) -> cat(cat(A,B;dims=3),cat(C,D;dims=3);dims=3),
            Conv((1,1), m.input_size=>m.input_size,elu),
            Chain(
                Conv((1,1), m.input_size=>m.input_size,elu;pad = 0),
                Conv((3,3), m.input_size=>m.input_size,elu; pad= 1)),
            Chain(
                Conv((1,1), m.input_size=>m.input_size,elu ; pad = 0),
                Conv((5,5) ,m.input_size=>m.input_size,elu, pad = 2)),
            Chain(
                x->maxpool(x,(3,3),stride=1,pad=1),
                Conv((1,1),m.input_size=>m.input_size,elu))

        ))
    return model(x)
end


struct Network_in_Network{}

end

function (m::Network_in_Network)(x)
    
    return
end

### my model

model = Chain(
    Conv((5,5),1 => 64, elu ),
    x->maxpool(x,(3,3);stride=1),
    BatchNorm(64),
    CBAM(64),
    x->maxpool(x,(3,3);stride=1),
    BatchNorm(64),

    )