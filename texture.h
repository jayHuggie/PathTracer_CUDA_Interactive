#pragma once

#include "torrey.cuh"
#include "cutil_math.h"

struct ConstantTexture {
    float3 color;
};

// enum TextureType {CONSTANT/*, IMAGE*/}; ---> TODO: add texture later

struct Texture {
    
    ConstantTexture constant;

    
    
    /* TODO: add texture later
    TextureType type;
    union {
        ConstantTexture constant;
        // ImageTexture
    };

    __host__ __device__ Texture(const ConstantTexture& ct) : type(CONSTANT), constant(ct) {}

    __host__ __device__ ~Texture() {
        switch(type) {
            case CONSTANT:
                constant.~ConstantTexture();
                break;
            // case IMAGE:
                // image.~ImageTexture();
                // break;
        }
    }

    __host__ __device__ Texture(const Texture& other) : type(other.type) {
        switch(type) {
            case CONSTANT:
                new (&constant) ConstantTexture(other.constant);
                break;
            // case IMAGE:
                // new (&image) ImageTexture(other.image);
                // break;
        }
    }

    __host__ __device__ Texture& operator=(const Texture& other) {
        if(this != &other) {
            this->~Texture();
            new (this) Texture(other);
        }
        return *this;
    }
    */

};

/*
struct ImageTexture {
    Image3 image;
    Real uscale = 1, vscale = 1;
    Real uoffset = 0, voffset = 0;
};*/

//using Texture = std::variant<ConstantTexture, ImageTexture>;

inline __device__ float3 eval(const Texture &texture, const float2 &uv) {
    return texture.constant.color;
}