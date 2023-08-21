#pragma once

#include "cutil_math.h"
#include "texture.h"
#include "torrey.cuh"
#include "cutil_math.h"

struct Diffuse {
    float3 reflectance;
};

struct Mirror {
    float3 reflectance;
};


struct Plastic {
    float eta; // IOR
    Texture reflectance;
};

struct Phong {
    Texture reflectance; // Ks
    float exponent; // alpha
};

enum MaterialType { DIFFUSE, MIRROR, PLASTIC, PHONG };

struct Material {
    MaterialType type;
    union {
        Diffuse diffuse;
        Mirror mirror;
        Plastic plastic;
        Phong phong;
    };

    __host__ __device__ Material(const Diffuse& d) : type(DIFFUSE), diffuse(d) {}
    __host__ __device__ Material(const Mirror& m) : type(MIRROR), mirror(m) {}
    __host__ __device__ Material(const Plastic& pl) : type(PLASTIC), plastic(pl) {}
    __host__ __device__ Material(const Phong& ph) : type(PHONG), phong(ph) {}

    __host__ __device__ ~Material() {
        switch(type) {
            case DIFFUSE:
                diffuse.~Diffuse();
                break;
            case MIRROR:
                mirror.~Mirror();
                break;
            case PLASTIC:
                plastic.~Plastic();
                break;
            case PHONG:
                phong.~Phong();
                break;
            // Handle other cases as I add more material types
        }
    }

    __host__ __device__ Material(const Material& other) : type(other.type) {
        switch(type) {
            case DIFFUSE:
                new (&diffuse) Diffuse(other.diffuse);
                break;
            case MIRROR:
                new (&mirror) Mirror(other.mirror);
                break;
            case PLASTIC:
                new (&plastic) Plastic(other.plastic);
                break;
            case PHONG:
                new (&phong) Phong(other.phong);
                break;
            // Handle other cases as I add more material types
        }
    }

    __host__ __device__ Material& operator=(const Material& other) {
        if(this != &other) {
            this->~Material();
            new (this) Material(other);
        }
        return *this;
    }
};

