#pragma once

#include "cutil_math.h"

struct PointLight {
    float3 intensity;
    float3 position;    
};

 
struct DiffuseAreaLight {
    int shape_id;
    float3 radiance;
};


enum LightType { POINTLIGHT, DIFFUSEAREALIGHT };

struct Light {
    LightType type;
    union {
        PointLight pointlight;
        DiffuseAreaLight diffusearealight;
    };

    Light(const PointLight& pl) : type(POINTLIGHT), pointlight(pl) {}
    Light(const DiffuseAreaLight& dal) : type(DIFFUSEAREALIGHT), diffusearealight(dal) {}
};