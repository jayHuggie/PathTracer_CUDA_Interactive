#pragma once

#include "cutil_math.h"

struct Intersection {
    float3 position;
    float3 geometric_normal;
    float3 shading_normal;
    float distance;
    float2 uv;
    int material_id;
    int area_light_id;
};

struct OptionalIntersection {
    bool valid;
    Intersection value;
};

struct OptionalFloat3 {
    bool valid;
    float3 value;
};