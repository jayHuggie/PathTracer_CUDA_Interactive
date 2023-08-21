#ifndef RAYH
#define RAYH
#include "cutil_math.h"

struct Ray
{
public:
    __device__ Ray() {}
    __device__ Ray(const float3& origin, const float3& direction)
            : org(origin), dir(direction)
    {}
    __device__ Ray(const float3& origin, const float3& direction, float tnear, float tfar)
            : org(origin), dir(direction), tnear(tnear), tfar(tfar)
        {}

    float3 org, dir;
    float tnear, tfar;
};

#endif