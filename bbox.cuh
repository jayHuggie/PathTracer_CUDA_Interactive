#pragma once
#include "torrey.cuh"
#include "parse_scene.h"
#include "cutil_math.h"
#include "ray.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

struct BBox {
    float3 p_min = float3{infinity<float>(),
                            infinity<float>(),
                            infinity<float>()};
    float3 p_max = float3{-infinity<float>(),
                            -infinity<float>(),
                            -infinity<float>()};
};

struct HitResult {
    bool didHit;
    float tnear;
    float tfar;
};


__host__ __device__ inline HitResult Hit(const BBox &bbox, Ray ray) {
    float3 inv_dir = make_float3(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);
    float3 t0 = (bbox.p_min - ray.org) * inv_dir;
    float3 t1 = (bbox.p_max - ray.org) * inv_dir;

if (inv_dir.x < 0.0f) {
    float temp = t0.x;
    t0.x = t1.x;
    t1.x = temp;
}
if (inv_dir.y < 0.0f) {
    float temp = t0.y;
    t0.y = t1.y;
    t1.y = temp;
}
if (inv_dir.z < 0.0f) {
    float temp = t0.z;
    t0.z = t1.z;
    t1.z = temp;
}

    float tnear = max(max(t0.x, t0.y), t0.z);
    float tfar = min(min(t1.x, t1.y), t1.z);

    return { tfar >= max(0.0f, tnear), tnear, tfar };
}


inline __host__ __device__ bool hit(const BBox &bbox, Ray ray) {
    // https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies/anoptimizedaabbhitmethod
    float3 inv_dir = make_float3(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);
    float3 t0 = (bbox.p_min - ray.org) * inv_dir;
    float3 t1 = (bbox.p_max - ray.org) * inv_dir;

    if (inv_dir.x < 0.0f) {
        float temp = t0.x;
        t0.x = t1.x;
        t1.x = temp;
    }
    if (inv_dir.y < 0.0f) {
        float temp = t0.y;
        t0.y = t1.y;
        t1.y = temp;
    }
    if (inv_dir.z < 0.0f) {
        float temp = t0.z;
        t0.z = t1.z;
        t1.z = temp;
    }
    ray.tnear = max(max(t0.x, t0.y), t0.z);
    ray.tfar = min(min(t1.x, t1.y), t1.z);
    return ray.tfar >= max(0.0f, ray.tnear);
}

__host__ __device__ inline int largest_axis(const BBox &box) {
    float3 extent = box.p_max - box.p_min;
    if (extent.x > extent.y && extent.x > extent.z) {
        return 0;
    } else if (extent.y > extent.x && extent.y > extent.z) {
        return 1;
    } else { // z is the largest
        return 2;
    }
}

__host__ __device__ inline BBox merge(const BBox &box1, const BBox &box2) {
    float3 p_min = float3{
        min(box1.p_min.x, box2.p_min.x),
        min(box1.p_min.y, box2.p_min.y),
        min(box1.p_min.z, box2.p_min.z)};
    float3 p_max = float3{
        max(box1.p_max.x, box2.p_max.x),
        max(box1.p_max.y, box2.p_max.y),
        max(box1.p_max.z, box2.p_max.z)};
    return BBox{p_min, p_max};
}