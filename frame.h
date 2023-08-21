#pragma once

#include "torrey.cuh"
#include "cutil_math.h"

// Define a new struct to replace std::pair<float3, float3>
struct Float3Pair {
    float3 first;
    float3 second;
};

/// Given a vector n, outputs two vectors such that all three vectors are
/// orthogonal to each other.
/// The approach here is based on Frisvad's paper
/// "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization"
/// https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
inline __device__ Float3Pair coordinate_system(const float3 &n) {
    Float3Pair result;
    if (n.z < float(-1 + 1e-6)) {
        result.first = make_float3(0, -1, 0);
        result.second = make_float3(-1, 0, 0);
    } else {
        float a = 1 / (1 + n.z);
        float b = -n.x * n.y * a;
        result.first = make_float3(1 - n.x * n.x * a, b, -n.x);
        result.second = make_float3(b, 1 - n.y * n.y * a, -n.y);
    }
    return result;
}

/// A "Frame" is a coordinate basis that consists of three orthogonal unit floats.
/// This is useful for sampling points on a hemisphere or defining anisotropic BSDFs.
struct Frame {
    __device__ Frame() {}

    __device__ Frame(const float3 &x, const float3 &y, const float3 &n)
        : x(x), y(y), n(n) {}

    __device__ Frame(const float3 &n) : n(n) {
        Float3Pair pair = coordinate_system(n);
        x = pair.first;
        y = pair.second;
    }

    __device__ float3& operator[](int i) {
        return *(&x + i);
    }
    __device__ const float3& operator[](int i) const {
        return *(&x + i);
    }
    float3 x, y, n;
};

inline __device__ Frame operator-(const Frame &frame) {
    return Frame(-frame.x, -frame.y, -frame.n);
}

inline __device__ float3 to_local(const Frame &frame, const float3 &v) {
    return float3{dot(v, frame.x), dot(v, frame.y), dot(v, frame.n)};
}

inline __device__ float3 to_world(const Frame &frame, const float3 &v) {
    return frame.x * v.x + frame.y * v.y + frame.n * v.z;
}
