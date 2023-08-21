#pragma once

// CMake insert NDEBUG when building with RelWithDebInfo
// This is an ugly hack to undo that...
#undef NDEBUG

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <algorithm>

#include <cuda_runtime.h>


// for suppressing unused warnings
#define UNUSED(x) (void)(x)

// We use double for most of our computation.
// Rendering is usually done in single precision Reals.
// However, torrey is an educational renderer with does not
// put emphasis on the absolute performance. 
// We choose double so that we do not need to worry about
// numerical accuracy as much when we render.
// Switching to floating point computation is easy --
// just set Real = float.

//using float = double;

// Lots of PIs!
const float c_PI = float(3.14159265358979323846);
const float c_INVPI = float(1.0) / c_PI;
const float c_TWOPI = float(2.0) * c_PI;
const float c_INVTWOPI = float(1.0) / c_TWOPI;
const float c_FOURPI = float(4.0) * c_PI;
const float c_INVFOURPI = float(1.0) / c_FOURPI;
const float c_PIOVERTWO = float(0.5) * c_PI;
const float c_PIOVERFOUR = float(0.25) * c_PI;

template <typename T>
inline T infinity() {
    return std::numeric_limits<T>::infinity();
}

namespace fs = std::filesystem;

inline std::string to_lowercase(const std::string &s) {
    std::string out = s;
    std::transform(s.begin(), s.end(), out.begin(), ::tolower);
    return out;
}

inline int modulo(int a, int b) {
    auto r = a % b;
    return (r < 0) ? r+b : r;
}

inline float modulo(float a, float b) {
    float r = ::fmodf(a, b);
    return (r < 0.0f) ? r+b : r;
}

inline double modulo(double a, double b) {
    double r = ::fmod(a, b);
    return (r < 0.0) ? r+b : r;
}

template <typename T>
inline T max(const T &a, const T &b) {
    return a > b ? a : b;
}

template <typename T>
inline T min(const T &a, const T &b) {
    return a < b ? a : b;
}

inline __device__ __host__ float radians(float deg) {
    return (c_PI / float(180)) * deg;
}

inline __device__ __host__ float degrees(float rad) {
    return (float(180) / c_PI) * rad;
}

inline __device__ __host__ float square(float s) {
    return s * s;
}

inline __host__ __device__ bool operator<(const float3 &a, const float3 &b) {
    if (a.x < b.x) return true;
    if (a.x > b.x) return false;
    if (a.y < b.y) return true;
    if (a.y > b.y) return false;
    return a.z < b.z;
}

inline __host__ __device__ bool operator>(const float3 &a, const float3 &b) {
    if (a.x > b.x) return true;
    if (a.x < b.x) return false;
    if (a.y > b.y) return true;
    if (a.y < b.y) return false;
    return a.z > b.z;
}
