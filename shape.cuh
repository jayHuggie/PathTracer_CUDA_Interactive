#pragma once

#include "cutil_math.h"
#include <cuda_runtime.h> 
#include "intersection.h"
#include <vector>
#include "ray.h"
#include <float.h>

inline void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

struct ShapeBase {
    int material_id = -1;
    int area_light_id = -1;
};

struct Sphere : public ShapeBase {
    float3 center;
    float radius;
};


struct TriangleMesh : public ShapeBase {
    float3* host_positions;
    int3* host_indices;
    float3* host_normals;
    float2* host_uvs;

    float3* device_positions;
    int3* device_indices;
    float3* device_normals;
    float2* device_uvs;

    int positions_size;
    int indices_size;
};

struct Triangle {
    int face_index;
    int mesh_index;  // Index into the TriangleMesh vector
};

inline void copyTriangleMeshToDevice(TriangleMesh &mesh) {
    // Allocate memory on the device and copy data from host to device
    checkCudaErrors(cudaMalloc((void**)&mesh.device_positions, mesh.positions_size * sizeof(float3)));
    checkCudaErrors(cudaMalloc((void**)&mesh.device_indices, mesh.indices_size * sizeof(int3)));
    checkCudaErrors(cudaMalloc((void**)&mesh.device_normals, mesh.positions_size * sizeof(float3)));  // Normals are per vertex
    checkCudaErrors(cudaMalloc((void**)&mesh.device_uvs, mesh.positions_size * sizeof(float2)));  // UVs are per vertex

    checkCudaErrors(cudaMemcpy(mesh.device_positions, mesh.host_positions, mesh.positions_size * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mesh.device_indices, mesh.host_indices, mesh.indices_size * sizeof(int3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mesh.device_normals, mesh.host_normals, mesh.positions_size * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mesh.device_uvs, mesh.host_uvs, mesh.positions_size * sizeof(float2), cudaMemcpyHostToDevice));
}

enum ShapeType {SPHERE, TRIANGLE};

struct Shape {
    ShapeType type;

    union {
        Sphere sphere;
        Triangle triangle;
    };

    __host__ __device__ Shape(const Sphere& s) : type(SPHERE), sphere(s) {}
    __host__ __device__ Shape(const Triangle& t) : type(TRIANGLE), triangle(t) {}


    __host__ __device__ ~Shape() {
        switch(type) {
            case SPHERE:
                sphere.~Sphere();
                break;
            case TRIANGLE:
                triangle.~Triangle();
                break;
        }
    }

    __host__ __device__ Shape(const Shape& other) : type(other.type) {
        switch(type) {
            case SPHERE:
                new (&sphere) Sphere(other.sphere);
                break;
            case TRIANGLE:
                new (&triangle) Triangle(other.triangle);
                break;
        }
    }

    __host__ __device__ Shape& operator=(const Shape& other) {
        if(this != &other) {
            this->~Shape();
            new (this) Shape(other);
        }
        return *this;
    }
};


/// Numerically stable quadratic equation solver at^2 + bt + c = 0
/// See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
/// returns false when it can't find solutions.
inline __device__ bool solve_quadratic(float a, float b, float c, float *t0, float *t1) {
    // Degenerated case
    if (a == 0) {
        if (b == 0) {
            return false;
        }
        *t0 = *t1 = -c / b;
        return true;
    }

    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return false;
    }
    float root_discriminant = sqrtf(discriminant);
    if (b >= 0) {
        *t0 = (- b - root_discriminant) / (2 * a);
        *t1 = 2 * c / (- b - root_discriminant);
    } else {
        *t0 = 2 * c / (- b + root_discriminant);
        *t1 = (- b + root_discriminant) / (2 * a);
    }
    return true;
}

inline __device__ OptionalIntersection find_intersection_with_sphere(const Sphere &sph, const Ray &ray) {

    OptionalIntersection result;

    float3 v = ray.org - sph.center;
    float A = dot(ray.dir, ray.dir);
    float B = 2 * dot(ray.dir, v);
    float C = dot(v, v) - sph.radius * sph.radius;
    float t0, t1;
    if (!solve_quadratic(A, B, C, &t0, &t1)) {
        // No intersection
        result.valid = false;
        return result;
    }
    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    float t = t0;
    if (t0 >= ray.tnear && t0 < ray.tfar) {
        t = t0;
    }
    if (t1 >= ray.tnear && t1 < ray.tfar && t < ray.tnear) {
        t = t1;
    }

    if (t >= ray.tnear && t < ray.tfar) {
        // Record the intersection
        float3 p = ray.org + t * ray.dir;
        float3 n = normalize(p - sph.center);
        float theta = acosf(n.y);
        float phi = atan2f(-n.z, n.x) + c_PI;
        float2 uv{phi / (2 * c_PI), theta / c_PI};
        result.valid = true;
 
        result.value = Intersection{p, // position
                                    n, // geometric normal
                                    n, // shading normal
                                    t, // distance
                                    uv, // UV
                                    sph.material_id,
                                    sph.area_light_id};
    }
    
    else {
        result.valid = false;
    }
    
    return result;
}

inline __device__ OptionalFloat3 intersect_triangle(const Ray &ray,
                       const float3 &p0,
                       const float3 &p1,
                       const float3 &p2) {
    OptionalFloat3 result;
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 s1 = cross(ray.dir, e2);
    float divisor = dot(s1, e1);
    if (divisor == 0) {
        result.valid = false;
        return result;
    }
    float inv_divisor = 1 / divisor;
    float3 s = ray.org - p0;
    float u = dot(s, s1) * inv_divisor;
    float3 s2 = cross(s, e1);
    float v = dot(ray.dir, s2) * inv_divisor;
    float t = dot(e2, s2) * inv_divisor;

    if (t > ray.tnear && t < ray.tfar && u >= 0 && v >= 0 && u + v <= 1) {
        result.valid = true;
        result.value = make_float3(u, v, t);
        return result;
    }
    result.valid = false;
    return result;
}
