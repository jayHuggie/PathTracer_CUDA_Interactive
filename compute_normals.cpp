#include "compute_normals.h"

// Numerical robust computation of angle between unit vectors
inline float unit_angle(const float3 &u, const float3 &v) {
    if (dot(u, v) < 0) {
        return (c_PI - 2) * asin(float(0.5) * length(v + u));
    } else {
        return 2 * asin(float(0.5) * length(v - u));
    }
}

std::vector<float3> compute_normals(const std::vector<float3> &vertices,
                                    const std::vector<int3> &indices) {
    std::vector<float3> normals(vertices.size(), float3{0, 0, 0});

    // Nelson Max, "Computing Vertex Normals from Facet Normals", 1999
    for (auto &index : indices) {
        float3 n = float3{0, 0, 0};
        const float3 &v0 = vertices[index.x];
        const float3 &v1 = vertices[index.y];
        const float3 &v2 = vertices[index.z];
        float3 side1 = v1 - v0, side2 = v2 - v0;
        n = cross(side1, side2);
        float l = length(n);
        if (l != 0) {
            n = n / l;
            float angle0 = unit_angle(normalize(side1), normalize(side2));
            normals[index.x] = normals[index.x] + n * angle0;

            side1 = v2 - v1; side2 = v0 - v1;
            float angle1 = unit_angle(normalize(side1), normalize(side2));
            normals[index.y] = normals[index.y] + n * angle1;

            side1 = v0 - v2; side2 = v1 - v2;
            float angle2 = unit_angle(normalize(side1), normalize(side2));
            normals[index.z] = normals[index.z] + n * angle2;
        }
    }

    for (auto &n : normals) {
        float l = length(n);
        if (l != 0) {
            n = n / l;
        } else {
            // degenerate normals, set it to 0
            n = float3{0, 0, 0};
        }
    }
    return normals;
}

