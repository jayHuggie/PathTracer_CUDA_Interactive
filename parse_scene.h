#pragma once

#include "torrey.cuh"
#include "cutil_math.h"

#include <filesystem>
#include <variant>
#include <vector>

struct ParsedCamera {
    float3 lookfrom;
    float3 lookat;
    float3 up;
    float vfov;
    int width, height;
};

struct ParsedImageTexture {
    fs::path filename;
    float uscale = 1, vscale = 1;
    float uoffset = 0, voffset = 0;
};

using ParsedColor = std::variant<float3 /* RGB */, ParsedImageTexture>;

struct ParsedDiffuse {
    ParsedColor reflectance;
};

struct ParsedMirror {
    ParsedColor reflectance;
};

struct ParsedPlastic {
    float eta; // index of refraction
    ParsedColor reflectance;
};

struct ParsedPhong {
    ParsedColor reflectance; // Ks
    float exponent; // alpha
};

struct ParsedBlinnPhong {
    ParsedColor reflectance; // Ks
    float exponent; // alpha
};

struct ParsedBlinnPhongMicrofacet {
    ParsedColor reflectance; // Ks
    float exponent; // alpha
};

using ParsedMaterial = std::variant<ParsedDiffuse,
                                    ParsedMirror,
                                    ParsedPlastic,
                                    ParsedPhong,
                                    ParsedBlinnPhong,
                                    ParsedBlinnPhongMicrofacet>;

struct ParsedPointLight {
    float3 position;
    float3 intensity;
};

struct ParsedDiffuseAreaLight {
    int shape_id;
    float3 radiance;
};

using ParsedLight = std::variant<ParsedPointLight, ParsedDiffuseAreaLight>;

/// A Shape is a geometric entity that describes a surface. E.g., a sphere, a triangle mesh, a NURBS, etc.
/// For each shape, we also store an integer "material ID" that points to a material, and an integer
/// "area light ID" that points to a light source if the shape is an area light. area_lightID is set to -1
/// if the shape is not an area light.
struct ParsedShapeBase {
    int material_id = -1;
    int area_light_id = -1;
};

struct ParsedSphere : public ParsedShapeBase{
    //int material_id = -1;
    //int area_light_id = -1;
    float3 position;
    float radius;
};

struct ParsedTriangleMesh : public ParsedShapeBase {
    std::vector<float3> positions;
    std::vector<int3> indices;
    std::vector<float3> normals;
    std::vector<float2> uvs;
};

using ParsedShape = std::variant<ParsedSphere, ParsedTriangleMesh>;

inline void set_material_id(ParsedShape &shape, int material_id) {
    std::visit([&](auto &s) { s.material_id = material_id; }, shape);
}
inline void set_area_light_id(ParsedShape &shape, int area_light_id) {
    std::visit([&](auto &s) { s.area_light_id = area_light_id; }, shape);
}
inline int get_material_id(const ParsedShape &shape) {
    return std::visit([&](const auto &s) { return s.material_id; }, shape);
}
inline int get_area_light_id(const ParsedShape &shape) {
    return std::visit([&](const auto &s) { return s.area_light_id; }, shape);
}
inline bool is_light(const ParsedShape &shape) {
    return get_area_light_id(shape) >= 0;
}

struct ParsedScene {
    ParsedCamera camera;
    std::vector<ParsedMaterial> materials;
    std::vector<ParsedLight> lights;
    std::vector<ParsedShape> shapes;
    float3 background_color;
    int samples_per_pixel;
};

ParsedScene parse_scene(const fs::path &filename);
