#pragma once

#include "parse_scene.h"
#include "cutil_math.h"
#include <cmath>
#include "ray.h"

const float POS_INFINITY = INFINITY;

struct Camera {
    float3 lookfrom;
    float3 lookat;
    float3 up;
    float vfov;
};

struct CameraRayData {
    float3 origin;
    float3 top_left_corner;
    float3 horizontal;
    float3 vertical;
};

inline Camera from_parsed_camera(const ParsedCamera &cam) {
    return Camera{cam.lookfrom, cam.lookat, cam.up, cam.vfov};
}

inline CameraRayData compute_camera_ray_data(const Camera &cam, int width, int height) {
    float aspect_ratio = float(width) / float(height);
    float viewport_height = 2.0 * tan(radians(cam.vfov / 2));
    float viewport_width = aspect_ratio * viewport_height;

    float3 cam_dir = normalize(cam.lookat - cam.lookfrom);
    float3 right = normalize(cross(cam_dir, cam.up));
    float3 new_up = cross(right, cam_dir);

    float3 origin = cam.lookfrom;
    float3 horizontal = viewport_width * right;
    float3 vertical = viewport_height * new_up;
    float3 top_left_corner =
        origin - horizontal / float(2) + vertical / float(2) + cam_dir;
    return CameraRayData{origin, top_left_corner, horizontal, vertical};
}

inline __device__ Ray generate_primary_ray(const CameraRayData &cam_ray_data, float u, float v) {
    float3 dir = normalize(cam_ray_data.top_left_corner +
                            u * cam_ray_data.horizontal -
                            v * cam_ray_data.vertical - cam_ray_data.origin);
    return Ray{cam_ray_data.origin, dir, float(0), POS_INFINITY};
}