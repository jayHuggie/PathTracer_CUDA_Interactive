#include "transform.h"


/// Much of the code is taken from pbrt https://github.com/mmp/pbrt-v3/tree/master/src

Matrix4x4 translate(const float3 &delta) {
    return Matrix4x4(float(1), float(0), float(0), delta.x,
                     float(0), float(1), float(0), delta.y,
                     float(0), float(0), float(1), delta.z,
                     float(0), float(0), float(0), float(1));
}

Matrix4x4 scale(const float3 &s) {
    return Matrix4x4(   s.x, float(0), float(0), float(0),
                     float(0),    s.y, float(0), float(0),
                     float(0), float(0),    s.z, float(0),
                     float(0), float(0), float(0), float(1));
}

Matrix4x4 rotate(float angle, const float3 &axis) {
    float3 a = normalize(axis);
    float s = sin(radians(angle));
    float c = cos(radians(angle));
    Matrix4x4 m;
    m(0, 0) = a.x * a.x + (1 - a.x * a.x) * c;
    m(0, 1) = a.x * a.y * (1 - c) - a.z * s;
    m(0, 2) = a.x * a.z * (1 - c) + a.y * s;
    m(0, 3) = 0;

    m(1, 0) = a.x * a.y * (1 - c) + a.z * s;
    m(1, 1) = a.y * a.y + (1 - a.y * a.y) * c;
    m(1, 2) = a.y * a.z * (1 - c) - a.x * s;
    m(1, 3) = 0;

    m(2, 0) = a.x * a.z * (1 - c) - a.y * s;
    m(2, 1) = a.y * a.z * (1 - c) + a.x * s;
    m(2, 2) = a.z * a.z + (1 - a.z * a.z) * c;
    m(2, 3) = 0;

    m(3, 0) = 0;
    m(3, 1) = 0;
    m(3, 2) = 0;
    m(3, 3) = 1;
    return m;
}

Matrix4x4 look_at(const float3 &pos, const float3 &look, const float3 &up) {
    Matrix4x4 m;
    float3 dir = normalize(look - pos);
    assert(length(cross(normalize(up), dir)) != 0);
    float3 left = normalize(cross(normalize(up), dir));
    float3 new_up = cross(dir, left);
    m(0, 0) = left.x;
    m(1, 0) = left.y;
    m(2, 0) = left.z;
    m(3, 0) = 0;
    m(0, 1) = new_up.x;
    m(1, 1) = new_up.y;
    m(2, 1) = new_up.z;
    m(3, 1) = 0;
    m(0, 2) = dir.x;
    m(1, 2) = dir.y;
    m(2, 2) = dir.z;
    m(3, 2) = 0;
    m(0, 3) = pos.x;
    m(1, 3) = pos.y;
    m(2, 3) = pos.z;
    m(3, 3) = 1;
    return m;
}

Matrix4x4 perspective(float fov) {
    float cot = float(1.0) / tan(radians(fov / 2.0));
    return Matrix4x4(    cot, float(0), float(0),  float(0),
                     float(0),     cot, float(0),  float(0),
                     float(0), float(0), float(1), float(-1),
                     float(0), float(0), float(1),  float(0));
}

float3 xform_point(const Matrix4x4 &xform, const float3 &pt) {
    float4 tpt = make_float4(
        xform(0, 0) * pt.x + xform(0, 1) * pt.y + xform(0, 2) * pt.z + xform(0, 3),
        xform(1, 0) * pt.x + xform(1, 1) * pt.y + xform(1, 2) * pt.z + xform(1, 3),
        xform(2, 0) * pt.x + xform(2, 1) * pt.y + xform(2, 2) * pt.z + xform(2, 3),
        xform(3, 0) * pt.x + xform(3, 1) * pt.y + xform(3, 2) * pt.z + xform(3, 3));
    float inv_w = float(1) / tpt.w;
    return make_float3(tpt.x * inv_w, tpt.y * inv_w, tpt.z * inv_w);
}

float3 xform_vector(const Matrix4x4 &xform, const float3 &vec) {
    return make_float3(xform(0, 0) * vec.x + xform(0, 1) * vec.y + xform(0, 2) * vec.z,
                   xform(1, 0) * vec.x + xform(1, 1) * vec.y + xform(1, 2) * vec.z,
                   xform(2, 0) * vec.x + xform(2, 1) * vec.y + xform(2, 2) * vec.z);
}

float3 xform_normal(const Matrix4x4 &inv_xform, const float3 &n) {
    return normalize(float3{
        inv_xform(0, 0) * n.x + inv_xform(1, 0) * n.y + inv_xform(2, 0) * n.z,
        inv_xform(0, 1) * n.x + inv_xform(1, 1) * n.y + inv_xform(2, 1) * n.z,
        inv_xform(0, 2) * n.x + inv_xform(1, 2) * n.y + inv_xform(2, 2) * n.z});
}
