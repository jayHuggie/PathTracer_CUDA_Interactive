#pragma once

#include "torrey.cuh"
#include "matrix.h"
#include "cutil_math.h"

/// A collection of 3D transformations.
Matrix4x4 translate(const float3 &delta);
Matrix4x4 scale(const float3 &scale);
Matrix4x4 rotate(float angle, const float3 &axis);
Matrix4x4 look_at(const float3 &pos, const float3 &look, const float3 &up);
Matrix4x4 perspective(float fov);
/// Actually transform the vectors given a transformation.
float3 xform_point(const Matrix4x4 &xform, const float3 &pt);
float3 xform_vector(const Matrix4x4 &xform, const float3 &vec);
float3 xform_normal(const Matrix4x4 &inv_xform, const float3 &n);
