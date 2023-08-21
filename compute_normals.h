#pragma once

#include <vector>
#include "cutil_math.h"
#include "torrey.cuh"

/// Given a pool of vertices and a triangle mesh defined on it,
/// compute a set of normals per vertex using weighted average of
/// face normals.
/// In particular, we implemented Nelson Max's method, c.f.,
/// "Computing Vertex Normals from Facet Normals", 1999
std::vector<float3> compute_normals(const std::vector<float3> &vertices,
                                     const std::vector<int3> &indices);
