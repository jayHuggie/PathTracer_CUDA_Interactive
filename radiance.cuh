#pragma once

#include "cutil_math.h"
#include "scene.h"

#include <curand_kernel.h>
#include <cfloat>
#include "ray.h"
#include "torrey.cuh"


#define MAX_DEPTH 50

__device__ float max_elem(const float3& a) {
    return fmaxf(fmaxf(a.x, a.y), a.z);
}


// Russian Roulette Ver.
 
__device__ float3 radiance(const GPUScene &scene, Ray ray, curandState &state) {
    float3 L = make_float3(0, 0, 0);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        OptionalIntersection hit_isect = intersect(scene, ray);
        //OptionalIntersection hit_isect = intersect(scene, ray);

        if (!hit_isect.valid) {
            L += throughput * scene.background_color;
            break;
        }

        float3 p = hit_isect.value.position;
        float3 n = hit_isect.value.shading_normal;

        if (hit_isect.value.area_light_id != -1) {
            const Light &light = scene.lights[hit_isect.value.area_light_id];
            if (light.type == DIFFUSEAREALIGHT) {
                DiffuseAreaLight diff_light = light.diffusearealight;
                if (dot(-ray.dir, n) > 0) {
                    L += throughput * diff_light.radiance;
                }
            }
        }

        if (dot(-ray.dir, n) < 0) {
            n = -n;
        }

        SamplingRecord sampling = sample_brdf(scene, hit_isect.value, n, -ray.dir, state);
        if (sampling.is_pure_specular) {
            if (max_elem(sampling.weight) > 0) {
                throughput *= sampling.weight;
            } else {
                break;
            }
        } else {
            BRDFEvalRecord brdf = eval_brdf(scene, hit_isect.value, n, -ray.dir, sampling.wo);
            if (max_elem(brdf.value) > 0 && brdf.pdf > 0) {
                throughput *= brdf.value / brdf.pdf;
            } else {
                break;
            }
        }

        ray = Ray{p, sampling.wo, 1e-4f, FLT_MAX};

        // Russian Roulette termination
        if (depth > 5) { 
            const float p = max(0.5f, 1 - max_elem(throughput));
            if (curand_uniform(&state) < p) {
                break;
            }
            throughput /= 1 - p;
        }

        
    }
    return L;
}

    // No Russian Roulette
/* 
__device__ float3 radiance(const GPUScene &scene, Ray ray, curandState &state) {
    float3 L = make_float3(0, 0, 0);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        OptionalIntersection hit_isect = intersect(scene, ray);

        if (!hit_isect.valid) {
            L += throughput * scene.background_color; // Add the background color
            break;
        }

        float3 p = hit_isect.value.position;
        float3 n = hit_isect.value.shading_normal;

        if (hit_isect.value.area_light_id != -1) {
            const Light &light = scene.lights[hit_isect.value.area_light_id];
            if (light.type == DIFFUSEAREALIGHT) {
                DiffuseAreaLight diff_light = light.diffusearealight;
                if (dot(-ray.dir, n) > 0) {
                    L += throughput * diff_light.radiance;
                }
            }
        }

        if (dot(-ray.dir, n) < 0) {
            n = -n;
        }

        SamplingRecord sampling = sample_brdf(scene, hit_isect.value, n, -ray.dir, state);
        if (sampling.is_pure_specular) {
            if (max_elem(sampling.weight) > 0) {
                throughput *= sampling.weight;
            } else {
                break;
            }
        } else {
            BRDFEvalRecord brdf = eval_brdf(scene, hit_isect.value, n, -ray.dir, sampling.wo);
            if (max_elem(brdf.value) > 0 && brdf.pdf > 0) {
                throughput *= brdf.value / brdf.pdf;
            } else {
                break;
            }
        }

        ray = Ray{p, sampling.wo, 1e-4f, FLT_MAX};
    }

    return L;
}
*/
