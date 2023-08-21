#pragma once

#include "cutil_math.h"
#include "camera.cuh"
#include "light.h"
#include "intersection.h"
#include "material.h"
#include "parse_scene.h"
#include "shape.cuh"
#include "ray.h"
#include "frame.h"
#include "bvh.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

struct Scene {
    Scene(const ParsedScene &scene);

    Camera camera;
    int width, height;
    std::vector<Shape> shapes;
    std::vector<Material> materials;
    std::vector<Light> lights;
    float3 background_color;

    int samples_per_pixel;

    
    // For the Triangle in the shapes to reference to.
    std::vector<TriangleMesh> meshes;

    std::vector<BVHNode> CPU_bvh_nodes;
    int bvh_root_id;

        void cleanupHostMemory() {
        for (auto& mesh : meshes) {
            delete[] mesh.host_positions;
            delete[] mesh.host_indices;
            delete[] mesh.host_normals;
            delete[] mesh.host_uvs;

            // Reset the pointers to ensure they aren't accessed after being deleted
            mesh.host_positions = nullptr;
            mesh.host_indices = nullptr;
            mesh.host_normals = nullptr;
            mesh.host_uvs = nullptr;
        }
    }
};


struct GPUScene {
    Camera camera;
    int width, height;
    int num_shapes;
    Shape* shapes;
    int num_materials;
    Material* materials;
    int num_lights;
    Light* lights;
    float3 background_color;
    int samples_per_pixel;


    int num_meshes;
    TriangleMesh* meshes;

    int num_bvh_nodes;
    BVHNode* GPU_bvh_nodes; 
    int bvh_root_index;

    void copyFrom(const Scene &scene) {
        camera = scene.camera;
        width = scene.width;
        height = scene.height;
        num_shapes = scene.shapes.size();
        num_materials = scene.materials.size();
        num_lights = scene.lights.size();
        background_color = scene.background_color;
        samples_per_pixel = scene.samples_per_pixel;


        num_meshes = scene.meshes.size();
        num_bvh_nodes = scene.bvh_root_id + 1;
        //printf("scene.bvh_root_id is %d\n",scene.bvh_root_id);
        //printf("num_bvh_nodes is %d\n", num_bvh_nodes);


        // Allocate GPU memory
        checkCudaErrors(cudaMalloc(&shapes, num_shapes * sizeof(Shape)));
        checkCudaErrors(cudaMalloc(&materials, num_materials * sizeof(Material)));
        checkCudaErrors(cudaMalloc(&lights, num_lights * sizeof(Light)));
        checkCudaErrors(cudaMalloc(&meshes, num_meshes * sizeof(TriangleMesh)));


        // Copy data to GPU
        checkCudaErrors(cudaMemcpy(materials, scene.materials.data(), num_materials * sizeof(Material), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(lights, scene.lights.data(), num_lights * sizeof(Light), cudaMemcpyHostToDevice));
        
        // Copy each mesh to device
        for (int i = 0; i < num_meshes; i++) {
            checkCudaErrors(cudaMemcpy(&meshes[i], &scene.meshes[i], sizeof(TriangleMesh), cudaMemcpyHostToDevice));
        }

        // Copy each shape to device
        for (int i = 0; i < num_shapes; i++) {
            checkCudaErrors(cudaMemcpy(&shapes[i], &scene.shapes[i], sizeof(Shape), cudaMemcpyHostToDevice));
        }

        // Allocate BVH nodes memory
        checkCudaErrors(cudaMalloc(&GPU_bvh_nodes, num_bvh_nodes * sizeof(BVHNode)));

        // Copy BVH nodes to device
        copyBVHNodesToDevice2(scene.CPU_bvh_nodes, GPU_bvh_nodes);

        // Set root node
        bvh_root_index = scene.bvh_root_id;
         
        printf("index is %d\n", bvh_root_index);
    
/*
    // CHECK!
    // Allocate Host Memory to print BVHNode info
    BVHNode* host_bvh_nodes = new BVHNode[num_bvh_nodes];
    
    // Copy BVHNode data from Device to Host
    checkCudaErrors(cudaMemcpy(host_bvh_nodes, GPU_bvh_nodes, num_bvh_nodes * sizeof(BVHNode), cudaMemcpyDeviceToHost));

    // Print BVHNode info on the Host
    for(int i = 0; i < num_bvh_nodes; i++) {
        printf("GPU Info!!\n");
        printf("Node %d:\n", i);
        printf("  left_node_id: %d\n", host_bvh_nodes[i].left_node_id);
        printf("  right_node_id: %d\n", host_bvh_nodes[i].right_node_id);
        printf("  primitive_id: %d\n", host_bvh_nodes[i].primitive_id);
    }

    // Clean up the host BVHNode memory
    delete[] host_bvh_nodes;
   */
    }

    void free() {
        // Free memory for shapes
        for (int i = 0; i < num_shapes; i++) {
            shapes[i].~Shape();
        }
        checkCudaErrors(cudaFree(shapes));

        // Free memory for materials
        checkCudaErrors(cudaFree(materials));

        // Free memory for lights
        checkCudaErrors(cudaFree(lights));

        // Free memory for each mesh
        for (int i = 0; i < num_meshes; i++) {
            checkCudaErrors(cudaFree(meshes[i].device_positions));
            checkCudaErrors(cudaFree(meshes[i].device_indices));
            checkCudaErrors(cudaFree(meshes[i].device_normals));
            checkCudaErrors(cudaFree(meshes[i].device_uvs));
        }
        
        // Free meshes memory
        checkCudaErrors(cudaFree(meshes));

        // Free BVH nodes memory
        checkCudaErrors(cudaFree(GPU_bvh_nodes));
        
    }

};


inline __device__ OptionalIntersection find_intersection_with_triangle(const Triangle &tri, const GPUScene &scene, const Ray &ray) {
    OptionalIntersection result;

    const TriangleMesh &mesh = scene.meshes[tri.mesh_index];
    int3 index = mesh.device_indices[tri.face_index];
    
    float3 p0 = mesh.device_positions[index.x];
    float3 p1 = mesh.device_positions[index.y];
    float3 p2 = mesh.device_positions[index.z];
    
    OptionalFloat3 uvt_ = intersect_triangle(ray, p0, p1, p2);
    
    if (uvt_.valid) {
        float2 b = make_float2(uvt_.value.x, uvt_.value.y);
        float t = uvt_.value.z;
        float3 p = (1 - b.x - b.y) * p0 + b.x * p1 + b.y * p2;
        float3 geometric_normal = normalize(cross(p1 - p0, p2 - p0));
        float2 uv = b;

        if (mesh.device_uvs != nullptr) {
            float2 uv0 = mesh.device_uvs[index.x];
            float2 uv1 = mesh.device_uvs[index.y];
            float2 uv2 = mesh.device_uvs[index.z];
            uv = (1 - b.x - b.y) * uv0 + b.x * uv1 + b.y * uv2;
        }
        
        float3 shading_normal = geometric_normal;
        if (mesh.device_normals != nullptr) {
            float3 n0 = mesh.device_normals[index.x];
            float3 n1 = mesh.device_normals[index.y];
            float3 n2 = mesh.device_normals[index.z];
            shading_normal = normalize((1 - b.x - b.y) * n0 + b.x * n1 + b.y * n2);
        }

        result.valid = true;
        result.value = Intersection{p, // position
                                    geometric_normal,
                                    shading_normal,
                                    t, // distance
                                    uv,
                                    mesh.material_id,
                                    mesh.area_light_id};
    }
    else {
        result.valid = false;
    }
    
    return result;
}


inline __device__ OptionalIntersection find_intersection_with_shape(const Shape &shape, const GPUScene &scene, const Ray &ray) {
    // Depending on the type of the shape, perform the appropriate intersection.
    switch (shape.type) {
        case SPHERE:
            return find_intersection_with_sphere(shape.sphere, ray);
        case TRIANGLE:
            return find_intersection_with_triangle(shape.triangle, scene, ray);
        default:
            // Return an invalid intersection
            return {false, Intersection{}};
    }
}

inline __device__ bool is_ray_occluded_by_shape(const Shape &shape, const GPUScene &scene, const Ray &ray) {
    OptionalIntersection intersection = find_intersection_with_shape(shape, scene, ray);
    return intersection.valid;
}


inline __device__ OptionalIntersection intersect(const GPUScene &scene, Ray ray) {
    OptionalIntersection closestIntersection;
    closestIntersection.valid = false;
    float closestT = FLT_MAX;

    const int MAX_DEPTH = 64;

    int stack[MAX_DEPTH];  // stack of node indices
    int stackIdx = 0; // top of the stack

    stack[stackIdx++] = scene.bvh_root_index; // start with the root node (assuming its index is 0)

    while (stackIdx > 0) {
        //bvhDepth++;
        if (stackIdx >= MAX_DEPTH) {
            printf("BVH traversal terminated due to stack overflow.\n");
            break;
        }

        int currentNodeIdx = stack[--stackIdx]; // pop node index from the stack
        const BVHNode &currentNode = scene.GPU_bvh_nodes[currentNodeIdx];

        if (currentNode.primitive_id != -1) {
            OptionalIntersection result = find_intersection_with_shape(scene.shapes[currentNode.primitive_id], scene, ray);
            if (result.valid && result.value.distance < closestT) {
                closestT = result.value.distance;
                closestIntersection = result;
            }
            continue;
        }

       
        HitResult leftHit = Hit(scene.GPU_bvh_nodes[currentNode.left_node_id].box, ray);
        HitResult rightHit = Hit(scene.GPU_bvh_nodes[currentNode.right_node_id].box, ray);

        if (leftHit.didHit && rightHit.didHit) {
            // Determine the order of traversal based on which bounding box is hit first
            if (leftHit.tnear < rightHit.tnear) {
                stack[stackIdx++] = currentNode.right_node_id;
                stack[stackIdx++] = currentNode.left_node_id;
            } else {
                stack[stackIdx++] = currentNode.left_node_id;
                stack[stackIdx++] = currentNode.right_node_id;
            }
        } else {
            if (leftHit.didHit) {
                stack[stackIdx++] = currentNode.left_node_id;
            }
            if (rightHit.didHit) {
                stack[stackIdx++] = currentNode.right_node_id;
            }
        }
    }

    return closestIntersection;
}




inline __device__ bool is_ray_occluded_by_GPUScene_bvh(const GPUScene &scene, const BVHNode &node, Ray ray) {
    if (node.primitive_id != -1) {
        // Assuming occluded for primitives (like spheres) returns a bool
        return is_ray_occluded_by_shape(scene.shapes[node.primitive_id], scene, ray);
    }
    
    if (hit(node.box, ray)) {
        const BVHNode &left = scene.GPU_bvh_nodes[node.left_node_id];
        const BVHNode &right = scene.GPU_bvh_nodes[node.right_node_id];

        if (is_ray_occluded_by_GPUScene_bvh(scene, left, ray)) {
            return true;
        }

        if (is_ray_occluded_by_GPUScene_bvh(scene, right, ray)) {
            return true;
        }
    }
    
    return false;
}

inline __device__ bool is_ray_occluded_by_GPUScene(const GPUScene &scene, const Ray &ray) {
    return is_ray_occluded_by_GPUScene_bvh(scene, scene.GPU_bvh_nodes[scene.bvh_root_index], ray);
}


inline __device__ float3 schlick_fresnel(const float3 &F0, float cos_theta) {
    float3 oneVec = make_float3(1.0f, 1.0f, 1.0f);
    return F0 + (oneVec - F0) * pow(1 - cos_theta, 5);
}

inline __device__ float3 sample_cos_hemisphere(const float2 &u) {
    float phi = c_TWOPI * u.x;
    float tmp = sqrt(clamp(1 - u.y, float(0), float(1)));
    return float3{
        cos(phi) * tmp, sin(phi) * tmp,
        sqrt(clamp(u.y, float(0), float(1)))
    };
}

// For Phong/Blinn
inline __device__ float3 sample_cos_n_hemisphere(const float2 &u, float exponent) {
    float phi = c_TWOPI * u.x;
    float cos_theta = pow(u.y, float(1)/(exponent + 1));
    float sin_theta = sqrt(clamp(1 - cos_theta * cos_theta, float(0), float(1)));
    return float3{
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    };
}

struct BRDFEvalRecord {
    float3 value;
    float pdf;
};

inline __device__ BRDFEvalRecord eval_brdf(const GPUScene &scene,
                                const Intersection &isect,
                                const float3 &n,
                                const float3 &wi,
                                const float3 &wo,
                                bool debug = false) {
    const Material &mat = scene.materials[isect.material_id];

    if (mat.type == DIFFUSE) {
        float3 value = mat.diffuse.reflectance *
                        (max(dot(wo, n), 0.0f) / c_PI);
        float pdf = (max(dot(wo, n), 0.0f) / c_PI);
        return {value, pdf};
    } else if (mat.type == MIRROR) {
        // pure specular reflection is handled somewhere else
        return {make_float3(0, 0, 0), 0};
    } else if (mat.type == PLASTIC) {
        float F0s = square((mat.plastic.eta - 1) / (mat.plastic.eta + 1));
        float3 F0 = make_float3(F0s, F0s, F0s);
        float3 F = schlick_fresnel(F0, dot(n, wi));
        // pure specular reflection is handled somewhere else
        float3 value = 
            (make_float3(1.0f,1.0f,1.0f) - F) * eval(mat.plastic.reflectance, isect.uv) *
            (max(dot(wo, n), 0.0f) / c_PI);
        float pdf = (1 - F.x) * (max(dot(wo, n), 0.0f) / c_PI);
        return {value, pdf};
    } else if (mat.type == PHONG) {
        float3 r = -wi + 2 * dot(wi, n) * n;
        float r_dot_wo = dot(r, wo);
        float n_dot_wo = dot(n, wo);
        if (r_dot_wo > 0 && n_dot_wo > 0) {
            float phong_response = 
                ((mat.phong.exponent + 1) / (2 * c_PI)) *
                pow(r_dot_wo, mat.phong.exponent);
            float3 value = 
                eval(mat.phong.reflectance, isect.uv) * 
                phong_response;
            float pdf = ((mat.phong.exponent + 1) / (2 * c_PI)) *
                pow(r_dot_wo, mat.phong.exponent);
            return {value, pdf};
        } else {
            // actual PDF may not be zero but it doesn't matter
            return {make_float3(0, 0, 0), 0};
        }
    }

    assert(false);
    return {make_float3(0, 0, 0), 0};
}


struct SamplingRecord {
    float3 wo;
    bool is_pure_specular = false;
    // Undefined if is_pure_specular = false
    float3 weight = float3{0, 0, 0}; 
};

inline __device__ SamplingRecord sample_brdf(
        const GPUScene &scene,
        const Intersection &isect,
        const float3 &n,
        const float3 &wi,
        curandState &state) {
    const Material &mat = scene.materials[isect.material_id];
    if (mat.type == DIFFUSE) {
        Frame frame(n);
        float2 u{curand_uniform(&state), curand_uniform(&state)};
        float3 wo = to_world(frame, sample_cos_hemisphere(u));
        return {wo};
    } else if (mat.type == MIRROR) {
        float3 wo = -wi + 2 * dot(wi, n) * n;
        float3 F0 = mat.mirror.reflectance;
        float3 F = schlick_fresnel(F0, dot(n, wo));
        return {wo, true /* is_pure_specular */, F};
    } else if (mat.type == PLASTIC) {
        float F0s = square((mat.plastic.eta - 1) / (mat.plastic.eta + 1));
        float3 F0{F0s, F0s, F0s};
        float3 F = schlick_fresnel(F0, dot(n, wi));
        if (curand_uniform(&state) <= F.x) {
            // Sample mirror reflection with probability F
            float3 wo = -wi + 2 * dot(wi, n) * n;
            // Weight is 1 since F cancels out
            return {wo, true, float3{1, 1, 1}};
        } else {
            // Sample diffuse with probability (1-F)
            Frame frame(n);
            float2 u{curand_uniform(&state), curand_uniform(&state)};
            float3 wo = to_world(frame, sample_cos_hemisphere(u));
            return {wo};
        }
    } else if (mat.type == PHONG) {
        float2 u{curand_uniform(&state), curand_uniform(&state)};
        float3 wo_local = sample_cos_n_hemisphere(u, mat.phong.exponent);
        Frame frame(-wi + 2 * dot(wi, n) * n);
        float3 wo_world = to_world(frame, wo_local);
        return {wo_world};
    } 
    assert(false);
    return {};
}

inline __device__ float3 float3_min(const float3 &a, const float3 &b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline __device__ float3 float3_max(const float3 &a, const float3 &b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
