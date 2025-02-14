#include "scene.h"

#include "flexception.h"

#include "texture.h"
#include <algorithm>


// Scene Constructor

Scene::Scene(const ParsedScene &scene) :
        camera(from_parsed_camera(scene.camera)),
        width(scene.camera.width),
        height(scene.camera.height),
        background_color(scene.background_color),
        samples_per_pixel(scene.samples_per_pixel) {
    
    // Extract triangle meshes from the parsed scene.
    int tri_mesh_count = 0;
    for (const ParsedShape &parsed_shape : scene.shapes) {
        if (std::get_if<ParsedTriangleMesh>(&parsed_shape)) {
            tri_mesh_count++;
        }
    }
    meshes.resize(tri_mesh_count);
    
    // Extract the shapes
    tri_mesh_count = 0;

    for (int i = 0; i < (int)scene.shapes.size(); i++) {
        const ParsedShape &parsed_shape = scene.shapes[i];
        if (auto *sph = std::get_if<ParsedSphere>(&parsed_shape)) {
            
            if (sph->area_light_id >= 0) {
                //printf("Pushing sphere with area light!\n");
                const ParsedDiffuseAreaLight &light =
                    std::get<ParsedDiffuseAreaLight>(
                        scene.lights[sph->area_light_id]);
                lights.push_back(DiffuseAreaLight{
                    (int)shapes.size(), light.radiance});
                    //printf("Light information: radiance (%f, %f, %f)\n", light.radiance.x, light.radiance.y, light.radiance.z);
            }
            shapes.push_back(
                Sphere{{sph->material_id, sph->area_light_id},
                    sph->position, sph->radius});
            //printf("Sphere information: material_id (%d), area_id (%d), position (%f, %f, %f), radius (%f)\n", sph->material_id, sph->area_light_id, sph->position.x, sph->position.y, sph->position.z, sph->radius);
        } else if (auto *parsed_mesh = std::get_if<ParsedTriangleMesh>(&parsed_shape)) {
        // Create new arrays on the host
        float3* positionsHost = new float3[parsed_mesh->positions.size()];
        int3* indicesHost = new int3[parsed_mesh->indices.size()];
        float3* normalsHost = new float3[parsed_mesh->normals.size()];
        float2* uvsHost = new float2[parsed_mesh->uvs.size()];

        // Copy data from vectors to arrays
        std::copy(parsed_mesh->positions.begin(), parsed_mesh->positions.end(), positionsHost);
        std::copy(parsed_mesh->indices.begin(), parsed_mesh->indices.end(), indicesHost);
        std::copy(parsed_mesh->normals.begin(), parsed_mesh->normals.end(), normalsHost);
        std::copy(parsed_mesh->uvs.begin(), parsed_mesh->uvs.end(), uvsHost);

        // Create a TriangleMesh on the host
        TriangleMesh meshHost = {
            {parsed_mesh->material_id, parsed_mesh->area_light_id},
            positionsHost, indicesHost,
            normalsHost, uvsHost,
            nullptr, nullptr, nullptr, nullptr,  // These are the device pointers, initially set to nullptr
            parsed_mesh->positions.size(), parsed_mesh->indices.size()  // Here we set the sizes
        };

        // Copy the TriangleMesh to the device
        copyTriangleMeshToDevice(meshHost);

        // Store the TriangleMesh into the mesh list
        meshes[tri_mesh_count] = meshHost;

        // Extract all the individual triangles
        for (int face_index = 0; face_index < (int)parsed_mesh->indices.size(); face_index++) {
            if (parsed_mesh->area_light_id >= 0) {
                //printf("Pushing triangle with area light!\n");
                const ParsedDiffuseAreaLight &light =
                    std::get<ParsedDiffuseAreaLight>(
                        scene.lights[parsed_mesh->area_light_id]);
                lights.push_back(DiffuseAreaLight{
                    (int)shapes.size(), light.radiance});
                //printf("Light information: radiance (%f, %f, %f)\n", light.radiance.x, light.radiance.y, light.radiance.z);
            }
            shapes.push_back(Triangle{face_index, tri_mesh_count});  // Note the change here
        }
        tri_mesh_count++;
    }
        else {
            assert(false);
        }
    }

    // Copy the materials
    for (const ParsedMaterial &parsed_mat : scene.materials) {
        if (auto *diffuse = std::get_if<ParsedDiffuse>(&parsed_mat)) {
            materials.push_back(Diffuse{std::get<float3>(diffuse->reflectance)});
        } else if (auto *mirror = std::get_if<ParsedMirror>(&parsed_mat)) {
            materials.push_back(Mirror{std::get<float3>(mirror->reflectance)});
        }
        else if (auto *plastic = std::get_if<ParsedPlastic>(&parsed_mat)) {
            materials.push_back(Plastic{
                plastic -> eta,
                std::get<float3>(plastic->reflectance)});
        }
        else if (auto *phong = std::get_if<ParsedPhong>(&parsed_mat)) {
            materials.push_back(Phong{
                std::get<float3>(phong->reflectance),
                phong -> exponent});
        }
    }
 
    // Copy the lights 
    for (const ParsedLight &parsed_light : scene.lights) {
        if (auto *point_light = std::get_if<ParsedPointLight>(&parsed_light)) {
            lights.push_back(PointLight{point_light->intensity, point_light->position});
            //printf("Point Light information: intensity (%f, %f, %f), position (%f, %f, %f)\n", point_light->intensity.x, point_light->intensity.y, point_light->intensity.z, point_light->position.x, point_light->position.y, point_light->position.z);
        } 
    }

    
    // Build BVH on the host
    std::vector<BBoxWithID> bboxes_host(shapes.size());

    for (int i = 0; i < (int)bboxes_host.size(); i++) {
        if (shapes[i].type == SPHERE) {
            const Sphere& sph = shapes[i].sphere;
            float3 p_min = sph.center - sph.radius;
            float3 p_max = sph.center + sph.radius;
            bboxes_host[i] = {BBox{p_min, p_max}, i};
        } else if (shapes[i].type == TRIANGLE) {
            const Triangle& tri = shapes[i].triangle;
            const TriangleMesh *mesh = &meshes[tri.mesh_index];
            int3 index = mesh->host_indices[tri.face_index];
            float3 p0 = mesh->host_positions[index.x];
            float3 p1 = mesh->host_positions[index.y];
            float3 p2 = mesh->host_positions[index.z];
            float3 p_min = float3_min(float3_min(p0, p1), p2);
            float3 p_max = float3_max(float3_max(p0, p1), p2);
            bboxes_host[i] = {BBox{p_min, p_max}, i}; 
        }
    }

    bvh_root_id = construct_bvh(bboxes_host, CPU_bvh_nodes);

    // Compute and print maximum BVH depth
    int maxDepth = computeMaxDepth(CPU_bvh_nodes, bvh_root_id);
    printf("Maximum BVH depth: %d\n", maxDepth);

    cleanupHostMemory();

}
