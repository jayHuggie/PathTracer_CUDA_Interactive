#pragma once
#ifndef BVH_CUH
#define BVH_CUH

#include "bbox.cuh"

struct BVHNode {
    BBox box;
    int left_node_id;
    int right_node_id;
    int primitive_id;

    BVHNode* device_left_node;
    BVHNode* device_right_node;
};


struct BBoxWithID {
    BBox box;
    int id;
};

bool compareOnAxis(const BBoxWithID &b1, const BBoxWithID &b2, int axis);

int computeMaxDepth(const std::vector<BVHNode>& node_pool, int nodeIdx);

int construct_bvh(const std::vector<BBoxWithID> &boxes, std::vector<BVHNode> &node_pool);


void copyBVHNodesToDevice2(const std::vector<BVHNode> &node_pool, BVHNode* device_bvh_nodes);

inline void freeBVHNodeFromDevice(BVHNode &node) {
    if (node.device_left_node) {
        checkCudaErrors(cudaFree(node.device_left_node));
    }
    if (node.device_right_node) {
        checkCudaErrors(cudaFree(node.device_right_node));
    }
}

#endif // BVH_CUH