#include "bvh.cuh"
#include <vector>

bool compareOnAxis(const BBoxWithID &b1, const BBoxWithID &b2, int axis) {
    float3 center1 = (b1.box.p_max + b1.box.p_min) / 2.0f;
    float3 center2 = (b2.box.p_max + b2.box.p_min) / 2.0f;
    
    switch(axis) {
        case 0: return center1.x < center2.x;
        case 1: return center1.y < center2.y;
        case 2: return center1.z < center2.z;
        default: return false;  // This should not be reached
    }
}

int construct_bvh(const std::vector<BBoxWithID> &boxes,
                  std::vector<BVHNode> &node_pool) {
    if (boxes.size() == 1) {
        BVHNode node;
        node.left_node_id = node.right_node_id = -1;
        node.primitive_id = boxes[0].id;
        node.box = boxes[0].box;
        node_pool.push_back(node);
        return node_pool.size() - 1;
    }

    BBox big_box;
    for (const BBoxWithID &b : boxes) {
        big_box = merge(big_box, b.box);
    }
    int axis = largest_axis(big_box);
    std::vector<BBoxWithID> local_boxes = boxes;
    std::sort(local_boxes.begin(), local_boxes.end(),
    [&](const BBoxWithID &b1, const BBoxWithID &b2) {
        return compareOnAxis(b1, b2, axis);
    });

    std::vector<BBoxWithID> left_boxes(
    	local_boxes.begin(),
    	local_boxes.begin() + local_boxes.size() / 2);
    std::vector<BBoxWithID> right_boxes(
    	local_boxes.begin() + local_boxes.size() / 2,
        local_boxes.end());

    BVHNode node;
    node.box = big_box;
    node.left_node_id = construct_bvh(left_boxes, node_pool);
    node.right_node_id = construct_bvh(right_boxes, node_pool);
    node.primitive_id = -1;
    node_pool.push_back(node);

    return node_pool.size() - 1;
}

int computeMaxDepth(const std::vector<BVHNode>& node_pool, int nodeIdx) {
    const BVHNode& node = node_pool[nodeIdx];

    if (node.primitive_id != -1) return 1; // leaf node
   
    int leftDepth = computeMaxDepth(node_pool, node.left_node_id);
    int rightDepth = computeMaxDepth(node_pool, node.right_node_id);

    return 1 + max(leftDepth, rightDepth);
}

__global__ void updateBVHNodeDevicePointers(BVHNode* nodes, int num_nodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_nodes) {
        return;
    }

    if (nodes[idx].left_node_id != -1) {
        nodes[idx].device_left_node = &nodes[nodes[idx].left_node_id];
    }

    if (nodes[idx].right_node_id != -1) {
        nodes[idx].device_right_node = &nodes[nodes[idx].right_node_id];
    }
}

void copyBVHNodesToDevice2(const std::vector<BVHNode> &node_pool, BVHNode* device_bvh_nodes) {
    // Step 1: Copy to a modifiable buffer
    std::vector<BVHNode> modifiable_buffer = node_pool;

    // Step 2: Modify the buffer
    for (int i = 0; i < modifiable_buffer.size(); ++i) {
        if (modifiable_buffer[i].left_node_id != -1 && modifiable_buffer[i].right_node_id != -1) {
            modifiable_buffer[i].device_left_node = &device_bvh_nodes[modifiable_buffer[i].left_node_id];
            modifiable_buffer[i].device_right_node = &device_bvh_nodes[modifiable_buffer[i].right_node_id];
        }
    }

    // Step 3: Copy the modified buffer to device memory
    checkCudaErrors(cudaMemcpy(device_bvh_nodes, modifiable_buffer.data(), modifiable_buffer.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
}

