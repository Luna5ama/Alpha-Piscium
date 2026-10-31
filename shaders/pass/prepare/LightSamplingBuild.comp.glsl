#include "/techniques/debug/LightSampling.glsl"

layout(local_size_x = 256) in;
#if LS_BUILD_STAGE == 0
const ivec3 workGroups = ivec3(256, 1, 1);
#elif LS_BUILD_STAGE == 1
const ivec3 workGroups = ivec3(SETTING_VOXEL_POOL_SIZE, 1, 1);
shared uint ls_spatial[8192];
shared uint ls_next;
shared uint ls_base;
shared uint ls_leafCount;
#else
const ivec3 workGroups = ivec3(LS_REDUCE_GROUPS, 1, 1);
#endif

void main() {
    #if LS_BUILD_STAGE == 0
    if (gl_GlobalInvocationID.x == 0u) {
        ls_allocatedNodes = 0u;
        ls_emitterCount = 0u;
        ls_overflowCount = 0u;
        ls_exhaustedCount = 0u;
        ls_traceCount = 0u;
        ls_visibleCount = 0u;
        ls_receiverCount = 0u;
        ls_reserved = 0u;
        #ifdef SETTING_DEBUG_LIGHT_AUDIT
        for (uint i = 0u; i < 16u; ++i) ls_reasons[i] = 0u;
        #endif
    }
    for (uint i = gl_GlobalInvocationID.x; i < uint(VOXEL_GRID_BRICKS); i += 65536u) {
        ls_nodes[uint(VOXEL_GRID_BRICKS) + i] = ls_emptyNode();
    }
    #elif LS_BUILD_STAGE == 1
    uint allocation = gl_WorkGroupID.x;
    if (allocation >= voxel_brickAllocCounter) return;
    uint lane = gl_LocalInvocationID.x;
    // Low 13 bits hold occupancy; upper bits hold the compact output slot.
    for (uint i = lane; i < 4096u; i += 256u) {
        uint materialID = voxel_decodeMaterialID(voxel_materials[allocation * 4096u + i]);
        ls_spatial[4096u + i] = uint(ls_isEmitter(materialID));
    }
    barrier();
    for (uint width = 4096u / uint(LS_TREE_ARITY); width != 0u; width >>= LS_TREE_BITS) {
        for (uint i = lane; i < width; i += 256u) {
            uint node = width + i;
            uint count = 0u;
            for (uint j = 0u; j < uint(LS_TREE_ARITY); ++j) count += ls_spatial[node * uint(LS_TREE_ARITY) + j];
            ls_spatial[node] = count;
        }
        barrier();
    }
    if (lane == 0u) {
        uint count = ls_spatial[1];
        ls_leafCount = count;
        // Keep the binary baseline's emitter capacity budget; unused slots stay reserved.
        ls_base = atomicAdd(ls_allocatedNodes, count * 2u);
        atomicAdd(ls_emitterCount, count);
        if (ls_base + count * 2u > LS_LOCAL_NODE_CAPACITY) atomicAdd(ls_overflowCount, count);
        ls_spatial[1] |= 1u << 13u;
        ls_next = 2u;
    }
    barrier();
    if (ls_leafCount == 0u || ls_base + ls_leafCount * 2u > LS_LOCAL_NODE_CAPACITY) return;
    uint base = LS_OUTER_NODE_COUNT + ls_base;
    for (uint width = 1u; width < 4096u; width <<= LS_TREE_BITS) {
        for (uint i = lane; i < width; i += 256u) {
            uint node = width + i;
            if ((ls_spatial[node] & 8191u) == 0u) continue;
            uint slot = ls_spatial[node] >> 13u;
            uint first = node * uint(LS_TREE_ARITY);
            uint count = 0u;
            for (uint j = 0u; j < uint(LS_TREE_ARITY); ++j) count += uint(ls_spatial[first + j] != 0u);
            uint child = count > 1u ? atomicAdd(ls_next, count) : slot;
            for (uint j = 0u; j < uint(LS_TREE_ARITY); ++j) {
                if (ls_spatial[first + j] != 0u) ls_spatial[first + j] |= child++ << 13u;
            }
        }
        barrier();
    }
    uint brickMorton = voxel_allocatedBricks[allocation];
    ivec3 gridOrigin = ((cameraPositionInt >> 4) - ivec3(VOXEL_GRID_SIZE / 2)) << 4;
    ivec3 brickOrigin = gridOrigin + ivec3(morton3D_30bDecode(brickMorton) << 4u);
    for (uint i = lane; i < 4096u; i += 256u) {
        if ((ls_spatial[4096u + i] & 8191u) == 0u) continue;
        uint materialID = voxel_decodeMaterialID(voxel_materials[allocation * 4096u + i]);
        HardcodedPBR pbr = hardcodedpbr_decode(materialID);
        vec3 position = vec3(brickOrigin + ivec3(morton3D_30bDecode(i)));
        float mass = ldexp(pow(pbr.emissive, SETTING_EMISSIVE_PBR_VALUE_CURVE), pbr.emissiveMultiplier);
        uint slot = ls_spatial[4096u + i] >> 13u;
        LightSamplingNode leaf = ls_emptyNode();
        leaf.boundsMinMass = vec4(position, mass);
        leaf.boundsMaxLink = vec4(position + 1.0, uintBitsToFloat(LS_LEAF_BIT | materialID));
        #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 4 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 5
        leaf.meanVariance = vec4(position + 0.5, 5.0 / 12.0);
        #endif
        ls_nodes[base + slot] = leaf;
    }
    memoryBarrierBuffer();
    barrier();
    for (uint width = 4096u / uint(LS_TREE_ARITY); width != 0u; width >>= LS_TREE_BITS) {
        for (uint i = lane; i < width; i += 256u) {
            uint node = width + i;
            LightSamplingNode merged = ls_emptyNode();
            uint childBase = 0u;
            uint count = 0u;
            for (uint j = 0u; j < uint(LS_TREE_ARITY); ++j) {
                uint entry = ls_spatial[node * uint(LS_TREE_ARITY) + j];
                if ((entry & 8191u) == 0u) continue;
                uint child = base + (entry >> 13u);
                if (count == 0u) childBase = child;
                merged = ls_merge(merged, ls_nodes[child], 0u);
                ++count;
            }
            if (count > 1u) {
                merged.boundsMaxLink.w = uintBitsToFloat(ls_childLink(childBase, count));
                ls_nodes[base + (ls_spatial[node] >> 13u)] = merged;
            }
        }
        memoryBarrierBuffer();
        barrier();
    }
    if (lane == 0u) ls_nodes[uint(VOXEL_GRID_BRICKS) + brickMorton] = ls_nodes[base + 1u];
    #else
    #if LS_REDUCE_LEVEL % LS_TREE_BITS == 0 || (VOXEL_GRID_BRICKS >> LS_REDUCE_LEVEL) == 1
    // A 32^3 grid has 15 Morton bits: its quaternary root has two children.
    #if LS_REDUCE_LEVEL % LS_TREE_BITS == 0
    const uint count = uint(LS_TREE_ARITY);
    #else
    const uint count = 1u << (LS_REDUCE_LEVEL % LS_TREE_BITS);
    #endif
    uint first = uint(VOXEL_GRID_BRICKS) >> LS_REDUCE_LEVEL;
    uint node = first + gl_GlobalInvocationID.x;
    if (node < first * 2u) {
        uint child = node * count;
        LightSamplingNode merged = ls_emptyNode();
        for (uint j = 0u; j < count; ++j) merged = ls_merge(merged, ls_nodes[child + j], 0u);
        merged.boundsMaxLink.w = uintBitsToFloat(ls_childLink(child, count));
        ls_nodes[node] = merged;
    }
    #endif
    #endif
}
