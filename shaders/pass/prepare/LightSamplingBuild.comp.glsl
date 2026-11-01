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
#elif LS_BUILD_STAGE == 2
#if VOXEL_GRID_SIZE == 64
#if LS_TREE_ARITY == 2
#define LS_WORLD_GROUPS LS_GROUPS_64_2
#elif LS_TREE_ARITY == 4
#define LS_WORLD_GROUPS LS_GROUPS_64_4
#else
#define LS_WORLD_GROUPS LS_GROUPS_64_8
#endif
#elif VOXEL_GRID_SIZE == 32
#if LS_TREE_ARITY == 2
#define LS_WORLD_GROUPS LS_GROUPS_32_2
#elif LS_TREE_ARITY == 4
#define LS_WORLD_GROUPS LS_GROUPS_32_4
#else
#define LS_WORLD_GROUPS LS_GROUPS_32_8
#endif
#else
#if LS_TREE_ARITY == 2
#define LS_WORLD_GROUPS LS_GROUPS_16_2
#elif LS_TREE_ARITY == 4
#define LS_WORLD_GROUPS LS_GROUPS_16_4
#else
#define LS_WORLD_GROUPS LS_GROUPS_16_8
#endif
#endif
const ivec3 workGroups = ivec3(LS_WORLD_GROUPS, 1, 1);
#else
const ivec3 workGroups = ivec3(1, 1, 1);
shared uvec3 ls_topKeys[16];
shared uint ls_topNodes[16];
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
    for (uint i = gl_GlobalInvocationID.x; i < LS_OUTER_NODE_COUNT; i += 65536u) ls_nodes[i] = ls_emptyNode();
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
    if (lane == 0u) {
        ls_nodes[ls_worldAddress(ls_unsignedBrick(brickOrigin >> 4), 0u)] = ls_nodes[base + 1u];
    }
    #elif LS_BUILD_STAGE == 2
    uvec3 low, dimensions;
    ls_worldRange(uint(LS_WORLD_LEVEL), low, dimensions);
    uint index = gl_GlobalInvocationID.x;
    if (index < dimensions.x * dimensions.y * dimensions.z) {
        uint childBase = ls_levelOffset(uint(LS_WORLD_LEVEL)) + index * uint(LS_TREE_ARITY);
        LightSamplingNode merged = ls_emptyNode();
        uint nonempty = 0u;
        uint onlyChild = 0u;
        for (uint j = 0u; j < uint(LS_TREE_ARITY); ++j) {
            LightSamplingNode child = ls_nodes[childBase + j];
            if (child.boundsMinMass.w == 0.0) continue;
            merged = ls_merge(merged, child, 0u);
            onlyChild = childBase + j;
            ++nonempty;
        }
        if (nonempty == 1u) merged = ls_nodes[onlyChild];
        if (nonempty > 1u) merged.boundsMaxLink.w = uintBitsToFloat(ls_childLink(childBase, uint(LS_TREE_ARITY)));
        uvec3 cell = ls_worldCell(index, low, dimensions);
        ls_nodes[ls_worldAddress(cell << ls_worldShift(uint(LS_WORLD_LEVEL)), uint(LS_WORLD_LEVEL))] = merged;
    }
    #else
    if (gl_GlobalInvocationID.x != 0u) return;
    uvec3 low, dimensions;
    ls_worldRange(uint(LS_WORLD_LEVELS), low, dimensions);
    uint count = 0u;
    for (uint i = 0u; i < dimensions.x * dimensions.y * dimensions.z; ++i) {
        if (ls_nodes[2u + i].boundsMinMass.w == 0.0) continue;
        ls_topKeys[count] = ls_worldCell(i, low, dimensions);
        ls_topNodes[count++] = 2u + i;
    }
    uint readBase = 0u;
    uint nextChild = 10u;
    for (uint level = uint(LS_WORLD_LEVELS) + 1u; count > 1u && level <= uint(84 / LS_TREE_BITS); ++level) {
        uint writeBase = 8u - readBase;
        uint nextCount = 0u;
        uvec3 previousShift = ls_worldShift(level - 1u);
        uvec3 shiftDelta = ls_worldShift(level) - previousShift;
        for (uint i = 0u; i < count; ++i) {
            uvec3 parent = ls_topKeys[readBase + i] >> shiftDelta;
            bool seen = false;
            for (uint j = 0u; j < i; ++j) seen = seen || all(equal(ls_topKeys[readBase + j] >> shiftDelta, parent));
            if (seen) continue;
            uint members = 0u;
            uint onlyChild = 0u;
            for (uint j = i; j < count; ++j) {
                if (any(notEqual(ls_topKeys[readBase + j] >> shiftDelta, parent))) continue;
                ++members;
                onlyChild = ls_topNodes[readBase + j];
            }
            LightSamplingNode merged;
            if (members == 1u) {
                merged = ls_nodes[onlyChild];
            } else {
                merged = ls_emptyNode();
                uint childBase = nextChild;
                for (uint ordinal = 0u; ordinal < uint(LS_TREE_ARITY); ++ordinal) {
                    for (uint j = i; j < count; ++j) {
                        uvec3 key = ls_topKeys[readBase + j];
                        if (any(notEqual(key >> shiftDelta, parent))) continue;
                        if (ls_worldOrdinal(key << previousShift, level) != ordinal) continue;
                        LightSamplingNode child = ls_nodes[ls_topNodes[readBase + j]];
                        ls_nodes[nextChild++] = child;
                        merged = ls_merge(merged, child, 0u);
                    }
                }
                merged.boundsMaxLink.w = uintBitsToFloat(ls_childLink(childBase, members));
            }
            // Compact toward already consumed inputs; later groups retain their input records.
            ls_nodes[2u + nextCount] = merged;
            ls_topKeys[writeBase + nextCount] = parent;
            ls_topNodes[writeBase + nextCount] = 2u + nextCount;
            ++nextCount;
        }
        readBase = writeBase;
        count = nextCount;
        memoryBarrierBuffer();
    }
    ls_nodes[1] = count == 0u ? ls_emptyNode() : ls_nodes[ls_topNodes[readBase]];
    #endif
}
