#include "/techniques/debug/LightSampling.glsl"

layout(local_size_x = 256) in;
#if LS_BUILD_STAGE == 0
const ivec3 workGroups = ivec3(256, 1, 1);
#elif LS_BUILD_STAGE == 1
const ivec3 workGroups = ivec3(SETTING_VOXEL_POOL_SIZE, 1, 1);
shared uint ls_scan[4096];
shared uint ls_base;
shared uint ls_padded;
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
        ls_guidedSelections = 0u;
    }
    for (uint i = gl_GlobalInvocationID.x; i < uint(VOXEL_GRID_BRICKS); i += 65536u) {
        ls_nodes[uint(VOXEL_GRID_BRICKS) + i] = ls_emptyNode();
    }
    #elif LS_BUILD_STAGE == 1
    uint allocation = gl_WorkGroupID.x;
    if (allocation >= voxel_brickAllocCounter) return;
    uint lane = gl_LocalInvocationID.x;
    for (uint i = lane; i < 4096u; i += 256u) {
        uint materialID = voxel_decodeMaterialID(voxel_materials[allocation * 4096u + i]);
        ls_scan[i] = uint(ls_isEmitter(materialID));
    }
    barrier();
    // Blelloch exclusive scan keeps the compact leaves in block Morton order.
    for (uint stride = 1u; stride < 4096u; stride <<= 1u) {
        for (uint i = (lane + 1u) * stride * 2u - 1u; i < 4096u; i += 512u * stride) {
            ls_scan[i] += ls_scan[i - stride];
        }
        barrier();
    }
    if (lane == 0u) {
        uint count = ls_scan[4095];
        ls_padded = count == 0u ? 0u : 1u << uint(findMSB(count - 1u) + 1);
        ls_base = atomicAdd(ls_allocatedNodes, ls_padded * 2u);
        atomicAdd(ls_emitterCount, count);
        if (ls_base + ls_padded * 2u > LS_LOCAL_NODE_CAPACITY) atomicAdd(ls_overflowCount, count);
        ls_scan[4095] = 0u;
    }
    barrier();
    if (ls_padded == 0u || ls_base + ls_padded * 2u > LS_LOCAL_NODE_CAPACITY) return;
    for (uint stride = 2048u; stride != 0u; stride >>= 1u) {
        for (uint i = (lane + 1u) * stride * 2u - 1u; i < 4096u; i += 512u * stride) {
            uint prefix = ls_scan[i - stride];
            ls_scan[i - stride] = ls_scan[i];
            ls_scan[i] += prefix;
        }
        barrier();
    }
    uint base = LS_OUTER_NODE_COUNT + ls_base;
    for (uint i = lane; i < ls_padded; i += 256u) ls_nodes[base + ls_padded + i] = ls_emptyNode();
    memoryBarrierBuffer();
    barrier();
    uint brickMorton = voxel_allocatedBricks[allocation];
    ivec3 gridOrigin = ((cameraPositionInt >> 4) - ivec3(VOXEL_GRID_SIZE / 2)) << 4;
    ivec3 brickOrigin = gridOrigin + ivec3(morton3D_30bDecode(brickMorton) << 4u);
    for (uint i = lane; i < 4096u; i += 256u) {
        uint materialID = voxel_decodeMaterialID(voxel_materials[allocation * 4096u + i]);
        if (!ls_isEmitter(materialID)) continue;
        HardcodedPBR pbr = hardcodedpbr_decode(materialID);
        vec3 position = vec3(brickOrigin + ivec3(morton3D_30bDecode(i)));
        float mass = ldexp(pow(pbr.emissive, SETTING_EMISSIVE_PBR_VALUE_CURVE), pbr.emissiveMultiplier);
        ls_nodes[base + ls_padded + ls_scan[i]] = LightSamplingNode(
            vec4(position, mass), vec4(position + 1.0, uintBitsToFloat(LS_LEAF_BIT | materialID))
        );
    }
    memoryBarrierBuffer();
    barrier();
    for (uint width = ls_padded >> 1u; width != 0u; width >>= 1u) {
        for (uint i = lane; i < width; i += 256u) {
            uint node = base + width + i;
            uint child = base + (width + i) * 2u;
            ls_nodes[node] = ls_merge(ls_nodes[child], ls_nodes[child + 1u], child);
        }
        memoryBarrierBuffer();
        barrier();
    }
    if (lane == 0u) ls_nodes[uint(VOXEL_GRID_BRICKS) + brickMorton] = ls_nodes[base + 1u];
    #else
    uint first = uint(VOXEL_GRID_BRICKS) >> LS_REDUCE_LEVEL;
    uint node = first + gl_GlobalInvocationID.x;
    if (node < first * 2u) ls_nodes[node] = ls_merge(ls_nodes[node * 2u], ls_nodes[node * 2u + 1u], node * 2u);
    #endif
}
