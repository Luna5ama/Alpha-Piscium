#include "/techniques/debug/LightSampling.glsl"

#if defined(SETTING_DEBUG_LIGHT_WEIGHT) && LS_BUILD_STAGE == 1
#include "/techniques/voxel/SurfaceData.glsl"

float ls_emissionMass(uint materialID) {
    GBufferData data = gbufferData_init();
    data.materialID = materialID;
    float mass = 0.0;
    for (uint face = 0u; face < 6u; ++face) {
        vec4 tc = unpackUnorm4x16(voxel_faceTexcoords[voxel_faceTexcoordIndex(materialID, face)]);
        for (uint sampleIndex = 0u; sampleIndex < 16u; ++sampleIndex) {
            vec2 uv = (vec2(sampleIndex & 3u, sampleIndex >> 2u) + 0.5) * 0.25;
            data.albedo = textureLod(usam_blockAtlasColor, mix(tc.xw, tc.zy, uv), 0.0).rgb;
            vec3 emission = material_decode(data).emissive;
            vec3 linearSRGB = colors_colorspaces_convert(COLORS_WORKING_COLORSPACE, COLORS_COLORSPACES_SRGB, emission);
            mass += dot(linearSRGB, vec3(0.2126, 0.7152, 0.0722));
        }
    }
    return mass / 96.0;
}
#endif

layout(local_size_x = 256) in;
#if LS_BUILD_STAGE == 0
const ivec3 workGroups = ivec3(256, 1, 1);
#elif LS_BUILD_STAGE == 1
const ivec3 workGroups = ivec3(SETTING_VOXEL_POOL_SIZE, 1, 1);
shared uint ls_spatial[8192];
shared uint ls_next;
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
        ls_reserved = 0u;
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
    for (uint width = 2048u; width != 0u; width >>= 1u) {
        for (uint i = lane; i < width; i += 256u) {
            uint node = width + i;
            ls_spatial[node] = ls_spatial[node * 2u] + ls_spatial[node * 2u + 1u];
        }
        barrier();
    }
    if (lane == 0u) {
        uint count = ls_spatial[1];
        ls_padded = count;
        ls_base = atomicAdd(ls_allocatedNodes, count * 2u);
        atomicAdd(ls_emitterCount, count);
        if (ls_base + count * 2u > LS_LOCAL_NODE_CAPACITY) atomicAdd(ls_overflowCount, count);
        ls_spatial[1] |= 1u << 13u;
        ls_next = 2u;
    }
    barrier();
    if (ls_padded == 0u || ls_base + ls_padded * 2u > LS_LOCAL_NODE_CAPACITY) return;
    uint base = LS_OUTER_NODE_COUNT + ls_base;
    for (uint width = 1u; width < 4096u; width <<= 1u) {
        for (uint i = lane; i < width; i += 256u) {
            uint node = width + i;
            if ((ls_spatial[node] & 8191u) == 0u) continue;
            uint slot = ls_spatial[node] >> 13u;
            uint left = ls_spatial[node * 2u];
            uint right = ls_spatial[node * 2u + 1u];
            if (left != 0u && right != 0u) {
                uint child = atomicAdd(ls_next, 2u);
                ls_spatial[node * 2u] |= child << 13u;
                ls_spatial[node * 2u + 1u] |= (child + 1u) << 13u;
            } else {
                ls_spatial[node * 2u + uint(left == 0u)] |= slot << 13u;
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
        #ifdef SETTING_DEBUG_LIGHT_WEIGHT
        mass = max(ls_emissionMass(materialID), mass * 0.000001);
        #endif
        uint slot = ls_spatial[4096u + i] >> 13u;
        ls_nodes[base + slot] = LightSamplingNode(
            vec4(position, mass), vec4(position + 1.0, uintBitsToFloat(LS_LEAF_BIT | materialID))
        );
    }
    memoryBarrierBuffer();
    barrier();
    for (uint width = 2048u; width != 0u; width >>= 1u) {
        for (uint i = lane; i < width; i += 256u) {
            uint node = width + i;
            uint left = ls_spatial[node * 2u];
            uint right = ls_spatial[node * 2u + 1u];
            if ((left & 8191u) == 0u || (right & 8191u) == 0u) continue;
            uint child = base + (left >> 13u);
            ls_nodes[base + (ls_spatial[node] >> 13u)] = ls_merge(ls_nodes[child], ls_nodes[child + 1u], child);
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
