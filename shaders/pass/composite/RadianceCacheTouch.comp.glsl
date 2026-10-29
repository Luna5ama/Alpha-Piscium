#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#ifdef MC_GL_VENDOR_NVIDIA
#extension GL_NV_shader_subgroup_partitioned : enable
#endif

#define RC_DATA_MODIFIER restrict buffer
#define GLOBAL_DATA_MODIFIER restrict buffer

layout(local_size_x = 16, local_size_y = 16) in;
#include "/techniques/gi/RadianceCacheUpdate.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/techniques/voxel/Voxelization.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Morton.glsl"
#include "/util/ThreadGroupTiling.glsl"

const vec2 workGroupsRender = vec2(0.25, 0.25);

void rc_touchFace(uint level, ivec3 worldCellCoord, uint faceId) {
    if (!rc_worldCellInCurrentClip(level, worldCellCoord)) {
        return;
    }

    uint entryIndex = rc_entryIndex(level, worldCellCoord);
    #ifdef MC_GL_VENDOR_NVIDIA
    uvec4 matchingFaces = subgroupPartitionNV((entryIndex << 3u) | faceId);
    if (subgroupBallotFindLSB(matchingFaces) != gl_SubgroupInvocationID) {
        return;
    }
    #endif

    uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);
    uint faceBit = rc_faceBit(faceId);
    uvec4 entry = rc_indirection[bufferIndex];
    uint oldKey = entry.z;
    bool pendingEntry = rc_entryMetaPendingFaceMask(entry.w) != 0u;
    if (pendingEntry) {
        oldKey = worldKeyHash;
    } else if (oldKey == RC_INVALID) {
        oldKey = atomicCompSwap(rc_indirection[bufferIndex].z, RC_INVALID, worldKeyHash);
    }
    if (oldKey == RC_INVALID || oldKey == worldKeyHash) {
        // PreClear invalidates this side before pending and screen faces are collected.
        if ((entry.y & faceBit) == 0u) {
            atomicOr(rc_indirection[bufferIndex].y, faceBit);
        }
        uint touchedMeta = rc_packEntryMeta(level, true) | rc_entryMetaScreenFaceBits(faceBit) | RC_ENTRY_META_LISTED;
        if ((entry.w & touchedMeta) != touchedMeta) {
            rc_markEntryListed(rc_currentSide(), entryIndex, touchedMeta);
        }
    } else {
        atomicAdd(rc_keyMismatchCounter, 1u);
    }
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy) << 2;
    texelPos += ivec2(morton_8bDecode(uint(frameCounter + morton_32bEncode(gl_GlobalInvocationID.xy)) & 15u));

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).r;
        if (viewZ > -65536.0) {
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), gData);

            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

            if (gData.materialID != 0u && gData.materialID != MATERIAL_ID_WATER && gData.materialID < 65533u) {
                vec3 scenePos = coords_pos_viewToWorld(viewPos - gData.geomNormal * 0.02, gbufferModelViewInverse);
                vec3 worldPos = scenePos + cameraPosition;
                vec3 worldGeomNormal = coords_dir_viewToWorld(gData.geomNormal);
                uint faceId = rc_faceIdFromNormal(worldGeomNormal);
                ivec3 faceNormalI = rc_faceNormalI(faceId);

                ivec3 ownerBlock = ivec3(floor(worldPos));
                bool neighborOpen = !voxel_opaqueAtBlock(ownerBlock + faceNormalI);
                if (neighborOpen) {
                    for (uint level = 0u; level < RC_CLIP_LEVELS; level++) {
                        ivec3 worldCellCoord = rc_worldCellCoord(worldPos, level);
                        rc_touchFace(level, worldCellCoord, faceId);
                    }
                }
            }
        }
    }
}
