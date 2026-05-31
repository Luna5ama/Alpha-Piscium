#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#define RC_DATA_MODIFIER restrict buffer
#include "/Base.glsl"
#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/techniques/voxel/Voxelization.glsl"
#include "/util/GBufferData.glsl"
#include "/util/MaterialIDConst.glsl"
#include "/util/Morton.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

bool rcVoxelOpaqueAtBlock(ivec3 worldBlockPos) {
    ivec3 cameraBrick = cameraPositionInt >> 4;
    ivec3 gridOrigin = (cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4;
    ivec3 gridBlockPos = worldBlockPos - gridOrigin;
    if (any(lessThan(gridBlockPos, ivec3(0))) || any(greaterThanEqual(gridBlockPos, ivec3(VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE)))) {
        return false;
    }

    ivec3 brickCoord = gridBlockPos >> 4;
    uint brickMorton = voxel_brickMorton(brickCoord);
    uint allocID = voxel_brickAllocID[brickMorton];
    if (allocID == VOXEL_UNALLOCATED) {
        return false;
    }

    ivec3 blockInBrick = gridBlockPos & 15;
    uint blockMorton = voxel_blockMorton(blockInBrick);
    uint materialID = voxel_materials[voxel_materialIndex(allocID, blockMorton)];
    return materialID != 0u && materialID != MATERIAL_ID_WATER;
}

void rcTouchFace(uint level, ivec3 worldCellCoord, uint faceId) {
    uint entryIndex = rcEntryIndex(level, worldCellCoord);
    uint bufferIndex = rcBufferEntryIndex(rcCurrentSide(), entryIndex);
    uint worldKeyHash = rcWorldKeyHash(level, worldCellCoord);
    uint oldKey = atomicCompSwap(rc_indirection[bufferIndex].z, RC_INVALID, worldKeyHash);
    if (oldKey == RC_INVALID || oldKey == worldKeyHash) {
        atomicOr(rc_indirection[bufferIndex].y, rcFaceBit(faceId));
        rc_indirection[bufferIndex].w = rcPackEntryMeta(level, 0u, true);
    } else {
        atomicAdd(rc_keyMismatchCounter, 1u);
    }
}

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    ivec2 texelPos = ivec2(workGroupOrigin + mortonPos);

    if (!all(lessThan(texelPos, uval_mainImageSizeI))) {
        return;
    }

    float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos.xy, 4, texelPos);
    if (viewZ <= -65536.0) {
        return;
    }

    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), gData);
    if (gData.materialID == 0u || gData.materialID == MATERIAL_ID_WATER || gData.materialID >= 65533u) {
        return;
    }

    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec3 feetPlayerPos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 worldPos = feetPlayerPos + cameraPosition;
    vec3 worldGeomNormal = coords_dir_viewToWorld(gData.geomNormal);
    uint faceId = rcFaceIdFromNormal(worldGeomNormal);
    ivec3 faceNormalI = rcFaceNormalI(faceId);

    ivec3 ownerBlock = ivec3(floor(worldPos - rcFaceNormal(faceId) * 0.02));
    bool neighborOpen = !rcVoxelOpaqueAtBlock(ownerBlock + faceNormalI);
    if (!neighborOpen) {
        return;
    }

    for (uint level = 0u; level < RC_CLIP_LEVELS; level++) {
        ivec3 worldCellCoord = rcWorldCellCoord(worldPos - rcFaceNormal(faceId) * 0.02, level);
        rcTouchFace(level, worldCellCoord, faceId);
    }
}
