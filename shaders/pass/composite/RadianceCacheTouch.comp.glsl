#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_NV_shader_subgroup_partitioned : enable

#define RC_DATA_MODIFIER restrict buffer
#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/techniques/voxel/Voxelization.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Morton.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.25, 0.25);

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
                        if (rc_touchFace(level, worldCellCoord, faceId)) {
                            rc_markScreenTouchedFace(level, worldCellCoord, faceId);
                        }
                    }
                }
            }
        }
    }
}
