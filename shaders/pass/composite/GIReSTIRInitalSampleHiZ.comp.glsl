#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(r32ui) uniform restrict writeonly uimage2D uimg_r32ui;
layout(rg32ui) uniform restrict writeonly uimage2D uimg_rg32ui;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(rgba8) uniform restrict writeonly image2D uimg_rgba8;

const vec2 workGroupsRender = vec2(1.0, 1.0);

#define RESTIR_INITIAL_CANDIDATE_WRITE
#include "/techniques/HiZCheck.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/InitialSample.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Morton.glsl"
#include "/util/ThreadGroupTiling.glsl"

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    ivec2 texelPos = ivec2(workGroupOrigin + mortonPos);

    restir_InitialCandidate candidate = restir_initialCandidate_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);

        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), gData);
            Material material = material_decode(gData);

            transient_restir_resampleMaterial_store(texelPos, resampleMaterial_pack(resampleMaterial_fromMaterial(material)));
            transient_solidAlbedo_store(texelPos, vec4(gData.albedo, gData.pbrSpecular.a));
            transient_geomViewNormal_store(texelPos, vec4(gData.geomNormal * 0.5 + 0.5, 0.0));
            transient_viewNormal_store(texelPos, vec4(gData.normal * 0.5 + 0.5, 0.0));

            vec3 V = normalize(-viewPos);
            float rayPdf = 0.0;
            vec3 rayDirView = restir_initialSample_generateRayDir(texelPos, gData.geomNormal, V, material, rayPdf);
            candidate = restir_initialCandidate_makeInvalid(rayDirView);

            if (rayPdf > 0.0) {
                candidate = restir_initialCandidate_makeVoxelFallback(rayDirView, rayPdf);
                //                SSTRay sstRay = sstray_setup(texelPos, viewPos, rayDirView);
                //                sst_trace(sstRay, GI_INITIAL_HIZ_STEPS);
                //
                //                if (sstRay.currT < 0.0) {
                //                    candidate = restir_initialCandidate_makeVoxelFallback(rayDirView, rayPdf);
                //                } else {
                //                    float hitDistance = restir_initialSample_handleRayResult(sstRay);
                //                    if (hitDistance > 0.0) {
                //                        if (!restir_initialSample_screenHitQuery(texelPos, gData.geomNormal, viewPos, rayDirView, rayPdf, hitDistance, candidate)) {
                //                            candidate = restir_initialCandidate_makeVoxelFallback(rayDirView, rayPdf);
                //                        }
                //                    } else {
                //                        candidate = restir_initialCandidate_makeVoxelFallback(rayDirView, rayPdf);
                //                    }
                //                }
            }
        } else {
            transient_geomViewNormal_store(texelPos, vec4(0.0));
            transient_viewNormal_store(texelPos, vec4(0.0));
            transient_restir_resampleMaterial_store(texelPos, vec4(0.0));
            transient_solidAlbedo_store(texelPos, vec4(0.0));
        }

        restir_initialCandidate_store(texelPos, candidate);
    }
}
