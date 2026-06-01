#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(rgba8) uniform restrict writeonly image2D uimg_rgba8;

#include "/techniques/gi/InitialSample.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Morton.glsl"
#include "/util/ThreadGroupTiling.glsl"

void main() {
    voxel_initShared();

    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    ivec2 texelPos = ivec2(workGroupOrigin + mortonPos);

    if (!all(lessThan(texelPos, uval_mainImageSizeI))) {
        return;
    }

    restir_InitialCandidate candidate = restir_initialCandidate_load(texelPos);
    if (!restir_initialCandidate_needsVoxelFallback(candidate)) {
        return;
    }

    float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).x;
    if (viewZ <= -65536.0 || candidate.pdf <= 0.0 || any(isnan(candidate.rayDirView))) {
        restir_initialCandidate_store(texelPos, restir_initialCandidate_makeInvalid(candidate.rayDirView));
        return;
    }

    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
    vec3 rayOriginView = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec3 rayOriginWorld = coords_pos_viewToWorld(rayOriginView, gbufferModelViewInverse) + cameraPosition;
    vec3 rayWorldDir = coords_dir_viewToWorld(candidate.rayDirView);

    VoxelRay voxelRay = voxelray_setup(rayOriginWorld + rayWorldDir * 0.01, rayWorldDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 128);
    candidate = restir_initialSample_buildVoxelCandidate(texelPos, rayOriginView, candidate.rayDirView, candidate.pdf, hit);
    restir_initialCandidate_store(texelPos, candidate);
}
