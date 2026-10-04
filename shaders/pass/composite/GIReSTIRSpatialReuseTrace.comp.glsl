// Traces the visibility rays queued by GIReSTIRPairedSpatialShade and rejects occluded neighbor selections.
#include "/Base.glsl"

layout(local_size_x = 128) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

#include "/util/BitPacking.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/ReservoirSplat.glsl"
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/techniques/gi/PathGuiding.glsl"

#ifdef PATH_GUIDING_ENABLED
layout(rg32ui) uniform restrict writeonly uimage2D uimg_rg32ui;
#endif

void main() {
    uint rayCount = global_restirVisibilityRayCount;
    uint groupStart = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * gl_WorkGroupSize.x;
    if (groupStart >= rayCount) {
        return;
    }

    voxel_initShared();

    uint queueIndex = groupStart + gl_LocalInvocationIndex;
    if (queueIndex >= rayCount) {
        return;
    }

    ivec2 texelPos = ivec2(unpackUInt2x16(indirectComputeData[queueIndex]));
    uvec4 provisional = transient_restir_pairwiseMISMetadata_fetch(texelPos);
    vec3 winL_out = nzpacking_unpackNormalOct32(provisional.x);
    float winHitDist = uintBitsToFloat(provisional.y);
    // xyz: BRDF-ratio specular, w: ReSTIR specular weight (roughness), both from SpatialShade.
    vec4 specularRatio = unpackHalf4x16(provisional.zw);

    float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).x;
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    uint packedPrimary = restir_splatFetchCurrentPrimary(texelPos);
    vec3 primaryViewPos = packedPrimary != 0u
        ? restir_splatUnpackPrimary(texelPos, packedPrimary, global_camProjInverse)
        : viewPos;
    SpatialSampleData centerSample = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelPos));
    vec3 geomNormal = centerSample.geomNormal;

    float normalOffset = min(0.005, winHitDist * 0.25);
    vec3 rayOriginView = primaryViewPos + geomNormal * normalOffset;
    vec3 expectedHitView = primaryViewPos + winL_out * winHitDist;
    vec3 rayOffsetView = expectedHitView - rayOriginView;
    float expectedHitDistance = length(rayOffsetView);
    vec3 worldPos = coords_pos_viewToWorld(rayOriginView, gbufferModelViewInverse) + vec3(cameraPositionInt) + cameraPositionFract;
    vec3 worldDir = coords_dir_viewToWorld(rayOffsetView * rcp(expectedHitDistance));
    VoxelRay voxelRay = voxelray_setup(worldPos, worldDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 128, true);
    vec3 expectedHitPos = worldPos + worldDir * expectedHitDistance;
    if (!hit.hit || distanceSq(hit.hitPos, expectedHitPos) > 0.05) {
        transient_gi_shadowHint_store(texelPos, vec4(1.0));
        // The occluded neighbor sample contributes nothing: visibility-deferred RIS has zero contribution here.
        // The BRDF-ratio specular has no visibility term and stays, as in SpatialShade's ratio-only output.
        transient_ssgiDiffOut_store(texelPos, vec4(0.0));
        transient_ssgiSpecOut_store(texelPos, vec4(specularRatio.xyz * (1.0 - specularRatio.w), 0.0));
        #ifdef PATH_GUIDING_ENABLED
        // The occluded neighbor sample is not a valid training direction for this pixel.
        transient_pathGuide_trainRecord_store(texelPos, uvec4(0u));
        #endif
        #if SETTING_DEBUG_OUTPUT
        imageStore(uimg_temp5, texelPos, vec4(1.0, 0.0, 0.0, 0.0));
        #endif
    }
}
