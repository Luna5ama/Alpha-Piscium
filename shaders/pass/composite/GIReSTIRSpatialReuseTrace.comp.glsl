// Traces the visibility rays queued by GIReSTIRPairedSpatialShade and rejects occluded neighbor selections.
#include "/Base.glsl"

layout(local_size_x = 128) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

#include "/util/BitPacking.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/ReservoirSplat.glsl"
#include "/techniques/voxel/VoxelTrace.glsl"

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
    vec4 resultY = uintBitsToFloat(transient_restir_pairwiseMISMetadata_fetch(texelPos));
    vec3 winL_out = resultY.xyz;
    float winHitDist = resultY.w;

    uint packedPrimary = restir_splatFetchCurrentPrimary(texelPos);
    vec3 primaryViewPos;
    if (packedPrimary != 0u) {
        primaryViewPos = restir_splatUnpackPrimary(texelPos, packedPrimary, global_camProjInverse);
    } else {
        float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).x;
        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
        primaryViewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    }
    vec3 geomNormal = nzpacking_unpackNormalOct32(transient_restir_spatialInput_fetch(texelPos).x);

    float normalOffset = min(0.05, winHitDist * 0.25);
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
        transient_ssgiDiffOut_store(texelPos, vec4(0.0));
        transient_ssgiSpecOut_store(texelPos, vec4(0.0));
        #if SETTING_DEBUG_OUTPUT
        imageStore(uimg_temp5, texelPos, vec4(1.0, 0.0, 0.0, 0.0));
        #endif
    }
}
