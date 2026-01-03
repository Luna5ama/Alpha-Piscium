#extension GL_KHR_shader_subgroup_ballot : enable

#define RAY_STEPS 104

#include "/techniques/gi/FinishTrace.comp.glsl"
#include "/techniques/gi/InitialSample.glsl"

layout(rgba16f) uniform restrict writeonly image2D uimg_temp1;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;

void handleRayResult(SSTRay sstRay) {
    ivec2 texelPos = sstRay.pRayOriginTexelPos;

    // If ray still didn't finish, force it to be a miss
    if (sstRay.currT >= 0.0) {
        sstRay.currT = -1.0;
    }

    float hitDistance = restir_initialSample_handleRayResult(sstRay);
    transient_gi_initialSampleHitDistance_store(texelPos, vec4(hitDistance));
}