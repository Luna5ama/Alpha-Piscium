#extension GL_KHR_shader_subgroup_ballot : enable

#define RAY_STEPS 104

#include "/techniques/gi/FinishTrace.comp.glsl"
#include "/techniques/gi/InitialSample.glsl"

layout(rgba16f) uniform restrict writeonly image2D uimg_temp1;
layout(rgba32ui) uniform restrict writeonly uimage2D uimg_rgba32ui;

void handleRayResult(SSTRay sstRay) {
    ivec2 texelPos = sstRay.pRayOriginTexelPos;

    // If ray still didn't finish, force it to be a miss
    if (sstRay.currT >= 0.0) {
        sstRay.currT = -1.0;
    }

    restir_InitialSampleData sampleData = restir_initialSample_handleRayResult(sstRay);
    transient_restir_initialSample_store(texelPos, restir_initialSampleData_pack(sampleData));
}