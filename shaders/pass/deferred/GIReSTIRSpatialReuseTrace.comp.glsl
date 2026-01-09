#extension GL_KHR_shader_subgroup_ballot : enable

#define RAY_STEPS 124

#include "/techniques/gi/FinishTrace.comp.glsl"
#include "/techniques/gi/InitialSample.glsl"
#include "/techniques/gi/Reservoir.glsl"

layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;
layout(rgba32ui) uniform restrict writeonly uimage2D uimg_rgba32ui;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

void handleRayResult(SSTRay sstRay) {
    ivec2 texelPos = sstRay.pRayOriginTexelPos;
    bool discardSptialReuse = true;
    if (sstRay.currT < -0.99) {
        discardSptialReuse = false;
    }

    if (discardSptialReuse) {
        history_restir_reservoirSpatial_store(texelPos, restir_reservoir_pack(restir_initReservoir()));
        transient_ssgiOut_store(texelPos, vec4(0.0));
        #if SETTING_DEBUG_OUTPUT
        imageStore(uimg_temp5, texelPos, vec4(0.0, 0.0, 1.0, 0.0));
        #endif
    }
}