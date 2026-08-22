#extension GL_KHR_shader_subgroup_ballot : enable

#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(r32ui) uniform restrict uimage2D uimg_r32ui;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;

#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/ReservoirSplat.glsl"
#include "/util/ThreadGroupTiling.glsl"

shared mat3 shared_prevViewToCurrView;
shared vec3 shared_prevViewToCurrViewTrans;

ReSTIRReservoir readPreviousReservoir(ivec2 texelPos) {
    uvec4 packedReservoir = history_restir_reservoirTemporal_load(texelPos);
    return restir_reservoir_unpack(packedReservoir);
}

uint readPreviousPrimary(ivec2 texelPos) {
    return bool(frameCounter & 1)
        ? history_restir_primary2_load(texelPos).x
        : history_restir_primary1_load(texelPos).x;
}

void writeSplatNext(ivec2 texelPos, uint nextNode) {
    // Spatial pass 0 initializes this transient after all splat chains are resolved.
    transient_restir_pairwiseMISMetadata_store(texelPos, uvec4(nextNode, 0u, 0u, 0u));
}

uint exchangeSplatHead(ivec2 texelPos, uint node) {
    return bool(frameCounter & 1)
        ? history_restir_primary1_atomicExchange(texelPos, node)
        : history_restir_primary2_atomicExchange(texelPos, node);
}

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    ivec2 texelPos = ivec2(workGroupOrigin + morton_8bDecode(threadIdx));

    if (threadIdx == 0u) {
        shared_prevViewToCurrView = mat3(gbufferModelView) * mat3(gbufferPrevModelViewInverse);
        shared_prevViewToCurrViewTrans = mat3(gbufferModelView) * (gbufferPrevModelViewInverse[3].xyz - uval_cameraDelta) + gbufferModelView[3].xyz;
    }
    barrier();

    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) {
        return;
    }

    ReSTIRReservoir previousReservoir = readPreviousReservoir(texelPos);
    uint packedPrimary = readPreviousPrimary(texelPos);
    if (!restir_isReservoirValid(previousReservoir) || packedPrimary == 0u) {
        return;
    }

    vec3 prevPrimary = restir_splatUnpackPrimary(texelPos, packedPrimary, global_prevCamProjInverse);
    vec3 currPrimary = shared_prevViewToCurrView * prevPrimary + shared_prevViewToCurrViewTrans;
    vec4 currClip = global_camProj * vec4(currPrimary, 1.0);
    if (currClip.z <= 0.0 || any(greaterThanEqual(abs(currClip.xy), currClip.ww))) {
        return;
    }

    vec2 currScreen = currClip.xy / currClip.w * 0.5 + 0.5 + uval_taaJitterUV;
    ivec2 outputTexelPos = ivec2(floor(currScreen * uval_mainImageSize));
    if (any(lessThan(outputTexelPos, ivec2(0))) || any(greaterThanEqual(outputTexelPos, uval_mainImageSizeI))) {
        return;
    }

    float currViewZ = texelFetch(usam_gbufferSolidViewZ, outputTexelPos, 0).x;
    if (currViewZ <= -65536.0) {
        return;
    }

    vec2 centerScreen = coords_texelToUV(outputTexelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
    vec3 centerViewPos = coords_toViewCoord(centerScreen, currViewZ, global_camProjInverse);
    vec3 prevGeomNormal = normalize(history_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
    vec3 currGeomNormal = normalize(shared_prevViewToCurrView * prevGeomNormal);
    vec3 centerGeomNormal = normalize(transient_geomViewNormal_fetch(outputTexelPos).xyz * 2.0 - 1.0);
    if (restir_splatSurfaceConfidence(currPrimary, currGeomNormal, centerViewPos, centerGeomNormal) <= 0.0) {
        return;
    }

    uint node = restir_splatEncodeNode(texelPos);
    uint previousHead = exchangeSplatHead(outputTexelPos, node);
    writeSplatNext(texelPos, previousHead);
}
