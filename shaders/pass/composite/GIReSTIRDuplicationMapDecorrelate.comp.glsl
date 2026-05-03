/*
    References:
        [LKW26] Lin, Daqi, et al. "ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing".
            Proceedings of the ACM on Computer Graphics and Interactive Techniques. 9, 1, Article 13 (2026).
            https://doi.org/10.1145/3804494

        You can find full license texts in /licenses
*/
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Reservoir.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Rand.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
layout(rgba16f) uniform writeonly image2D uimg_temp1;

shared vec4 sm_hitViewPos[1024];

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    bool frameCond = bool(frameCounter & 1);

    ivec2 basePos = ivec2(workGroupOrigin) - ivec2(8);
    uvec2 localId = gl_LocalInvocationID.xy;

    for(int i = 0; i < 4; i++) {
        ivec2 offset = ivec2(i % 2, i / 2) * 16;
        ivec2 smPos = ivec2(localId) + offset;
        ivec2 loadPos = basePos + smPos;

        vec4 hitViewPos = vec4(0.0);
        if (all(greaterThanEqual(loadPos, ivec2(0))) && all(lessThan(loadPos, uval_mainImageSizeI))) {
            uvec4 packedRes;
            if (frameCond) {
                packedRes = history_restir_reservoirTemporal1_fetch(loadPos);
            } else {
                packedRes = history_restir_reservoirTemporal2_fetch(loadPos);
            }
            ReSTIRReservoir r = restir_reservoir_unpack(packedRes);
            vec2 screenPos = coords_texelToUV(loadPos, uval_mainImageSizeRcp);
            float viewZ = texelFetch(usam_gbufferSolidViewZ, loadPos, 0).x;
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
            hitViewPos = vec4(viewPos + r.Y.xyz * r.Y.w, 1.0);
        }
        sm_hitViewPos[smPos.y * 32 + smPos.x] = hitViewPos;
    }

    barrier();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        uvec4 centerPackedReservoir;
        if (frameCond) {
            centerPackedReservoir = history_restir_reservoirTemporal1_fetch(texelPos);
        } else {
            centerPackedReservoir = history_restir_reservoirTemporal2_fetch(texelPos);
        }
        ReSTIRReservoir centerReservoir = restir_reservoir_unpack(centerPackedReservoir);

        ivec2 centerSmPos = ivec2(mortonPos) + ivec2(8);
        vec3 centerHitViewPos = sm_hitViewPos[centerSmPos.y * 32 + centerSmPos.x].xyz;

        float dupCount = 0.0;
        const int radius = 8;
        for(int y = -radius; y <= radius; y++) {
            for(int x = -radius; x <= radius; x++) {
                if(x == 0 && y == 0) continue;

                ivec2 neighborSmPos = centerSmPos + ivec2(x, y);
                vec4 neighborInfo = sm_hitViewPos[neighborSmPos.y * 32 + neighborSmPos.x];

                if (neighborInfo.w > 0.0) {
                    vec3 diff = centerHitViewPos - neighborInfo.xyz;
                    float d = dot(diff, diff);
                    float a = pow2(0.00001);
                    float score = a * rcp(a + d);

                    dupCount += score;
                }
            }
        }

        float duplicationScore = dupCount / 288.0;
        duplicationScore = saturate(duplicationScore);

        ReSTIRReservoir reservoir = restir_reservoir_unpack(centerPackedReservoir);
        const float cCapDefault = float(SETTING_GI_TEMPORAL_REUSE_LIMIT);
        float cCapMin = 1.0;
        float alpha = 0.1;
        float duplicationScorePower = pow(duplicationScore, alpha);
        float expectedCCap = mix(cCapDefault, cCapMin, duplicationScorePower);
        #if SETTING_DEBUG_OUTPUT
        imageStore(uimg_temp1, texelPos, vec4(duplicationScorePower));
        #endif
        reservoir.m = min(reservoir.m, expectedCCap);
        uvec4 packedReservoir = restir_reservoir_pack(reservoir);
        if (frameCond) {
            history_restir_reservoirTemporal1_store(texelPos, packedReservoir);
        } else {
            history_restir_reservoirTemporal2_store(texelPos, packedReservoir);
        }
    }
}
