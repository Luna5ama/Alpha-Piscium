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

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        bool frameCond = bool(frameCounter & 1);
        ReSTIRReservoir centerReservoir = restir_reservoir_unpack(
            frameCond ? history_restir_reservoirTemporal1_fetch(texelPos) : history_restir_reservoirTemporal2_fetch(texelPos)
        );
        vec2 centerScreenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        float centerViewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).x;
        vec3 centerViewPos = coords_toViewCoord(centerScreenPos, centerViewZ, global_camProjInverse);
        vec3 centerHitViewPos = centerViewPos + centerReservoir.Y.xyz * centerReservoir.Y.w;

        float dupCount = 0.0;
        const int radius = 8;
        for(int y = -radius; y <= radius; y++) {
            for(int x = -radius; x <= radius; x++) {
                if(x == 0 && y == 0) continue;
                ivec2 neighborPos = texelPos + ivec2(x, y);
                if (all(greaterThanEqual(neighborPos, ivec2(0))) && all(lessThan(neighborPos, uval_mainImageSizeI))) {
                    ReSTIRReservoir neighborReservoir = restir_reservoir_unpack(
                        frameCond ? history_restir_reservoirTemporal1_fetch(neighborPos) : history_restir_reservoirTemporal2_fetch(neighborPos)
                    );
                    vec2 neighborScreenPos = coords_texelToUV(neighborPos, uval_mainImageSizeRcp);
                    float neighborViewZ = texelFetch(usam_gbufferSolidViewZ, neighborPos, 0).x;
                    vec3 neighborViewPos = coords_toViewCoord(neighborScreenPos, neighborViewZ, global_camProjInverse);
                    vec3 neighborHitViewPos = neighborViewPos + neighborReservoir.Y.xyz * neighborReservoir.Y.w;

                    vec3 diff = centerHitViewPos - neighborHitViewPos;
                    float d = dot(diff, diff);
                    float a = pow2(0.00001);
                    float score = a * rcp(a + d);

                    dupCount += score;
                }
            }
        }

        float duplicationScore = dupCount / 288.0;
        duplicationScore = saturate(duplicationScore);

        uvec4 packedReservoir;
        if (frameCond) {
            packedReservoir = history_restir_reservoirTemporal1_fetch(texelPos);
        } else {
            packedReservoir = history_restir_reservoirTemporal2_fetch(texelPos);
        }
        ReSTIRReservoir reservoir = restir_reservoir_unpack(packedReservoir);
        const float cCapDefault = float(SETTING_GI_TEMPORAL_REUSE_LIMIT);
        float cCapMin = 1.0;
        float alpha = 0.1;
        float duplicationScorePower = pow(duplicationScore, alpha);
        float expectedCCap = mix(cCapDefault, cCapMin, duplicationScorePower);
        #if SETTING_DEBUG_OUTPUT
        imageStore(uimg_temp1, texelPos, vec4(duplicationScorePower));
        #endif
        reservoir.m = min(reservoir.m, expectedCCap);
        packedReservoir = restir_reservoir_pack(reservoir);
        if (frameCond) {
            history_restir_reservoirTemporal1_store(texelPos, packedReservoir);
        } else {
            history_restir_reservoirTemporal2_store(texelPos, packedReservoir);
        }
    }
}

