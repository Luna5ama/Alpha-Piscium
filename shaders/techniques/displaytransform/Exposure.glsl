/*
    References:
        [SOB22] Sobotka, Troy. "AgX". 2022.
            https://sobotka.github.io/AgX
        [WRE23] Wrensch, Benjamin. "Minimal AgX Implementation". IOLITE Development Blog. 2023.
            MIT License. Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch)
            https://iolite-engine.com/blog_posts/minimal_agx_implementation
        [LIN24] linlin, "AgX". 2024.
            MIT License. Copyright (c) 2024 linlin
            https://github.com/bWFuanVzYWth/AgX

        You can find full license texts in /licenses

    Credits:
        - GeforceLegend - Optimized AgX curve function (https://github.com/GeForceLegend)
*/

// All values used to derive this implementation are sourced from Troy’s initial AgX implementation/OCIO config file available here:
//   https://github.com/sobotka/AgX

#include "/util/Colors.glsl"
#include "/util/Rand.glsl"

shared uint shared_avgLumHistogram[256];
shared vec3 shared_sum[16];
#ifdef SETTING_DEBUG_AE
shared uint shared_lumHistogram[256];
#endif

void _displaytransform_exposure_init() {
    shared_avgLumHistogram[gl_LocalInvocationIndex] = 0u;
    #ifdef SETTING_DEBUG_AE
    shared_lumHistogram[gl_LocalInvocationIndex] = 0u;
    #endif
    if (gl_LocalInvocationIndex < 16) {
        shared_sum[gl_LocalInvocationIndex] = vec3(0.0);
    }
    barrier();
}

void _displaytransform_exposure_apply(inout vec4 color) {
    color.rgb *= exp2(global_aeData.expValues.z);
}

const float SHADOW_LUMA_THRESHOLD = SETTING_EXPOSURE_S_LUM / 255.0;
const float HIGHLIGHT_LUMA_THRESHOLD = SETTING_EXPOSURE_H_LUM / 255.0;

void _displaytransform_exposure_update(inout vec4 color) {
    float lumimance = colors2_colorspaces_luma(COLORS2_OUTPUT_COLORSPACE, saturate(color.rgb)); // WTF Photoshop
    uint not0Flag = uint(any(greaterThan(color.rgb, vec3(0.0))));

    float pixelNoise = rand_stbnVec1(ivec2(gl_GlobalInvocationID.xy), frameCounter);
    if (bool(not0Flag)) {
        uint binIndex = clamp(uint(lumimance * 256.0), 0u, 255u);
        atomicAdd(shared_avgLumHistogram[binIndex], uint(color.a + pixelNoise));
    }

    {
        uint highlightFlag = not0Flag;
        highlightFlag &= uint(lumimance >= HIGHLIGHT_LUMA_THRESHOLD);
        uint shadowFlag = not0Flag;
        shadowFlag &= uint(lumimance <= SHADOW_LUMA_THRESHOLD);

        float highlightV = float(highlightFlag) * color.a;
        float shadowV = float(shadowFlag) * color.a;
        float totalV = float(not0Flag) * color.a;

        vec3 sumV = vec3(highlightV, shadowV, totalV);
        vec3 sum = subgroupAdd(sumV);
        if (subgroupElect()) {
            shared_sum[gl_SubgroupID] = sum;
        }
    }

    #ifdef SETTING_DEBUG_AE
    {
        uint binIndex = clamp(uint(lumimance * 256.0), 0u, 255u);
        atomicAdd(shared_lumHistogram[binIndex], 1u);
    }
    #endif

    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        vec3 partialSum = shared_sum[gl_SubgroupInvocationID];
        vec3 sum = subgroupAdd(partialSum);
        if (subgroupElect()) {
            float noise = rand_stbnVec1(ivec2(gl_WorkGroupID.xy), frameCounter);
            atomicAdd(global_aeData.highlightCount, uint(sum.x + noise));
            atomicAdd(global_aeData.shadowCount, uint(sum.y + noise));
            atomicAdd(global_aeData.weightSum, uint(sum.z + noise));
        }
    }

    atomicAdd(global_aeData.avgLumHistogram[gl_LocalInvocationIndex], shared_avgLumHistogram[gl_LocalInvocationIndex]);
    #ifdef SETTING_DEBUG_AE
    atomicAdd(global_aeData.lumHistogram[gl_LocalInvocationIndex], shared_lumHistogram[gl_LocalInvocationIndex]);
    #endif
}