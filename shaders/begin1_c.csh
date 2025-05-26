#version 460 compatibility

#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/_Base.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1, 1, 1);

shared uint shared_histogram[256];

void main() {
    shared_histogram[gl_LocalInvocationID.x] = 0u;
    barrier();

    float topBin = float(max(global_lumHistogram[256], 1));
    uint binCount = global_lumHistogram[gl_LocalInvocationID.x];
    {
        uint binCountWeighted = binCount * gl_LocalInvocationID.x;
        uint subgroupSum = subgroupAdd(binCountWeighted);
        if (subgroupElect()) {
            shared_histogram[gl_SubgroupID] = subgroupSum;
        }
    }
    barrier();

    if (gl_LocalInvocationID.x < gl_NumSubgroups) {
        uint partialSum = shared_histogram[gl_LocalInvocationID.x];
        uint subgroupSum = subgroupAdd(partialSum);
        if (subgroupElect()) {
            shared_histogram[0] = subgroupSum;
        }
    }
    barrier();

    global_lumHistogram[gl_LocalInvocationIndex] = 0u;

    if (gl_LocalInvocationID.x == 0) {
        uint histogramCounting = shared_histogram[0];
        float totalPixel = float(global_lumHistogram[257]);

        float averageBinIndex = float(histogramCounting) / max(totalPixel - global_lumHistogram[0], 1.0);
        float averageLuminance = exp2(averageBinIndex / 255.0) - 1.0;

        #ifdef SETTING_EXPOSURE_MANUAL
        global_exposure = vec4(0.0, 0.0, 0.0, exp2(SETTING_EXPOSURE_MANUAL_VALUE));
        #else
        vec4 expLast = global_exposure;
        vec4 expNew;

        // Keep the average luminance at SETTING_EXPOSURE_AVG_LUM_TARGET
        const float MAX_DELTA_AVG_LUM = 1.5;
        expNew.x = (SETTING_EXPOSURE_AVG_LUM_TARGET / averageLuminance);
        expNew.x = clamp(expNew.x, 1.0 / MAX_DELTA_AVG_LUM, MAX_DELTA_AVG_LUM);

        // Keep top SETTING_EXPOSURE_TOP_PERCENT % of pixels in the top bin
        const float MAX_DELTA_TOP_BIN = 1.1;
        float top5Percent = totalPixel * SETTING_EXPOSURE_TOP_BIN_PERCENT * 0.01;
        expNew.y = (top5Percent / topBin);
        expNew.y = clamp(expNew.y, 1.0 / MAX_DELTA_TOP_BIN, MAX_DELTA_TOP_BIN);

        expNew.xy = expNew.xy * expLast.xy;
        expNew.xy = mix(expLast.xy, expNew.xy, vec2(exp2(-SETTING_EXPOSURE_AVG_LUM_TIME), exp2(-SETTING_EXPOSURE_TOP_BIN_TIME)));
        expNew.xy = clamp(expNew.xy, exp2(SETTING_EXPOSURE_MIN_EXP), exp2(SETTING_EXPOSURE_MAX_EXP));

        float totalWeight = SETTING_EXPOSURE_TOP_BIN_MIX + SETTING_EXPOSURE_AVG_LUM_MIX;
        expNew.w = expNew.x * SETTING_EXPOSURE_AVG_LUM_MIX;
        expNew.w += expNew.y * SETTING_EXPOSURE_TOP_BIN_MIX;
        expNew.w /= totalWeight;

        expNew.z = averageLuminance;// Debug
        global_exposure = expNew;
        #endif

        global_lumHistogram[256] = 0u;
        global_lumHistogram[257] = 0u;
    }

}