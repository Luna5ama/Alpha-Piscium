#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/util/Math.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1, 1, 1);

shared uint shared_avgLumBinCountSum[16];
#ifdef SETTING_DEBUG_AE
shared uint shared_lumBinCountSum[16];
shared uint shared_maxBinCount[16];
#endif

// https://www.desmos.com/calculator/zsvpbvwdhl
float logQuadSoftLimit(float x, float k) {
    return sign(x) * log2(1.0 + abs(x) + pow2(x)) * k;
}

float sqrtTanhSoftLimit(float x, float k) {
    return x * tanh(k * sqrt(abs(x)));
}

void main() {
    if (gl_LocalInvocationIndex < 16) {
        shared_avgLumBinCountSum[gl_LocalInvocationIndex] = 0u;
        #ifdef SETTING_DEBUG_AE
        shared_lumBinCountSum[gl_LocalInvocationIndex] = 0u;
        shared_maxBinCount[gl_LocalInvocationIndex] = 0u;
        #endif
    }
    barrier();

    {
        uint avgLumBinCount = global_aeData.avgLumHistogram[gl_LocalInvocationID.x];
        uint avgLumBinCountWeighted = avgLumBinCount * gl_LocalInvocationID.x;
        uint avgLumBinCountWeightedSum = subgroupAdd(avgLumBinCountWeighted);

        #ifdef SETTING_DEBUG_AE
        uint lumBinCount = global_aeData.lumHistogram[gl_LocalInvocationID.x];
        uint lumBinCountWeighted = lumBinCount * gl_LocalInvocationID.x;
        uint lumBinCountWeightedSum = subgroupAdd(lumBinCountWeighted);
        uint lumBinCountV = gl_LocalInvocationID.x > 0 && gl_LocalInvocationID.x < 255 ? lumBinCount : 0u;
        uint lumBinCountMaxV = subgroupMax(lumBinCountV);
        #endif

        if (subgroupElect()) {
            shared_avgLumBinCountSum[gl_SubgroupID] = avgLumBinCountWeightedSum;
            #ifdef SETTING_DEBUG_AE
            shared_lumBinCountSum[gl_SubgroupID] = lumBinCountWeightedSum;
            shared_maxBinCount[gl_SubgroupID] = lumBinCountMaxV;
            #endif
        }
    }
    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        uint avgLumPartialSum = shared_avgLumBinCountSum[gl_SubgroupInvocationID];
        uint avgLumBinCountSum = subgroupAdd(avgLumPartialSum);
        #ifdef SETTING_DEBUG_AE
        uint lumPartialSum = shared_lumBinCountSum[gl_SubgroupInvocationID];
        uint lumSum = subgroupAdd(lumPartialSum);
        uint lumParialMax = shared_maxBinCount[gl_SubgroupInvocationID];
        uint lumMax = subgroupMax(lumParialMax);
        #endif
        if (subgroupElect()) {
            shared_avgLumBinCountSum[0] = avgLumBinCountSum;
            #ifdef SETTING_DEBUG_AE
            uint totalPixel = global_mainImageSizeI.x * global_mainImageSizeI.y;
            global_aeData.finalAvgLum = float(lumSum) / float(totalPixel);
            global_aeData.lumHistogramMaxBinCount = lumMax;
            #endif
        }
    }
    barrier();

    global_aeData.avgLumHistogram[gl_LocalInvocationIndex] = 0u;

    if (gl_LocalInvocationID.x == 0) {
        float shadowCount = float(max(global_aeData.shadowCount, 1));
        float highlightCount = float(max(global_aeData.highlightCount, 1));
        float totalWeight = float(max(global_aeData.weightSum, 1));

        global_aeData.shadowCount = 0u;
        global_aeData.highlightCount = 0u;
        global_aeData.weightSum = 0u;

        uint histogramCounting = shared_avgLumBinCountSum[0];

        float averageBinIndex = (float(histogramCounting) / totalWeight) + 0.5;
        float averageLuminance = averageBinIndex / 256.0;

        #ifdef SETTING_EXPOSURE_MANUAL
        global_aeData.expValues = vec3(SETTING_EXPOSURE_MANUAL_EV_COARSE + SETTING_EXPOSURE_MANUAL_EV_FINE);
        #else
        vec3 expLast = global_aeData.expValues;
        vec3 expNew;

        const float MIN_EXP = SETTING_EXPOSURE_MIN_EV;
        const float MAX_EXP = SETTING_EXPOSURE_MAX_EV;
        const float FRAME_TIME_60FPS_SECS = 1.0 / 60.0;

        const float MIN_LUM_TARGET = SETTING_EXPOSURE_AVG_LUM_MIN_TARGET / 255.0;
        const float MAX_LUM_TARGET = SETTING_EXPOSURE_AVG_LUM_MAX_TARGET / 255.0;
        float expCurveValue = pow(linearStep(MIN_EXP, MAX_EXP, expLast.z), SETTING_EXPOSURE_AVG_LUM_TARGET_CURVE);
        float lumTarget = mix(MAX_LUM_TARGET, MIN_LUM_TARGET, expCurveValue);
        expNew.x = log2(lumTarget / averageLuminance);
        expNew.x = logQuadSoftLimit(expNew.x, 1.0);

        // Keep top SETTING_EXPOSURE_TOP_PERCENT % of pixels in the top bin
        vec2 hsPercents = vec2(SETTING_EXPOSURE_H_PERCENT, SETTING_EXPOSURE_S_PERCENT) * (totalWeight * 0.01);
        // x: shadow, y: highlight
        vec2 hsExps = log2(vec2(shadowCount, hsPercents.x) / vec2(hsPercents.y, highlightCount));
        expNew.y = expNew.x;
        expNew.y = max(expNew.y, hsExps.x);
        expNew.y = min(expNew.y, hsExps.y) * 0.5 + expNew.y * 0.5;
        expNew.y = sqrtTanhSoftLimit(expNew.y, (SETTING_EXPOSURE_HS_TIME + 1.0) * 0.05);

        expNew.xy = expNew.xy + expLast.xy;
        vec2 timeFactor = exp2(-vec2(SETTING_EXPOSURE_AVG_LUM_TIME, SETTING_EXPOSURE_HS_TIME) + log2(max(frameTime / FRAME_TIME_60FPS_SECS, 1.0)));
        expNew.x = mix(expLast.x, expNew.x, timeFactor.x);
        expNew.x = clamp(expNew.x, MIN_EXP, MAX_EXP);
        expNew.y = clamp(expNew.y, expNew.x - 2.0, expNew.x + 2.0);
        expNew.y = mix(expLast.y, expNew.y, timeFactor.y);
        expNew.y = clamp(expNew.y, MIN_EXP - 1.0, MAX_EXP + 1.0);

        const float AE_TOTAL_WEIGHT = SETTING_EXPOSURE_AVG_LUM_MIX + SETTING_EXPOSURE_HS_MIX;
        expNew.z = expNew.x * SETTING_EXPOSURE_AVG_LUM_MIX;
        expNew.z += expNew.y * SETTING_EXPOSURE_HS_MIX;
        expNew.z /= AE_TOTAL_WEIGHT;

        global_aeData.expValues = expNew;
        #endif
    }
}