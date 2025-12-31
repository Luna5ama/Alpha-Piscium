#extension GL_KHR_shader_subgroup_arithmetic : enable

#define GLOBAL_DATA_MODIFIER buffer

#include "/util/Math.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1, 1, 1);

#ifdef SETTING_DEBUG_AE
shared uint shared_lumBinCountSum[16];
shared uint shared_maxBinCount[16];
#endif

// https://www.desmos.com/calculator/zsvpbvwdhl
float sqrtTanhSoftLimit(float x, float k) {
    return x * tanh(k * sqrt(abs(x)));
}

void main() {
    #ifdef SETTING_DEBUG_AE
    if (gl_LocalInvocationIndex < 16) {
        shared_lumBinCountSum[gl_LocalInvocationIndex] = 0u;
        shared_maxBinCount[gl_LocalInvocationIndex] = 0u;
    }
    barrier();
    {
        uint lumBinCount = global_aeData.lumHistogram[gl_LocalInvocationID.x];
        uint lumBinCountWeighted = lumBinCount * gl_LocalInvocationID.x;
        uint lumBinCountWeightedSum = subgroupAdd(lumBinCountWeighted);
        uint lumBinCountV = gl_LocalInvocationID.x > 0 && gl_LocalInvocationID.x < 255 ? lumBinCount : 0u;
        uint lumBinCountMaxV = subgroupMax(lumBinCountV);

        if (subgroupElect()) {
            shared_lumBinCountSum[gl_SubgroupID] = lumBinCountWeightedSum;
            shared_maxBinCount[gl_SubgroupID] = lumBinCountMaxV;
        }
    }
    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        uint lumPartialSum = shared_lumBinCountSum[gl_SubgroupInvocationID];
        uint lumSum = subgroupAdd(lumPartialSum);
        uint lumParialMax = shared_maxBinCount[gl_SubgroupInvocationID];
        uint lumMax = subgroupMax(lumParialMax);
        if (subgroupElect()) {
            uint totalPixel = uval_mainImageSizeI.x * uval_mainImageSizeI.y;
            global_aeData.finalAvgLum = float(lumSum) / float(totalPixel);
            global_aeData.lumHistogramMaxBinCount = lumMax;
        }
    }
    barrier();
    #endif

    if (gl_LocalInvocationID.x == 0) {
        float shadowCount = float(max(global_aeData.shadowCount, 1));
        float highlightCount = float(max(global_aeData.highlightCount, 1));
        float totalWeight = float(max(global_aeData.weightSum, 1));

        global_aeData.shadowCount = 0u;
        global_aeData.highlightCount = 0u;
        global_aeData.weightSum = 0u;

        float averageLuminance = global_aeData.screenAvgLum.w;

        #ifdef SETTING_EXPOSURE_MANUAL
        global_aeData.expValues = vec4(SETTING_EXPOSURE_MANUAL_EV_COARSE + SETTING_EXPOSURE_MANUAL_EV_FINE);
        #else
        vec4 expLast = global_aeData.expValues;
        vec4 expNew = vec4(0.0);

        const float MIN_EXP = SETTING_EXPOSURE_MIN_EV;
        const float MAX_EXP = SETTING_EXPOSURE_MAX_EV;
        const float FRAME_TIME_60FPS_SECS = 1.0 / 60.0;

        const int INIT_FRAMES = 32;
        float initFadeFactor = linearStep(1.0, float(INIT_FRAMES), float(min(frameCounter, INIT_FRAMES)));

        const float MIN_LUM_TARGET = SETTING_EXPOSURE_AVG_LUM_MIN_TARGET / 255.0;
        const float MAX_LUM_TARGET = SETTING_EXPOSURE_AVG_LUM_MAX_TARGET / 255.0;
        float expCurveValue = pow(pow2(linearStep(MIN_EXP, MAX_EXP, expLast.z)), exp2(SETTING_EXPOSURE_AVG_LUM_TARGET_CURVE));
        float lumTarget = mix(MAX_LUM_TARGET, MIN_LUM_TARGET, expCurveValue);
        expNew.x = log2(lumTarget / averageLuminance);
        float avgDelta = 2.0;
        expNew.x = (1.0 / (1.0 + exp(pow3(-0.5 * expNew.x)))) * avgDelta * 2.0 - avgDelta;

        // Keep top SETTING_EXPOSURE_TOP_PERCENT % of pixels in the top bin
        vec2 hsPercents = vec2(SETTING_EXPOSURE_H_PERCENT, SETTING_EXPOSURE_S_PERCENT) * (totalWeight * 0.01);
        global_aeData.hsPercents = vec2(highlightCount, shadowCount) / totalWeight;

        vec2 timeFactor = exp2(-vec2(SETTING_EXPOSURE_AVG_LUM_TIME, SETTING_EXPOSURE_HS_TIME) + log2(min(frameTime / FRAME_TIME_60FPS_SECS, 1.0)));
        timeFactor = pow(timeFactor, vec2(max(1e-16, initFadeFactor)));

        // x: shadow, y: highlight
        vec2 hsExps = log2(vec2(shadowCount, hsPercents.x) / vec2(hsPercents.y, highlightCount));
        float k = (SETTING_EXPOSURE_HS_TIME + 1.0) * 0.05;
        expNew.w = sqrtTanhSoftLimit(min(hsExps.y, 0.0), k);
        expNew.w = max(expNew.w, sqrtTanhSoftLimit(max(hsExps.x, 0.0), k));
        expNew.w = clamp(expNew.w, SETTING_EXPOSURE_HS_MIN_EV_DELTA, SETTING_EXPOSURE_HS_MAX_EV_DELTA);
        expNew.w = mix(expLast.w, expNew.w, timeFactor.y);

        expNew.x = expNew.x + expLast.x;
        expNew.y = expLast.x + expNew.w;

        expNew.xy = mix(expLast.xy, expNew.xy, timeFactor.xy);
        expNew.x = clamp(expNew.x, MIN_EXP, MAX_EXP);

        float avgLumMix = SETTING_EXPOSURE_AVG_LUM_MIX + 1e-16;
        float hsMix = SETTING_EXPOSURE_HS_MIX * initFadeFactor + 1e-16;
        expNew.z = expNew.x * avgLumMix;
        expNew.z += expNew.y * hsMix;
        expNew.z /= (avgLumMix + hsMix);
        expNew.z = mix(expLast.z, expNew.z, (timeFactor.x + timeFactor.y) / 2.0);

        global_aeData.expValues = expNew;
        #endif
    }
}