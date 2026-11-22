#define GLOBAL_DATA_MODIFIER buffer
#include "/util/Math.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1, 1, 1);
shared uint counts[256];

void main() {
    uint i = gl_LocalInvocationIndex;
    counts[i] = global_aeData.histogram[i];
    global_aeData.histogram[i] = 0u;
    #ifdef SETTING_DEBUG_AE
    global_aeData.debugHistogram[i] = counts[i];
    #endif
    barrier();
    if (i == 0u) {
        uint total = 0u;
        uint largest = 0u;
        for (int bin = 0; bin < 256; ++bin) {
            total += counts[bin];
            largest = max(largest, counts[bin]);
        }
        #ifdef SETTING_DEBUG_AE
        global_aeData.maxBinWeight = largest;
        #endif
        if (total > 0u) {
            float low = float(total) * (SETTING_EXPOSURE_S_PERCENT * 0.01);
            float high = float(total) * (1.0 - SETTING_EXPOSURE_H_PERCENT * 0.01);
            uint before = 0u;
            float logSum = 0.0;
            for (int bin = 0; bin < 256; ++bin) {
                uint after = before + counts[bin];
                float overlap = max(0.0, min(float(after), high) - max(float(before), low));
                logSum += overlap * (-24.0 + (float(bin) + 0.5) * (40.0 / 256.0));
                before = after;
            }
            float logY = logSum / (high - low);
            global_aeData.meteredLogY = logY;
            // AP scene units: adapt gradually between night and daylight targets.
            float daylight = pow(smoothstep(-13.0, -2.0, logY), exp2(SETTING_EXPOSURE_AVG_LUM_TARGET_CURVE));
            float targetY = mix(float(SETTING_EXPOSURE_AVG_LUM_MIN_TARGET), float(SETTING_EXPOSURE_AVG_LUM_MAX_TARGET), daylight) / 255.0;
            float target = clamp(log2(targetY) - logY, min(SETTING_EXPOSURE_MIN_EV, SETTING_EXPOSURE_MAX_EV), max(SETTING_EXPOSURE_MIN_EV, SETTING_EXPOSURE_MAX_EV));
            global_aeData.targetEV = target;
            #ifndef SETTING_EXPOSURE_MANUAL
            float blend = 1.0 - exp2(-frameTime * 60.0 / exp2(SETTING_EXPOSURE_AVG_LUM_TIME));
            global_aeData.cameraEV = mix(global_aeData.cameraEV, target, blend);
            #endif
        }
        gbufferPrevModelViewInverse = gbufferModelViewInverse;
        global_prevCameraDelta = uval_cameraDelta;
        global_shadowView = shadowModelView;
        global_shadowViewInverse = shadowModelViewInverse;
    }
}
