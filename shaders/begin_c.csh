#version 460 compatibility

#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "_Util.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1, 1, 1);

layout(rgba16f) restrict uniform image2D uimg_main;

shared uint shared_lumHistogram[256];

void main() {
    shared_lumHistogram[gl_LocalInvocationID.x] = 0u;
    barrier();

    float topBin = float(max(global_lumHistogram[256], 1));
    uint binCount = global_lumHistogram[gl_LocalInvocationID.x];
    {
        uint binCountWeighted = binCount * gl_LocalInvocationID.x;
        uint subgroupSum = subgroupAdd(binCountWeighted);
        if (subgroupElect()) {
            shared_lumHistogram[gl_SubgroupID] = subgroupSum;
        }
    }
    barrier();

    if (gl_LocalInvocationID.x < gl_NumSubgroups) {
        uint partialSum = shared_lumHistogram[gl_LocalInvocationID.x];
        uint subgroupSum = subgroupAdd(partialSum);
        if (subgroupElect()) {
            shared_lumHistogram[0] = subgroupSum;
        }
    }
    barrier();

    global_lumHistogram[gl_LocalInvocationIndex] = 0u;

    if (gl_LocalInvocationID.x == 0) {
        global_lumHistogram[256] = 0u;

        ivec2 mainImgSize = imageSize(uimg_main);
        float totalPixel = mainImgSize.x * mainImgSize.y;

        float averageBinIndex = float(shared_lumHistogram[0]) / max(totalPixel, 1.0);
        float averageLuminance = exp2(averageBinIndex / 255.0) * 16.0 - 16.0;

        vec4 expLast = global_exposure;
        vec4 expNew;

        // Keep top SETTING_AE_TOP_PERCENT% of pixels in the top bin
        float top5Percent = totalPixel * SETTING_AE_TOP_BIN_PERCENT;
        expNew.x = (top5Percent / topBin) * expLast.x;
        expNew.x = clamp(expNew.x, 0.00001, 16.0);

        // Keep the average luminance at SETTING_AE_AVG_LUMA_TARGET
        expNew.y = (SETTING_AE_AVG_LUMA_TARGET / averageLuminance) * expLast.y;
        expNew.y = clamp(expNew.y, 0.00001, 16.0);

        expNew.xy = mix(expLast.xy, expNew.xy, vec2(0.01 * exp2(-SETTING_AE_TOP_BIN_TIME), exp2(-SETTING_AE_AVG_LUMA_TIME)));

        expNew.w = mix(expNew.x, expNew.y, SETTING_AE_AVG_LUMA);

        expNew.b = averageLuminance; // Debug
        global_exposure = expNew;
    }

}