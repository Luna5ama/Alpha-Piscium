#version 460 compatibility

#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "_Util.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1, 1, 1);

layout(rgba16f) restrict uniform image2D uimg_main;

shared uint shared_lumHistogram[gl_NumSubgroups];

void main() {
    if (gl_LocalInvocationID.x < gl_NumSubgroups) {
        shared_lumHistogram[gl_LocalInvocationID.x] = 0u;
    }
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
            global_lumHistogram[0] = subgroupSum;
        }
    }
    barrier();

    global_lumHistogram[gl_LocalInvocationIndex] = 0u;

    if (gl_LocalInvocationID.x == 0) {
        global_lumHistogram[256] = 0u;

        ivec2 mainImgSize = imageSize(uimg_main);
        float totalPixel = mainImgSize.x * mainImgSize.y;

        float averageBinIndex = float(shared_lumHistogram[0]) / max(totalPixel, 1.0);
        float averageLuminance = exp2(averageBinIndex / 255.0 * 16.0) - 1.0;

        vec4 expLast = global_exposure;
        vec4 expNew;

        // Keep top 5% of pixels in the top bin
        float top5Percent = totalPixel * 0.05;
        expNew.x = (top5Percent / topBin) * expLast.x;
        expNew.x = clamp(expNew.x, 0.00001, 1.0);

        // Keep the average luminance at 0.05
        const float targetLuminance = 0.05;
        float targetExposure = targetLuminance / averageLuminance;
        targetExposure = clamp(targetExposure, 0.0, 1.0);
        expNew.y = targetExposure;

        expNew.xyz = mix(expLast.xyz, expNew.xyz, exp2(-SETTING_AE_TIME));

        expNew.w = min(expNew.x, expNew.y);
        expNew.w = mix(expLast.w, expNew.w, exp2(-SETTING_AE_TIME));

        global_exposure = expNew;
    }

}