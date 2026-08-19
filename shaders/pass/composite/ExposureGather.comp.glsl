#extension GL_KHR_shader_subgroup_arithmetic : enable

#define GLOBAL_DATA_MODIFIER buffer

#include "/techniques/displaytransform/ExposureUpdate.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1, 1, 1);

shared uint shared_lumBinCountSum[16];
shared uint shared_maxBinCount[16];

void main() {
    if (gl_LocalInvocationIndex < 16) {
        shared_lumBinCountSum[gl_LocalInvocationIndex] = 0u;
        shared_maxBinCount[gl_LocalInvocationIndex] = 0u;
    }
    barrier();

    uint lumBinCount = global_aeData.lumHistogram[gl_LocalInvocationID.x];
    uint lumBinCountWeighted = lumBinCount * gl_LocalInvocationID.x;
    uint lumBinCountWeightedSum = subgroupAdd(lumBinCountWeighted);
    uint lumBinCountV = gl_LocalInvocationID.x > 0 && gl_LocalInvocationID.x < 255 ? lumBinCount : 0u;
    uint lumBinCountMaxV = subgroupMax(lumBinCountV);

    if (subgroupElect()) {
        shared_lumBinCountSum[gl_SubgroupID] = lumBinCountWeightedSum;
        shared_maxBinCount[gl_SubgroupID] = lumBinCountMaxV;
    }
    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        uint lumPartialSum = shared_lumBinCountSum[gl_SubgroupInvocationID];
        uint lumSum = subgroupAdd(lumPartialSum);
        uint lumPartialMax = shared_maxBinCount[gl_SubgroupInvocationID];
        uint lumMax = subgroupMax(lumPartialMax);
        if (subgroupElect()) {
            uint totalPixel = uval_mainImageSizeI.x * uval_mainImageSizeI.y;
            global_aeData.finalAvgLum = float(lumSum) / float(totalPixel);
            global_aeData.lumHistogramMaxBinCount = lumMax;
        }
    }
    barrier();

    if (gl_LocalInvocationID.x == 0) {
        exposure_update();
    }

    global_aeData.lumHistogram[gl_LocalInvocationIndex] = 0u;
}
