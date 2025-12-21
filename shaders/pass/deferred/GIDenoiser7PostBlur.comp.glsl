#include "/util/Rand.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

#define GI_DENOISE_PASS 2
#define GI_DENOISE_SAMPLES 8
#include "/techniques/gi/DenoiseBlur.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        GIHistoryData historyData = gi_historyData_init();
        gi_historyData_unpack1(historyData, transient_gi1Reprojected_fetch(texelPos));
        gi_historyData_unpack2(historyData, transient_gi2Reprojected_fetch(texelPos));
        gi_historyData_unpack3(historyData, transient_gi3Reprojected_fetch(texelPos));
        gi_historyData_unpack4(historyData, transient_gi4Reprojected_fetch(texelPos));
        gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

        // TODO: post blur radius
        const vec3 baseKernelRadius = vec3(4.0, 0.125, 32.0);
//        const vec3 baseKernelRadius = vec3(32.0, 32.0, 32.0);
        vec2 noise2 = rand_stbnVec2(texelPos + ivec2(5, 7), frameCounter);
        gi_blur(texelPos, baseKernelRadius, historyData, noise2);
    }
}
