#include "/util/Rand.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

#define GI_DENOISE_PASS 1
#define GI_DENOISE_SAMPLES 8
#include "/techniques/gi/DenoiseBlur.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        GIHistoryData historyData = gi_historyData_init();
        gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

        float kernelRadius = 32.0;
        float accumReduction = 1.0 / (1.0 + historyData.realHistoryLength * HISTORY_LENGTH);
        accumReduction = sqrt(accumReduction);
        kernelRadius *= accumReduction;
        vec2 noise2 = rand_stbnVec2(texelPos, frameCounter);
        gi_blur(texelPos, kernelRadius, noise2);
    }
}
