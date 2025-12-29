#extension GL_KHR_shader_subgroup_ballot : enable

#include "/util/Rand.glsl"
#include "/techniques/HiZCheck.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

#define GI_DENOISE_PASS 2
#define GI_DENOISE_SAMPLES 8
#include "/techniques/gi/DenoiseBlur.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        if (hiz_groupGroundCheckSubgroup(gl_WorkGroupID.xy, 4)) {
            // X: history length radius scale
            // Y: variance heuristic radius scale
            // Z: min radius
            // W: max radius
            const vec4 baseKernelRadius = vec4(16.0, 64.0, 2.0, 32.0);
            vec2 noise2 = rand_stbnVec2(texelPos + ivec2(5, 7), frameCounter);
            gi_blur(texelPos, baseKernelRadius, noise2);
        }
    }
}
