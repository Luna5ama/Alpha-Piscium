#extension GL_KHR_shader_subgroup_ballot : enable

#define GI_DENOISE_PASS 1
#define GI_DENOISE_SAMPLES 8
// X: history length radius scale
// Y: variance heuristic radius scale
// Z: min radius
// W: max radius
#define GI_DENOISE_BLUR_RADIUS vec4(32.0, 16.0, 8.0, 32.0)
#define GI_DENOISE_RAND_NOISE_OFFSET ivec2(0, 0)
#include "/techniques/gi/DenoiseBlur.glsl"