#extension GL_KHR_shader_subgroup_ballot : enable

#include "/Base.glsl"

#define GI_DENOISE_PASS 2
#define GI_DENOISE_SAMPLES SETTING_DENOISER_SPATIAL_SAMPLES_POST
// X: history length radius scale
// Y: min radius
// Z: max radius
#ifdef SETTING_VIDEO_RENDER_MODE
#define GI_DENOISE_BLUR_RADIUS vec3(8.0, 2.0, 16.0)
#else
#define GI_DENOISE_BLUR_RADIUS vec3(32.0, 8.0, 64.0)
#endif
#define GI_DENOISE_RAND_NOISE_OFFSET ivec2(5, 7)
#include "/techniques/gi/DenoiseBlur.glsl"
