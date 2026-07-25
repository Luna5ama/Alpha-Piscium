#define GLOBAL_DATA_MODIFIER restrict buffer
#define FSR3_BIND_SHADING_CHANGE_PYRAMID
#include "/techniques/ffx/fsr3upscaler/Integration.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_common.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_shading_change_pyramid.glsl"

layout(local_size_x = 256) in;

void main() {
    ComputeShadingChangePyramid(FfxUInt32x3(gl_WorkGroupID), FfxUInt32(gl_LocalInvocationIndex));
}
