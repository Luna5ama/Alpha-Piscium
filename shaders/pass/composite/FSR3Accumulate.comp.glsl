#define GLOBAL_DATA_MODIFIER restrict buffer
#define FSR3_BIND_ACCUMULATE
#include "/techniques/ffx/fsr3upscaler/Integration.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_common.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_sample.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_upsample.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_reproject.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_accumulate.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

void main() {
    FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (all(lessThan(pos, UpscaleSize()))) Accumulate(pos);
}
