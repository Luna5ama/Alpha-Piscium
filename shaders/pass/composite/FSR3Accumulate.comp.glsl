#define GLOBAL_DATA_MODIFIER restrict buffer
#define FSR3_BIND_ACCUMULATE
#include "/techniques/ffx/fsr3upscaler/Integration.glsl"
// Keep the imported SDK helpers intact and replace their active reconstruction transform locally.
#define Tonemap FSR3VendorTonemap
#define InverseTonemap FSR3VendorInverseTonemap
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_common.glsl"
#undef Tonemap
#undef InverseTonemap
#include "/util/AgxInvertible.glsl"

FfxFloat32x3 Tonemap(FfxFloat32x3 color) {
    return agxInvertible_forwardRange(color, -24.0f, 32.0f);
}

FfxFloat32x3 InverseTonemap(FfxFloat32x3 color) {
    return agxInvertible_inverseRange(color, -24.0f, 32.0f);
}
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_sample.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_upsample.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_reproject.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_accumulate.glsl"

layout(local_size_x = 16, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

void main() {
    FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (all(lessThan(pos, UpscaleSize()))) Accumulate(pos);
}
