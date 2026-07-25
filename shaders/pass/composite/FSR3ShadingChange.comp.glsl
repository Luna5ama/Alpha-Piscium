#define GLOBAL_DATA_MODIFIER restrict buffer
#define FSR3_BIND_SHADING_CHANGE
#include "/techniques/ffx/fsr3upscaler/Integration.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_common.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_shading_change.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(RENDER_SCALE_HALF, RENDER_SCALE_HALF);

void main() {
    FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (all(lessThan(pos, ShadingChangeRenderSize()))) ShadingChange(pos);
}
