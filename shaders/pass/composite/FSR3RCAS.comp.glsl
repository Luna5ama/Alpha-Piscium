#define GLOBAL_DATA_MODIFIER restrict buffer
#define FSR3_BIND_RCAS
#include "/techniques/ffx/fsr3upscaler/Integration.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_common.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_rcas.glsl"

layout(local_size_x = 64) in;
const vec2 workGroupsRender = vec2(4.0, 0.0625);

void main() {
    RCAS(FfxUInt32x3(gl_LocalInvocationID), FfxUInt32x3(gl_WorkGroupID), FfxUInt32x3(gl_GlobalInvocationID));
    if (all(equal(gl_GlobalInvocationID, uvec3(0)))) {
        global_fsr3ImageSizes = ivec4(RenderSize(), UpscaleSize());
    }
}
