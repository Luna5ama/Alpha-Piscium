#define GLOBAL_DATA_MODIFIER restrict buffer
#define FSR3_BIND_PREPARE_INPUTS
#include "/techniques/ffx/fsr3upscaler/Integration.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_common.glsl"
#include "/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_prepare_inputs.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(RENDER_SCALE_FACTOR, RENDER_SCALE_FACTOR);

void main() {
    FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, RenderSize()))) return;

    if (FSR3HistoryReset()) {
        ivec2 outputStart = pos * UpscaleSize() / RenderSize();
        ivec2 outputEnd = (pos + 1) * UpscaleSize() / RenderSize();
        for (int y = outputStart.y; y < outputEnd.y; y++) {
            for (int x = outputStart.x; x < outputEnd.x; x++) {
                imageStore(uimg_fsr3UpscaleAtlas, FSR3OutputTexel(ivec2(x, y)), vec4(0.0));
            }
        }
    }
    PrepareInputs(pos);
}
