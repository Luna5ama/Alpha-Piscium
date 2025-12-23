layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba8) uniform writeonly image2D uimg_rgba8;
layout(rgb10_a2) uniform writeonly image2D uimg_rgb10_a2;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
#include "/techniques/gi/Reproject.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float currViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        GBufferData gData = gbufferData_init();
        gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
        gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

        gi_reproject(texelPos, currViewZ, gData);

        transient_geomViewNormal_store(texelPos, vec4(gData.geomNormal * 0.5 + 0.5, 0.0));
        transient_viewNormal_store(texelPos, vec4(gData.normal * 0.5 + 0.5, 0.0));
    }
}
