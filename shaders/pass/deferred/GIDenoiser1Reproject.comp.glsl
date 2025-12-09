#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/Reproject.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_temp1;
layout(rgba16f) uniform restrict writeonly image2D uimg_temp2;
layout(rgba16f) uniform restrict writeonly image2D uimg_temp3;
layout(rgba16f) uniform restrict writeonly image2D uimg_temp4;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        GBufferData gData = gbufferData_init();
        gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
        gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

        gi_reproject(screenPos, viewZ, gData.normal, gData.geomNormal, gData.isHand);
    }
}
