#include "/util/Coords.glsl"
#include "/util/FullScreenComp.glsl"
#include "/techniques/atmospherics/clouds/ss/Common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(r32f) uniform restrict writeonly image2D uimg_r32f;

vec3 sampleViewPos(ivec2 sampleTexelPos) {
    float sampleViewZ = texelFetch(usam_gbufferSolidViewZ, sampleTexelPos, 0).r;
    vec2 sampleScreenPos = coords_texelToUV(sampleTexelPos, uval_mainImageSizeRcp);
    return coords_toViewCoord(sampleScreenPos, sampleViewZ, global_camProjInverse);
}

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec3 centerViewPos = sampleViewPos(texelPos);

        vec3 dViewPosdx;
        if (texelPos.x > 0 && texelPos.x < uval_mainImageSizeI.x - 1) {
            dViewPosdx = (sampleViewPos(texelPos + ivec2(1, 0)) - sampleViewPos(texelPos + ivec2(-1, 0))) * 0.5;
        } else if (texelPos.x < uval_mainImageSizeI.x - 1) {
            dViewPosdx = sampleViewPos(texelPos + ivec2(1, 0)) - centerViewPos;
        } else {
            dViewPosdx = centerViewPos - sampleViewPos(texelPos + ivec2(-1, 0));
        }

        vec3 dViewPosdy;
        if (texelPos.y > 0 && texelPos.y < uval_mainImageSizeI.y - 1) {
            dViewPosdy = (sampleViewPos(texelPos + ivec2(0, 1)) - sampleViewPos(texelPos + ivec2(0, -1))) * 0.5;
        } else if (texelPos.y < uval_mainImageSizeI.y - 1) {
            dViewPosdy = sampleViewPos(texelPos + ivec2(0, 1)) - centerViewPos;
        } else {
            dViewPosdy = centerViewPos - sampleViewPos(texelPos + ivec2(0, -1));
        }

        float pixelSize = length(dViewPosdx) * length(dViewPosdy);
        transient_screenPixelSize_store(texelPos, vec4(pixelSize));
        transient_caustics_input_store(texelPos, vec4(0.0));
    }
}
