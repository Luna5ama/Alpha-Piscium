#include "/util/Coords.glsl"
#include "/util/FullScreenComp.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(r32f) uniform writeonly image2D uimg_causticsPhoton;

vec3 sampleViewPos(ivec2 sampleTexelPos) {
    float sampleViewZ = texelFetch(usam_gbufferViewZ, sampleTexelPos, 0).r;
    vec2 sampleScreenPos = coords_texelToUV(sampleTexelPos, uval_mainImageSizeRcp);
    return coords_toViewCoord(sampleScreenPos, sampleViewZ, global_camProjInverse);
}

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec3 centerViewPos = sampleViewPos(texelPos);

        vec4 dViewPosdx = vec4(0.0);
        if (texelPos.x < uval_mainImageSizeI.x - 1) {
            dViewPosdx += vec4(sampleViewPos(texelPos + ivec2(1, 0)) - centerViewPos, 1.0);
        }
        if (texelPos.x > 0) {
            dViewPosdx += vec4(-(sampleViewPos(texelPos + ivec2(-1, 0)) - centerViewPos), 1.0);
        }

        vec4 dViewPosdy = vec4(0.0);
        if (texelPos.y < uval_mainImageSizeI.y - 1) {
            dViewPosdy += vec4(sampleViewPos(texelPos + ivec2(0, 1)) - centerViewPos, 1.0);
        }
        if (texelPos.y > 0) {
            dViewPosdy += vec4(-(sampleViewPos(texelPos + ivec2(0, -1)) - centerViewPos), 1.0);
        }
        dViewPosdx.xyz /= dViewPosdx.w;
        dViewPosdy.xyz /= dViewPosdy.w;

        float pixelSize = length(dViewPosdx.xyz) * length(dViewPosdy.xyz);
        ivec2 writePos = texelPos;
        writePos.y += uval_mainImageSizeI.y;
        imageStore(uimg_causticsPhoton, writePos, vec4(pixelSize));
    }
}