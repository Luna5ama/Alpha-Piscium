#define GLOBAL_DATA_MODIFIER buffer

#include "/util/Coords.glsl"
#include "/util/Math.glsl"

#define SPD_CHANNELS 2
#include "/techniques/ffx/spd/SPD.comp.glsl"

layout(rg32f) uniform coherent image2D uimg_hiz;
const vec2 workGroupsRender = vec2(0.25, 0.25);

vec4 spd_reduce4(vec4 v0, vec4 v1, vec4 v2, vec4 v3) {
    vec4 result = vec4(0.0);
    result.x = max4(v0.x, v1.x, v2.x, v3.x);
    result.y = min4(v0.y, v1.y, v2.y, v3.y);
    return result;
}
vec2 spd_loadInput(ivec2 texelPos, uint slice) {
    float viewZ = texelFetch(usam_gbufferViewZ, clamp(texelPos, ivec2(0), uval_mainImageSizeI - 1), 0).r;
    float revZ = coords_viewZToReversedZ(viewZ, near);
    if (all(lessThan(texelPos, uval_mainImageSizeI))){
        imageStore(uimg_hiz, texelPos, vec4(revZ));
    }
    return vec2(revZ);
}
vec2 spd_loadOutput(ivec2 texelPos, uint level, uint slice) {
    ivec4 mipTile = global_hizTiles[level];
    ivec2 readPos = mipTile.xy + clamp(texelPos, ivec2(0), mipTile.zw - 1);
    return imageLoad(uimg_hiz, readPos).rg;
}
void spd_storeOutput(ivec2 texelPos, uint level, uint slice, vec2 value) {
    ivec4 mipTile = global_hizTiles[level];
    ivec2 storePos = mipTile.xy + texelPos;
    if (all(lessThan(texelPos, mipTile.zw))) {
        imageStore(uimg_hiz, storePos, vec4(value, 0.0, 0.0));
    }
}
uint spd_mipCount() {
    return min(findMSB(min(uval_mainImageSizeI.x, uval_mainImageSizeI.y)), 12u);
}
void spd_init() {
    // Do nothing
}