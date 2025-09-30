#define GLOBAL_DATA_MODIFIER \

#include "/util/Coords.glsl"

#define SPD_CHANNELS 1
#define SPD_OP 1
#include "/techniques/ffx/spd/SPD.comp.glsl"

layout(r32f) uniform coherent image2D uimg_hiz;
const vec2 workGroupsRender = vec2(0.25, 0.25);

float spd_loadInput(ivec2 texelPos) {
    float viewZ = texelFetch(usam_gbufferViewZ, clamp(texelPos, ivec2(0), global_mainImageSizeI - 1), 0).r;
    float revZ = coords_viewZToReversedZ(viewZ, near);
    ivec2 offsetTexelPos = texelPos + 1;
    if (all(lessThanEqual(offsetTexelPos, global_mainImageSizeI))) {
        if (offsetTexelPos.x == global_mainImageSizeI.x) {
            imageStore(uimg_hiz, texelPos + ivec2(1, 0), vec4(revZ));
        }
        if (offsetTexelPos.y == global_mainImageSizeI.y) {
            imageStore(uimg_hiz, texelPos + ivec2(0, 1), vec4(revZ));
        }
        if (all(equal(offsetTexelPos, global_mainImageSizeI))) {
            imageStore(uimg_hiz, texelPos + ivec2(1, 1), vec4(revZ));
        }
    }
    if (all(lessThan(texelPos, global_mainImageSizeI))){
        imageStore(uimg_hiz, texelPos, vec4(revZ));
    }
    return revZ;
}
float spd_loadOutput(ivec2 texelPos, uint level) {
    ivec4 mipTile = global_mipmapTiles[0][level];
    ivec2 readPos = mipTile.xy + clamp(texelPos, ivec2(0), mipTile.zw - 1);
    return imageLoad(uimg_hiz, readPos).r;
}
void spd_storeOutput(ivec2 texelPos, uint level, float value) {
    ivec4 mipTile = global_mipmapTiles[0][level];
    ivec2 storePos = mipTile.xy + texelPos;
    ivec2 offsetTexelPos = texelPos + 1;
    if (all(lessThanEqual(offsetTexelPos, mipTile.zw))) {
        if (offsetTexelPos.x == mipTile.z) {
            imageStore(uimg_hiz, storePos + ivec2(1, 0), vec4(value));
        }
        if (offsetTexelPos.y == mipTile.w) {
            imageStore(uimg_hiz, storePos + ivec2(0, 1), vec4(value));
        }
        if (all(equal(offsetTexelPos, mipTile.zw))) {
            imageStore(uimg_hiz, storePos + ivec2(1, 1), vec4(value));
        }
    }
    if (all(lessThanEqual(texelPos, mipTile.zw))) {
        imageStore(uimg_hiz, storePos, vec4(value));
    }
}
uint spd_mipCount() {
    return min(findMSB(min(global_mainImageSizeI.x, global_mainImageSizeI.y)), 12u);
}