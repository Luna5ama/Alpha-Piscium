#define GLOBAL_DATA_MODIFIER buffer

#include "/util/GBufferData.glsl"
#include "/util/Math.glsl"

#define SPD_CHANNELS 4
#include "/techniques/ffx/spd/SPD.comp.glsl"

layout(rgba32f) uniform coherent image2D uimg_hiz;
const vec2 workGroupsRender = vec2(0.25, 0.25);

vec4 spd_reduce4(vec4 v0, vec4 v1, vec4 v2, vec4 v3) {
    vec4 result = vec4(0.0);
    result.x = max4(v0.x, v1.x, v2.x, v3.x);
    result.y = min4(v0.y, v1.y, v2.y, v3.y);
    result.z = max4(v0.z, v1.z, v2.z, v3.z);
    result.w = min4(v0.w, v1.w, v2.w, v3.w);
    return result;
}
vec4 spd_loadInput(ivec2 texelPos, uint slice) {
    ivec2 pixel = clamp(texelPos, ivec2(0), uval_mainImageSizeI - 1);
    float viewZ = texelFetch(usam_gbufferSolidViewZ, pixel, 0).r;
    float revZ = coords_viewZToReversedZ(viewZ, near);
    vec4 bounds = vec4(revZ);
    if (viewZ > -65536.0) {
        GBufferData data = gbufferData_init();
        gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, pixel, 0), data);
        vec2 uv = coords_texelToUV(pixel, uval_mainImageSizeRcp) - uval_taaJitterUV;
        vec3 viewRay = coords_toViewCoord(uv, -1.0, global_camProjInverse);
        vec2 halfPixel = abs(vec2(global_camProjInverse[0][0], global_camProjInverse[1][1])) * uval_mainImageSizeRcp;
        // On a plane, reciprocal depth is affine in raster coordinates. Bound its
        // variation to all four pixel corners, retaining the leaf's 2% depth gate.
        float slope = dot(abs(data.geomNormal.xy), halfPixel);
        float incidence = abs(dot(data.geomNormal, viewRay));
        float radius = min(0.02, (slope + 2e-6) / max(incidence, 1e-20));
        bounds.zw = revZ * vec2(1.0 + radius, 1.0 - radius);
    }
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        imageStore(uimg_hiz, texelPos, bounds);
    }
    return bounds;
}
vec4 spd_loadOutput(ivec2 texelPos, uint level, uint slice) {
    ivec4 mipTile = global_hizTiles[level];
    ivec2 readPos = mipTile.xy + clamp(texelPos, ivec2(0), mipTile.zw - 1);
    return imageLoad(uimg_hiz, readPos);
}
void spd_storeOutput(ivec2 texelPos, uint level, uint slice, vec4 value) {
    ivec4 mipTile = global_hizTiles[level];
    ivec2 storePos = mipTile.xy + texelPos;
    if (all(lessThan(texelPos, mipTile.zw))) {
        imageStore(uimg_hiz, storePos, value);
        #ifdef VOXY
        if (global_giWideState == 1 && level > 0u) {
            ivec2 widePos = storePos + ivec2(global_hizTiles[1].z, 0);
            if (widePos.x < uval_mainImageSizeI.x) imageStore(uimg_hiz, widePos, vec4(value.ba, 0.0, 0.0));
        }
        #endif
    }
}
uint spd_mipCount() {
    return min(findMSB(min(uval_mainImageSizeI.x, uval_mainImageSizeI.y)), 12u);
}
void spd_init() {
    // Do nothing
}