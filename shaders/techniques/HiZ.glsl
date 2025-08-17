#include "/util/Coords.glsl"
#include "/util/Math.glsl"

float hiz_closest_load(ivec2 texelPos, int level) {
    ivec4 mipTile = global_mipmapTiles[0][level];
    ivec2 readPos = mipTile.xy + clamp(texelPos, ivec2(0), mipTile.zw - 1);
    return texelFetch(usam_hiz, readPos, 0).r;
}

float hiz_closest_sample(vec2 texelPos, int level) {
    vec4 mipTile = vec4(global_mipmapTiles[0][level]);
    vec2 readPos = mipTile.xy + clamp(texelPos, vec2(0.5), mipTile.zw - 0.5);
    return texture(usam_hiz, readPos / vec2(textureSize(usam_hiz, 0))).r;
}

vec4 hiz_closest_gather(vec2 texelPos, int level) {
    vec4 mipTile = vec4(global_mipmapTiles[0][level]);
    vec2 readPos = mipTile.xy + clamp(texelPos, vec2(1.0), mipTile.zw - 1.0);
    return textureGather(usam_hiz, readPos / vec2(textureSize(usam_hiz, 0)), 0);
}

bool hiz_groupGroundCheck(uvec2 groupOrigin, int level) {
    return hiz_closest_load(ivec2(groupOrigin), level) > coords_viewZToReversedZ(-65536.0, near);
}

bool hiz_groupGroundCheck4x4(vec2 groupOrigin, int level) {
    vec4 hizValues = hiz_closest_gather(groupOrigin + vec2(-1.0, -1.0), level);
    hizValues = max(hizValues, hiz_closest_gather(groupOrigin + vec2(1.0, -1.0), level));
    hizValues = max(hizValues, hiz_closest_gather(groupOrigin + vec2(-1.0, 1.0), level));
    hizValues = max(hizValues, hiz_closest_gather(groupOrigin + vec2(1.0, 1.0), level));
    return max4(hizValues) > coords_viewZToReversedZ(-65536.0, near);
}

float hiz_furthest_load(ivec2 texelPos, int level) {
    ivec4 mipTile = global_mipmapTiles[1][level];
    ivec2 readPos = mipTile.xy + clamp(texelPos, ivec2(0), mipTile.zw - 1);
    return texelFetch(usam_hiz, readPos, 0).g;
}

float hiz_furthest_sample(vec2 texelPos, int level) {
    vec4 mipTile = vec4(global_mipmapTiles[1][level]);
    vec2 readPos = mipTile.xy + clamp(texelPos, vec2(0.5), mipTile.zw - 0.5);
    return texture(usam_hiz, readPos / vec2(textureSize(usam_hiz, 0))).g;
}

vec4 hiz_furthest_gather(vec2 texelPos, int level) {
    vec4 mipTile = vec4(global_mipmapTiles[1][level]);
    vec2 readPos = mipTile.xy + clamp(texelPos, vec2(1.0), mipTile.zw - 1.0);
    return textureGather(usam_hiz, readPos / vec2(textureSize(usam_hiz, 0)), 0);
}

bool hiz_groupSkyCheck(uvec2 groupOrigin, int level) {
    return hiz_furthest_load(ivec2(groupOrigin), level) <= coords_viewZToReversedZ(-65536.0, near);
}

bool hiz_groupSkyCheck4x4(vec2 groupOrigin, int level) {
    vec4 hizValues = hiz_furthest_gather(groupOrigin + vec2(-1.0, -1.0), level);
    hizValues = min(hizValues, hiz_furthest_gather(groupOrigin + vec2(1.0, -1.0), level));
    hizValues = min(hizValues, hiz_furthest_gather(groupOrigin + vec2(-1.0, 1.0), level));
    hizValues = min(hizValues, hiz_furthest_gather(groupOrigin + vec2(1.0, 1.0), level));
    return min4(hizValues) <= coords_viewZToReversedZ(-65536.0, near);
}