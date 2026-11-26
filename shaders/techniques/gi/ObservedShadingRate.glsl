#ifndef INCLUDE_techniques_gi_ObservedShadingRate_glsl
#define INCLUDE_techniques_gi_ObservedShadingRate_glsl a

#include "/util/Coords.glsl"
#include "/util/HardcodedPBR.glsl"
#include "/util/VoxyFaceNormal.glsl"

uint gi_quadPhase(ivec2 base) {
    // Each cell gets all four offsets in four frames without a screen-wide phase.
    uint scramble = uint(base.x >> 1) * 0x85ebca6bu ^ uint(base.y >> 1) * 0xc2b2ae35u;
    return ((scramble >> 19u) + uint(frameCounter)) & 3u;
}

ivec2 gi_quadOffset(uint phase) {
    return ivec2(phase & 1u, phase >> 1u);
}

ivec2 gi_shadingPixel(ivec2 pixel, float z) {
    #if SETTING_GI_USE_REFERENCE != 0
    // The reference integrator averages a canonical sample at each actual pixel.
    return pixel;
    #else
    if (global_giWideState != 3) return pixel;
    ivec2 base = pixel & ~1;
    ivec2 other = base + gi_quadOffset(gi_quadPhase(base));
    if (other == pixel || any(greaterThanEqual(other, uval_mainImageSizeI))) return pixel;
    uvec4 a = texelFetch(usam_gbufferVoxySolidData, pixel, 0);
    if (a.w != floatBitsToUint(z)) return pixel;
    uvec4 b = texelFetch(usam_gbufferVoxySolidData, other, 0);
    float otherZ = texelFetch(usam_gbufferSolidViewZ, other, 0).r;
    uint face = bitfieldExtract(a.y, 24, 3);
    uint material = a.z >> 16u;
    if (b.w != floatBitsToUint(otherZ) || material != b.z >> 16u || face != bitfieldExtract(b.y, 24, 3)) return pixel;
    HardcodedPBR surface = hardcodedpbr_decode(material);
    if (!surface.isFullCube || surface.metalIndex != 0u || surface.roughness < 0.5) return pixel;
    vec2 uv = coords_texelToUV(pixel, uval_mainImageSizeRcp) - 0.5 * uval_taaJitterUV;
    vec2 otherUV = uv + vec2(other - pixel) * uval_mainImageSizeRcp;
    vec3 aView = coords_toViewCoord(uv, z, global_camProjInverse);
    vec3 bView = coords_toViewCoord(otherUV, otherZ, global_camProjInverse);
    vec3 delta = mat3(gbufferModelViewInverse) * (bView - aView);
    vec3 n = voxy_faceWorldNormal(face);
    vec3 origin = cameraPositionFract + (gbufferModelViewInverse * vec4(aView, 1.0)).xyz - n * 0.0625;
    // Distinct layers, blocks, and silhouettes retain their own irradiance sample.
    if (abs(dot(n, delta)) > 0.001 || any(notEqual(floor(origin), floor(origin + delta)))) return pixel;
    return other;
    #endif
}

#endif
