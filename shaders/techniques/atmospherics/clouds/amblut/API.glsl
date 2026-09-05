#ifndef INCLUDE_clouds_amblut_API_glsl
#define INCLUDE_clouds_amblut_API_glsl a

#include "/util/Coords.glsl"
#include "/util/Math.glsl"

vec2 clouds_amblut_uv(vec3 viewDir, vec2 jitter) {
    vec2 uv = coords_equirectanglarForward(viewDir);
    return fract(uv + (jitter - 0.5) / 32.0);
}

vec4 clouds_amblut_sampleRaw(vec2 uv) {
    vec2 texelPos = clamp(uv * 32.0, vec2(0.5), vec2(31.5));
    return persistent_cloudsAmbLUT_sample(texelPos / vec2(32.0, 33.0));
}

vec3 clouds_amblut_sample(vec2 uv) {
    return clouds_amblut_sampleRaw(uv).rgb;
}

#endif
