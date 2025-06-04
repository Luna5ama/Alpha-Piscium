/*
    References:
        [QUI15] Quilez, Inigo. "Texture Repetition". 2015.
            https://iquilezles.org/articles/texturerepetition/
        [QUI17] Quilez, Inigo. "Texture Repetition V". 2017.
            MIT License. Copyright (c) 2017 Inigo Quilez
            https://www.shadertoy.com/view/Xtl3zf

        You can find full license texts in /licenses
*/
#ifndef INCLUDE_util_Sampling_glsl
#define INCLUDE_util_Sampling_glsl a

#include "Hash.glsl"
#include "Math.glsl"

vec4 sampling_bSplineWeights(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return vec4(
        (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
        (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
        (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
        t3 / 6.0
    );
}

vec4 sampling_catmullRomWeights(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return vec4(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2
    );
}

#if DERIVATIVE_AVAILABLE
// [QUI17]
vec3 sampling_textureRepeatGrad(sampler2D t, vec2 uv, float v ) {
    float k = texture(iChannel1, 0.005 * uv).x; // cheap (cache friendly) lookup

    vec2 duvdx = dFdx(uv);
    vec2 duvdy = dFdy(uv);

    float l = k * 8.0;
    float f = fract(l);

    float ia = floor(l);
    float ib = ia + 1.0;

    vec2 offa = sin(vec2(3.0, 7.0) * ia); // can replace with any other hash
    vec2 offb = sin(vec2(3.0, 7.0) * ib); // can replace with any other hash

    vec3 cola = textureGrad(t, uv + v * offa, duvdx, duvdy).xyz;
    vec3 colb = textureGrad(t, uv + v * offb, duvdx, duvdy).xyz;

    return mix(cola, colb, smoothstep(0.2, 0.8, f - 0.1 * sum3(cola - colb)));
}
#endif

// [QUI17]
vec4 sampling_textureRepeat(sampler2D t, vec2 uv, float v) {
    float k = texture(noisetex, 0.005 * uv).x; // cheap (cache friendly) lookup

    float l = k * 8.0;
    float f = fract(l);

    float ia = floor(l);
    float ib = ia + 1.0;

    vec2 offa = sin(vec2(3.0, 7.0) * ia); // can replace with any other hash
    vec2 offb = sin(vec2(3.0, 7.0) * ib); // can replace with any other hash

    vec4 cola = texture(t, uv + v * offa);
    vec4 colb = texture(t, uv + v * offb);

    return mix(cola, colb, smoothstep(0.2, 0.8, f - 0.1 * sum4(cola - colb)));
}


#endif