/*
    References:
        [QUI13a] Quilez, Inigo. "Noise - gradient - 2D". 2013.
            MIT License. Copyright (c) 2013 Inigo Quilez.
            https://www.shadertoy.com/view/XdXGW8
        [QUI13b] Quilez, Inigo. "Noise - gradien  - 3D". 2013.
            MIT License. Copyright (c) 2013 Inigo Quilez.
            https://www.shadertoy.com/view/Xsl3Dl
        [QUI17a] Quilez, Inigo. "Noise - Gradient - 2D - Deriv". 2017.
            MIT License. Copyright (c) 2017 Inigo Quilez.
            https://www.shadertoy.com/view/XdXBRH
        [QUI17b] Quilez, Inigo. "Noise - Gradient - 3D - Deriv". 2017.
            MIT License. Copyright (c) 2017 Inigo Quilez.
            https://www.shadertoy.com/view/4dffRH
        [QUI17c] Quilez, Inigo. "Gradient Noise Derivatives". 2017.
            https://iquilezles.org/articles/gradientnoise/

        You can find full license texts in /licenses
*/

#ifndef INCLUDE_util_noise_GradientNoise_glsl
#define INCLUDE_util_noise_GradientNoise_glsl a

#include "_Common.glsl"
#include "/util/Rand.glsl"

// -------------------------------------------------- Value Noise 2D --------------------------------------------------
vec2 _GradientNoise_2D_hash(uvec2 x) {
    return hash_uintToFloat(hash_22_q2(x)) * 2.0 - 1.0;
}

vec2 _GradientNoise_2D_hash_hf(ivec2 x) {
    return rand_stbnVec2(x, 0u) * 2.0 - 1.0;
}

// [QUI13a]
float GradientNoise_2D_value(vec2 x) {
    uvec2 i = _noise_hash_coord(x);
    vec2 w = fract(x);

    vec2 u = _NOISE_INTERPO(w);

    vec2 ga = _GradientNoise_2D_hash(i + uvec2(0, 0));
    vec2 gb = _GradientNoise_2D_hash(i + uvec2(1, 0));
    vec2 gc = _GradientNoise_2D_hash(i + uvec2(0, 1));
    vec2 gd = _GradientNoise_2D_hash(i + uvec2(1, 1));

    float va = dot(ga, w - vec2(0.0, 0.0));
    float vb = dot(gb, w - vec2(1.0, 0.0));
    float vc = dot(gc, w - vec2(0.0, 1.0));
    float vd = dot(gd, w - vec2(1.0, 1.0));

    float xy0 = mix(va, vb, u.x);
    float xy1 = mix(vc, vd, u.x);
    float value = mix(xy0, xy1, u.y);

    return value;
}

float GradientNoise_2D_value_hf(vec2 x) {
    ivec2 i = _noise_hash_coord_signed(x);
    vec2 w = fract(x);

    vec2 u = _NOISE_INTERPO(w);

    vec2 ga = _GradientNoise_2D_hash_hf(i + ivec2(0, 0));
    vec2 gb = _GradientNoise_2D_hash_hf(i + ivec2(1, 0));
    vec2 gc = _GradientNoise_2D_hash_hf(i + ivec2(0, 1));
    vec2 gd = _GradientNoise_2D_hash_hf(i + ivec2(1, 1));

    float va = dot(ga, w - vec2(0.0, 0.0));
    float vb = dot(gb, w - vec2(1.0, 0.0));
    float vc = dot(gc, w - vec2(0.0, 1.0));
    float vd = dot(gd, w - vec2(1.0, 1.0));

    float xy0 = mix(va, vb, u.x);
    float xy1 = mix(vc, vd, u.x);
    float value = mix(xy0, xy1, u.y);

    return value;
}

// [QUI17a]
vec2 GradientNoise_2D_grad(vec2 x) {
    uvec2 i = _noise_hash_coord(x);
    vec2 w = fract(x);

    vec2 u = _NOISE_INTERPO(w);
    vec2 du = _NOISE_INTERPO_GRAD(w);

    vec2 ga = _GradientNoise_2D_hash(i + uvec2(0, 0));
    vec2 gb = _GradientNoise_2D_hash(i + uvec2(1, 0));
    vec2 gc = _GradientNoise_2D_hash(i + uvec2(0, 1));
    vec2 gd = _GradientNoise_2D_hash(i + uvec2(1, 1));

    float va = dot(ga, w - vec2(0.0, 0.0));
    float vb = dot(gb, w - vec2(1.0, 0.0));
    float vc = dot(gc, w - vec2(0.0, 1.0));
    float vd = dot(gd, w - vec2(1.0, 1.0));

    vec2 g = mix(mix(ga, gb, u.x), mix(gc, gd, u.x), u.y);
    vec2 d = mix(vec2(vb, vc)-va, vd - vec2(vc, vb), u.yx);
    vec2 grad = g + du * d;

    return grad;
}


vec2 GradientNoise_2D_grad_hf(vec2 x) {
    ivec2 i = _noise_hash_coord_signed(x);
    vec2 w = fract(x);

    vec2 u = _NOISE_INTERPO(w);
    vec2 du = _NOISE_INTERPO_GRAD(w);

    vec2 ga = _GradientNoise_2D_hash_hf(i + ivec2(0, 0));
    vec2 gb = _GradientNoise_2D_hash_hf(i + ivec2(1, 0));
    vec2 gc = _GradientNoise_2D_hash_hf(i + ivec2(0, 1));
    vec2 gd = _GradientNoise_2D_hash_hf(i + ivec2(1, 1));

    float va = dot(ga, w - vec2(0.0, 0.0));
    float vb = dot(gb, w - vec2(1.0, 0.0));
    float vc = dot(gc, w - vec2(0.0, 1.0));
    float vd = dot(gd, w - vec2(1.0, 1.0));

    vec2 g = mix(mix(ga, gb, u.x), mix(gc, gd, u.x), u.y);
    vec2 d = mix(vec2(vb, vc)-va, vd - vec2(vc, vb), u.yx);
    vec2 grad = g + du * d;

    return grad;
}

// [QUI17a]
vec3 GradientNoise_2D_valueGrad(vec2 x) {
    uvec2 i = _noise_hash_coord(x);
    vec2 w = fract(x);

    vec2 u = _NOISE_INTERPO(w);
    vec2 du = _NOISE_INTERPO_GRAD(w);

    vec2 ga = _GradientNoise_2D_hash(i + uvec2(0, 0));
    vec2 gb = _GradientNoise_2D_hash(i + uvec2(1, 0));
    vec2 gc = _GradientNoise_2D_hash(i + uvec2(0, 1));
    vec2 gd = _GradientNoise_2D_hash(i + uvec2(1, 1));

    float va = dot(ga, w - vec2(0.0, 0.0));
    float vb = dot(gb, w - vec2(1.0, 0.0));
    float vc = dot(gc, w - vec2(0.0, 1.0));
    float vd = dot(gd, w - vec2(1.0, 1.0));

    float xy0 = mix(va, vb, u.x);
    float xy1 = mix(vc, vd, u.x);
    float value = mix(xy0, xy1, u.y);

    vec2 g = mix(mix(ga, gb, u.x), mix(gc, gd, u.x), u.y);
    vec2 d = mix(vec2(vb, vc)-va, vd - vec2(vc, vb), u.yx);
    vec2 grad = g + du * d;

    return vec3(value, grad);
}


vec3 GradientNoise_2D_valueGrad_hf(vec2 x) {
    ivec2 i = _noise_hash_coord_signed(x);
    vec2 w = fract(x);

    vec2 u = _NOISE_INTERPO(w);
    vec2 du = _NOISE_INTERPO_GRAD(w);

    vec2 ga = _GradientNoise_2D_hash_hf(i + ivec2(0, 0));
    vec2 gb = _GradientNoise_2D_hash_hf(i + ivec2(1, 0));
    vec2 gc = _GradientNoise_2D_hash_hf(i + ivec2(0, 1));
    vec2 gd = _GradientNoise_2D_hash_hf(i + ivec2(1, 1));

    float va = dot(ga, w - vec2(0.0, 0.0));
    float vb = dot(gb, w - vec2(1.0, 0.0));
    float vc = dot(gc, w - vec2(0.0, 1.0));
    float vd = dot(gd, w - vec2(1.0, 1.0));

    float xy0 = mix(va, vb, u.x);
    float xy1 = mix(vc, vd, u.x);
    float value = mix(xy0, xy1, u.y);

    vec2 g = mix(mix(ga, gb, u.x), mix(gc, gd, u.x), u.y);
    vec2 d = mix(vec2(vb, vc)-va, vd - vec2(vc, vb), u.yx);
    vec2 grad = g + du * d;

    return vec3(value, grad);
}

float GradientNoise_2D_value_fbm(FBMParameters params, mat2 rotationMatrix, vec2 position) {
    float value = 0.0;
    float amplitude = 1.0;
    float currentFrequency = params.frequency;
    vec2 currPosition = position * params.frequency;
    for (uint i = 0; i < params.octaveCount; i++) {
        value += GradientNoise_2D_value(currPosition) * amplitude;
        amplitude *= params.persistence;
        currPosition = (rotationMatrix * currPosition) * params.lacunarity;
    }
    return value;
}

float GradientNoise_2D_value_hf_fbm(FBMParameters params, mat2 rotationMatrix, vec2 position) {
    float value = 0.0;
    float amplitude = 1.0;
    float currentFrequency = params.frequency;
    vec2 currPosition = position;
    for (uint i = 0; i < params.octaveCount; i++) {
        currPosition = (rotationMatrix * currPosition) * currentFrequency;
        value += GradientNoise_2D_value_hf(currPosition) * amplitude;
        amplitude *= params.persistence;
        currentFrequency *= params.lacunarity;
    }
    return value;
}

vec2 GradientNoise_2D_grad_fbm(FBMParameters params, mat2 rotationMatrix, vec2 position) {
    vec2 value = vec2(0.0);
    float amplitude = 1.0;
    float currentFrequency = params.frequency;
    vec2 currPosition = position;
    for (uint i = 0; i < params.octaveCount; i++) {
        currPosition = (rotationMatrix * currPosition) * currentFrequency;
        value += GradientNoise_2D_grad(currPosition) * amplitude;
        amplitude *= params.persistence;
        currentFrequency *= params.lacunarity;
    }
    return value;
}

vec3 GradientNoise_2D_valueGrad_fbm(FBMParameters params, mat2 rotationMatrix, vec2 position) {
    vec3 value = vec3(0.0);
    float amplitude = 1.0;
    float currentFrequency = params.frequency;
    vec2 currPosition = position;
    for (uint i = 0; i < params.octaveCount; i++) {
        currPosition = (rotationMatrix * currPosition) * currentFrequency;
        value += GradientNoise_2D_valueGrad(currPosition) * amplitude;
        amplitude *= params.persistence;
        currentFrequency *= params.lacunarity;
    }
    return value;
}


// -------------------------------------------------- Value Noise 3D --------------------------------------------------
vec3 _GradientNoise_3D_hash(uvec3 x) {
    return hash_uintToFloat(hash_33_q2(x)) * 2.0 - 1.0;
}

// [QUI13b]
float GradientNoise_3D_value(vec3 x) {
    uvec3 i = _noise_hash_coord(x);
    vec3 w = fract(x);

    vec3 u = _NOISE_INTERPO(w);

    vec3 ga = _GradientNoise_3D_hash(i + ivec3(0, 0, 0));
    vec3 gb = _GradientNoise_3D_hash(i + ivec3(1, 0, 0));
    vec3 gc = _GradientNoise_3D_hash(i + ivec3(0, 1, 0));
    vec3 gd = _GradientNoise_3D_hash(i + ivec3(1, 1, 0));
    vec3 ge = _GradientNoise_3D_hash(i + ivec3(0, 0, 1));
    vec3 gf = _GradientNoise_3D_hash(i + ivec3(1, 0, 1));
    vec3 gg = _GradientNoise_3D_hash(i + ivec3(0, 1, 1));
    vec3 gh = _GradientNoise_3D_hash(i + ivec3(1, 1, 1));

    float va = dot(ga, w - vec3(0.0, 0.0, 0.0));
    float vb = dot(gb, w - vec3(1.0, 0.0, 0.0));
    float vc = dot(gc, w - vec3(0.0, 1.0, 0.0));
    float vd = dot(gd, w - vec3(1.0, 1.0, 0.0));
    float ve = dot(ge, w - vec3(0.0, 0.0, 1.0));
    float vf = dot(gf, w - vec3(1.0, 0.0, 1.0));
    float vg = dot(gg, w - vec3(0.0, 1.0, 1.0));
    float vh = dot(gh, w - vec3(1.0, 1.0, 1.0));

    float xy0 = mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
    float xy1 = mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y);
    float value = mix(xy0, xy1, u.z);

    return value;
}

// [QUI17b]
vec4 GradientNoise_3D_valueGrad(vec3 x) {
    uvec3 i = _noise_hash_coord(x);
    vec3 w = fract(x);

    vec3 u = _NOISE_INTERPO(w);
    vec3 du = _NOISE_INTERPO_GRAD(w);

    vec3 ga = _GradientNoise_3D_hash(i + ivec3(0, 0, 0));
    vec3 gb = _GradientNoise_3D_hash(i + ivec3(1, 0, 0));
    vec3 gc = _GradientNoise_3D_hash(i + ivec3(0, 1, 0));
    vec3 gd = _GradientNoise_3D_hash(i + ivec3(1, 1, 0));
    vec3 ge = _GradientNoise_3D_hash(i + ivec3(0, 0, 1));
    vec3 gf = _GradientNoise_3D_hash(i + ivec3(1, 0, 1));
    vec3 gg = _GradientNoise_3D_hash(i + ivec3(0, 1, 1));
    vec3 gh = _GradientNoise_3D_hash(i + ivec3(1, 1, 1));

    float va = dot(ga, w - vec3(0.0, 0.0, 0.0));
    float vb = dot(gb, w - vec3(1.0, 0.0, 0.0));
    float vc = dot(gc, w - vec3(0.0, 1.0, 0.0));
    float vd = dot(gd, w - vec3(1.0, 1.0, 0.0));
    float ve = dot(ge, w - vec3(0.0, 0.0, 1.0));
    float vf = dot(gf, w - vec3(1.0, 0.0, 1.0));
    float vg = dot(gg, w - vec3(0.0, 1.0, 1.0));
    float vh = dot(gh, w - vec3(1.0, 1.0, 1.0));

    float xy0 = mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
    float xy1 = mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y);
    float value = mix(xy0, xy1, u.z);

    vec3 g = mix(mix(mix(ga, gb, u.x), mix(gc, gd, u.x), u.y), mix(mix(ge, gf, u.x), mix(gg, gh, u.x), u.y), u.z);
    vec3 d = mix(
        mix(vec3(vb, vc, ve) - va, vec3(vd - vc, vd - vb, vf - vb), u.yxx),
        mix(vec3(vf - ve, vg - ve, vg-vc), vh-vec3(vg, vf, vd), u.yxx),
        u.zzy
    );
    vec3 grad = g + du * d;

    return vec4(value, grad);
}

vec3 GradientNoise_3D_grad(vec3 x) {
    uvec3 i = _noise_hash_coord(x);
    vec3 w = fract(x);

    vec3 u = _NOISE_INTERPO(w);
    vec3 du = _NOISE_INTERPO_GRAD(w);

    vec3 ga = _GradientNoise_3D_hash(i + ivec3(0, 0, 0));
    vec3 gb = _GradientNoise_3D_hash(i + ivec3(1, 0, 0));
    vec3 gc = _GradientNoise_3D_hash(i + ivec3(0, 1, 0));
    vec3 gd = _GradientNoise_3D_hash(i + ivec3(1, 1, 0));
    vec3 ge = _GradientNoise_3D_hash(i + ivec3(0, 0, 1));
    vec3 gf = _GradientNoise_3D_hash(i + ivec3(1, 0, 1));
    vec3 gg = _GradientNoise_3D_hash(i + ivec3(0, 1, 1));
    vec3 gh = _GradientNoise_3D_hash(i + ivec3(1, 1, 1));

    float va = dot(ga, w - vec3(0.0, 0.0, 0.0));
    float vb = dot(gb, w - vec3(1.0, 0.0, 0.0));
    float vc = dot(gc, w - vec3(0.0, 1.0, 0.0));
    float vd = dot(gd, w - vec3(1.0, 1.0, 0.0));
    float ve = dot(ge, w - vec3(0.0, 0.0, 1.0));
    float vf = dot(gf, w - vec3(1.0, 0.0, 1.0));
    float vg = dot(gg, w - vec3(0.0, 1.0, 1.0));
    float vh = dot(gh, w - vec3(1.0, 1.0, 1.0));


    vec3 g = mix(mix(mix(ga, gb, u.x), mix(gc, gd, u.x), u.y), mix(mix(ge, gf, u.x), mix(gg, gh, u.x), u.y), u.z);
    vec3 d = mix(
        mix(vec3(vb, vc, ve) - va, vec3(vd - vc, vd - vb, vf - vb), u.yxx),
        mix(vec3(vf - ve, vg - ve, vg-vc), vh-vec3(vg, vf, vd), u.yxx),
        u.zzy
    );
    vec3 grad = g + du * d;

    return grad;
}

float GradientNoise_3D_value_fbm(FBMParameters params, vec3 position) {
    float value = 0.0;
    float amplitude = 1.0;
    float currentFrequency = params.frequency;
    for (uint i = 0; i < params.octaveCount; i++) {
        value += GradientNoise_3D_value(position * currentFrequency) * amplitude;
        amplitude *= params.persistence;
        currentFrequency *= params.lacunarity;
    }
    return value;
}

vec3 GradientNoise_3D_grad_fbm(FBMParameters params, vec3 position) {
    vec3 value = vec3(0.0);
    float amplitude = 1.0;
    float currentFrequency = params.frequency;
    for (uint i = 0; i < params.octaveCount; i++) {
        value += GradientNoise_3D_grad(position * currentFrequency) * amplitude;
        amplitude *= params.persistence;
        currentFrequency *= params.lacunarity;
    }
    return value;
}

#endif