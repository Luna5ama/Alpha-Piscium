#ifndef INCLUDE_util_noise__Common_glsl
#define INCLUDE_util_noise__Common_glsl a

#include "/util/Hash.glsl"

// 0: cubic
// 1: quintic
#define _NOISE_INTERPOLANT 1

struct FBMParameters {
    float frequency;
    float persistence;
    float lacunarity;
    uint octaveCount;
};

#if _NOISE_INTERPOLANT == 1
#define _NOISE_INTERPO(w) (w * w * w * (w * (w * 6.0 - 15.0) + 10.0))
#define _NOISE_INTERPO_GRAD(w) (30.0 * w * w * (w * (w - 2.0) + 1.0))
#else
#define _NOISE_INTERPO(w) (w * w * (3.0 - 2.0 * w))
#define _NOISE_INTERPO_GRAD(w) (6.0 * w * (1.0 - w))
#endif

ivec2 _noise_hash_coord_signed(vec2 x) {
    return ivec2(floor(x));
}

ivec3 _noise_hash_coord_signed(vec3 x) {
    return ivec3(floor(x));
}

uvec2 _noise_hash_coord(vec2 x) {
    return uvec2(_noise_hash_coord_signed(x));
}

uvec3 _noise_hash_coord(vec3 x) {
    return uvec3(_noise_hash_coord_signed(x));
}

#endif