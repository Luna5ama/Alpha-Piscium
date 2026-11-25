#ifndef INCLUDE_util_VoxyFaceNormal_glsl
#define INCLUDE_util_VoxyFaceNormal_glsl a

vec3 voxy_faceWorldNormal(uint face) {
    return normalize(vec3(
        uint((face >> 1u) == 2u),
        uint((face >> 1u) == 0u),
        uint((face >> 1u) == 1u)
    ) * (float(int(face) & 1) * 2.0 - 1.0));
}

#endif
