#ifndef INCLUDE_util_Mat2_glsl
#define INCLUDE_util_Mat2_glsl a

#define MAT2_GOLDEN_ANGLE mat2(-0.737368878, 0.675490294, -0.675490294, -0.737368878)

mat2 mat2_identity() {
    return mat2(
        1.0, 0.0,
        0.0, 1.0
    );
}

mat2 mat2_rotate(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(
        c,   -s,
        s,    c
    );
}

#endif