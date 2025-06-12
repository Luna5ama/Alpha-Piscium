#ifndef INCLUDE_util_Mat3_glsl
#define INCLUDE_util_Mat3_glsl a

mat3 mat3_rotateX(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
        1.0, 0.0, 0.0,
        0.0, c,   -s,
        0.0, s,    c
    );
}

mat3 mat3_rotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
        c,   0.0, s,
        0.0, 1.0, 0.0,
       -s,   0.0, c
    );
}

mat3 mat3_rotateZ(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
        c,   -s,  0.0,
        s,    c,  0.0,
        0.0,  0.0, 1.0
    );
}

#endif