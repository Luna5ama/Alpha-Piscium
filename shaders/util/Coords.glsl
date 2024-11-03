#ifndef INCLUDE_Coords.glsl
#define INCLUDE_Coords.glsl
#include "../_Base.glsl"
#include "Math.glsl"
#include "R2Seqs.glsl"

float coords_linearizeDepth(float depth, float near, float far) {
    return (near * far) / (depth * (near - far) + far);
}

vec3 coords_toViewCoord(vec2 texCoord, float viewZ, mat4 projInv) {
    vec2 ndcXY = texCoord * 2.0 - 1.0;
    vec2 clipXY = ndcXY * -viewZ;
    vec2 viewXY = clipXY * vec2(projInv[0][0], projInv[1][1]);
    return vec3(viewXY, viewZ);
}

mat4 coords_shadowDeRotateMatrix(mat4 shadowMatrix) {
    vec2 p1 = (shadowMatrix * vec4(0.0, -1000.0, 0.0, 1.0)).xy;
    vec2 p2 = (shadowMatrix * vec4(0.0, 1000.0, 0.0, 1.0)).xy;

    float angle1 = -atan(p1.y, p1.x);

    float cos1 = cos(angle1 - PI_HALF_CONST) * 0.9;
    float sin1 = sin(angle1 - PI_HALF_CONST) * 0.9;

    return mat4(
            cos1, sin1, 0.0, 0.0,
            -sin1, cos1, 0.0, 0.0,
            0.0, 0.0, 0.1, 0.0,
            0.0, 0.0, 0.0, 1.0
    );
}

vec2 coords_taaJitter() {
    return r2Seq2(frameCounter) - 0.5;
}

mat4 coords_taaJitterMat() {
    vec2 jitter = coords_taaJitter() * 2.0 * viewResolution.zw;
    return mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            jitter.x, jitter.y, 0.0, 1.0
    );
}
#endif