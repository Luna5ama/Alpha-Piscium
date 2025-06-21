#ifndef INCLUDE_util_Mat4_glsl
#define INCLUDE_util_Mat4_glsl a

mat4 mat4_createOrthographicMatrix(float left, float right, float bottom, float top, float near, float far) {
    // Calculate matrix components
    float width = right - left;
    float height = top - bottom;
    float depth = far - near;

    // Avoid division by zero
    width = width != 0.0 ? width : 1.0;
    height = height != 0.0 ? height : 1.0;
    depth = depth != 0.0 ? depth : 1.0;

    // Build the orthographic projection matrix
    return mat4(
        2.0 / width, 0.0, 0.0, 0.0,
        0.0, 2.0 / height, 0.0, 0.0,
        0.0, 0.0, -2.0 / depth, 0.0,
        -(right + left) / width, -(top + bottom) / height, -(far + near) / depth, 1.0
    );
}

mat4 mat4_infRevZFromRegular(mat4 regularPerspective, float zNear) {
    mat4 infRevZ = regularPerspective;
    infRevZ[2][2] = 0.0;
    infRevZ[3][2] = -1.0;
    infRevZ[2][3] = zNear;
    infRevZ[3][3] = 0.0;
    return infRevZ;
}

#endif