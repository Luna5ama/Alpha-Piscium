#ifndef INCLUDE_util_Mat4_glsl
#define INCLUDE_util_Mat4_glsl a

mat4 mat4_createOrthographicMatrix(float left, float right, float bottom, float top, float nearZ, float farZ) {
    // Calculate matrix components
    float width = right - left;
    float height = top - bottom;
    float depth = farZ - nearZ;

    // Avoid division by zero
    width = width != 0.0 ? width : 1.0;
    height = height != 0.0 ? height : 1.0;
    depth = depth != 0.0 ? depth : 1.0;

    // Build the orthographic projection matrix
    return mat4(
        2.0 / width, 0.0, 0.0, 0.0,
        0.0, 2.0 / height, 0.0, 0.0,
        0.0, 0.0, -2.0 / depth, 0.0,
        -(right + left) / width, -(top + bottom) / height, -(farZ + nearZ) / depth, 1.0
    );
}

mat4 mat4_infRevZFromRegular(mat4 regularPerspective, float zNear) {
    return mat4(
        regularPerspective[0][0], regularPerspective[0][1], regularPerspective[0][2], regularPerspective[0][3],
        regularPerspective[1][0], regularPerspective[1][1], regularPerspective[1][2], regularPerspective[1][3],
        regularPerspective[2][0], regularPerspective[2][1], 0.0, -1.0,
        regularPerspective[3][0], regularPerspective[3][1], zNear, 0.0
    );
}

#endif