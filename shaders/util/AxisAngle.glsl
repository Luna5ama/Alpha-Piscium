#ifndef INCLUDE_util_AxisAngle_glsl
#define INCLUDE_util_AxisAngle_glsl a

struct AxisAngle {
    vec3 axis; // axis vector
    vec3 csc; // cos(angle), sin(angle), 1 - cos(angle)
};

AxisAngle AxisAngle_init(vec3 axis, float angle) {
    AxisAngle aa;
    aa.axis = axis;
    float c = cos(angle);
    float s = sin(angle);
    aa.csc = vec3(c, s, 1.0 - c);
    return aa;
}

// Rodrigues' Rotation Formula
vec3 AxisAngle_transform(AxisAngle aa, vec3 v) {
    return v * aa.csc.x + cross(aa.axis, v) * aa.csc.y + aa.axis * (dot(v, aa.axis) * aa.csc.z);
}

#endif