#ifndef INCLUDE_GI_OBSERVED_SURFACE_FOOTPRINT
#define INCLUDE_GI_OBSERVED_SURFACE_FOOTPRINT

// Require the pixel footprint on the native face to fit one block. A coarse
// observation must not turn the air beside a distant lamp into another lamp.
bool gi_singleObservedCell(vec3 ray, mat3 viewToWorld, vec2 invFocalLength, vec3 normal, float z, vec3 fraction) {
    float facing = dot(ray, normal);
    if (abs(facing) < 1e-5) return false;
    vec2 scale = -2.0 * z * uval_mainImageSizeRcp
               * invFocalLength;
    vec3 dx = viewToWorld[0] * scale.x;
    vec3 dy = viewToWorld[1] * scale.y;
    dx -= ray * (dot(dx, normal) / facing);
    dy -= ray * (dot(dy, normal) / facing);
    vec3 margin = min(fraction, 1.0 - fraction) + abs(normal);
    return all(greaterThan(margin, 0.5 * (abs(dx) + abs(dy)) + 0.001));
}
#endif
