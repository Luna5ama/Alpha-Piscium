#ifndef INCLUDE_displaytransform_TargetGamut
#define INCLUDE_displaytransform_TargetGamut

// Columns are the chromaticity vectors of the three primaries (Y = 1).
mat3 gamutToXYZ(vec2 r, vec2 g, vec2 b) {
    const vec3 white = vec3(.3127, .3290, .3583) / .3290;
    mat3 axes = mat3(vec3(r, 1.0-r.x-r.y)/r.y, vec3(g, 1.0-g.x-g.y)/g.y, vec3(b, 1.0-b.x-b.y)/b.y);
    vec3 scale = inverse(axes) * white;
    return mat3(axes[0]*scale.r, axes[1]*scale.g, axes[2]*scale.b);
}

mat3 rec709ToXYZ() { return gamutToXYZ(vec2(640,330)*.001, vec2(300,600)*.001, vec2(150,60)*.001); }
mat3 rec2020ToXYZ() { return gamutToXYZ(vec2(708,292)*.001, vec2(170,797)*.001, vec2(131,46)*.001); }
mat3 p3ToXYZ() { return gamutToXYZ(vec2(680,320)*.001, vec2(265,690)*.001, vec2(150,60)*.001); }

mat3 targetGamutToXYZ() {
    #if SETTING_TARGET_GAMUT == 0
    return rec709ToXYZ();
    #elif SETTING_TARGET_GAMUT == 2
    return rec2020ToXYZ();
    #elif SETTING_TARGET_GAMUT == 3
    const vec2 white = vec2(.3127, .3290);
    const vec2 r = vec2(SETTING_TARGET_RX, SETTING_TARGET_RY) * .001;
    const vec2 g = vec2(SETTING_TARGET_GX, SETTING_TARGET_GY) * .001;
    const vec2 b = vec2(SETTING_TARGET_BX, SETTING_TARGET_BY) * .001;
    mat2 triangle = mat2(r-b, g-b);
    if (max(max(r.x+r.y, g.x+g.y), b.x+b.y) <= 1.000001 && abs(determinant(triangle)) > 1e-6) {
        vec2 w = inverse(triangle) * (white-b);
        if (min(w.x, min(w.y, 1.0-w.x-w.y)) > 1e-5) return gamutToXYZ(r,g,b);
    }
    #endif
    return p3ToXYZ();
}

bool samePrimaries(mat3 a, mat3 b) {
    return all(lessThan(abs(a[0]-b[0]), vec3(1e-6))) &&
           all(lessThan(abs(a[1]-b[1]), vec3(1e-6))) &&
           all(lessThan(abs(a[2]-b[2]), vec3(1e-6)));
}
bool targetIsRec709() { return samePrimaries(targetGamutToXYZ(),rec709ToXYZ()); }
#endif
