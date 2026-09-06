#define CT_STORAGE coherent restrict buffer
#include "/techniques/atmospherics/clouds/Transport.glsl"

layout(local_size_x = 1024) in;
const ivec3 workGroups = ivec3(24, 1, 1);
#ifdef CT_UNCOLLIDED
shared vec3 boundary;
#endif

void main() {
    int angle = int(gl_WorkGroupID.x);
    vec3 omega = ct_direction(angle);
    ivec3 signV = ivec3(sign(omega));
    #ifdef CT_UNCOLLIDED
    if (gl_LocalInvocationIndex == 0u) {
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        vec2 layerRange = atmosphere.bottom + vec2(SETTING_CLOUDS_CU_HEIGHT, SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS);
        boundary = ct_boundary(ct_worldCenter() + vec3(0.0, (layerRange.x + layerRange.y) * 0.5, 0.0), omega, layerRange, 0.5);
    }
    barrier();
    #endif
    // One group owns one ordinate. Planes of x+y+z have no dependencies within
    // a plane; buffer barriers publish every upwind value before the next plane.
    for (int plane = 0; plane <= 3 * (CT_N - 1); ++plane) {
        for (int xy = int(gl_LocalInvocationIndex); xy < CT_N * CT_N; xy += 1024) {
            ivec3 logical = ivec3(xy % CT_N, xy / CT_N, plane - xy % CT_N - xy / CT_N);
            if (logical.z < 0 || logical.z >= CT_N) continue;
            ivec3 p = mix(ivec3(CT_N - 1) - logical, logical, greaterThan(signV, ivec3(0)));
            int i = ct_index(p, CT_N);
            vec3 rates = abs(omega) / ct_size(p, 0);
            vec3 extinction = ctCoefficient(i, 0).a * CT_EXTINCTION;
            #ifdef CT_UNCOLLIDED
            vec3 numerator = vec3(0.0);
            #elif defined(CT_CORRECTION)
            vec3 numerator = ctCoefficient(i, 0).a * CT_SCATTERING * ct_delta(p);
            #else
            vec3 numerator = ctSource(angle * CT_CELLS + i).rgb;
            #endif
            for (int axis = 0; axis < 3; ++axis) {
                ivec3 q = p - ivec3(equal(ivec3(axis), ivec3(0, 1, 2))) * signV;
                if (ct_inside(q, CT_N)) {
                    #ifdef CT_CORRECTION
                    numerator += rates[axis] * ctLifted(angle * CT_CELLS + ct_index(q, CT_N)).rgb;
                    #else
                    numerator += rates[axis] * ctRadiance(angle * CT_CELLS + ct_index(q, CT_N)).rgb;
                    #endif
                }
                #ifdef CT_UNCOLLIDED
                else numerator += rates[axis] * boundary;
                #endif
            }
            #ifdef CT_CORRECTION
            ctLifted(angle * CT_CELLS + i) = vec4(numerator / (extinction + rates.x + rates.y + rates.z), 0.0);
            #else
            ctRadiance(angle * CT_CELLS + i) = vec4(numerator / (extinction + rates.x + rates.y + rates.z), 0.0);
            #endif
        }
        memoryBarrierBuffer();
        barrier();
    }
}
