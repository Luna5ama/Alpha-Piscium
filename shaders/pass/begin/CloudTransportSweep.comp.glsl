#define CT_STORAGE restrict buffer
#include "/techniques/atmospherics/clouds/Transport.glsl"

layout(local_size_x = 1024) in;
const ivec3 workGroups = ivec3(24, 1, 1);
shared float planeRadiance[3 * CT_N * CT_N];

void main() {
    if (ct_skipIteration()) return;
    int angle = int(gl_WorkGroupID.x);
    #ifdef CT_UNCOLLIDED
    float crossSection = CLOUDS_CU_EXTINCTION_COEFF;
    #else
    float crossSection = ct_reducedExtinction(angle);
    #endif
    vec3 omega = ct_direction(angle);
    ivec3 signV = ivec3(sign(omega));
    #ifdef CT_UNCOLLIDED
    if (gl_LocalInvocationIndex == 0u) {
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        vec2 layerRange = atmosphere.bottom + vec2(SETTING_CLOUDS_CU_HEIGHT, SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS);
        vec3 boundary = ct_boundary(ct_worldCenter() + vec3(0.0, (layerRange.x + layerRange.y) * 0.5, 0.0), omega, layerRange, 0.5);
        planeRadiance[0] = boundary.r;
        planeRadiance[CT_N * CT_N] = boundary.g;
        planeRadiance[2 * CT_N * CT_N] = boundary.b;
    }
    barrier();
    vec3 boundary = vec3(planeRadiance[0], planeRadiance[CT_N * CT_N], planeRadiance[2 * CT_N * CT_N]);
    #endif
    // Read the previous x+y+z plane before overwriting it. Descending chunks
    // keep lower-index dependencies intact and fit a 64^2 RGB plane in 48 KiB.
    for (int plane = 0; plane <= 3 * (CT_N - 1); ++plane) {
        for (int chunk = CT_N * CT_N - 1024; chunk >= 0; chunk -= 1024) {
            int xy = chunk + int(gl_LocalInvocationIndex);
            ivec3 logical = ivec3(xy % CT_N, xy / CT_N, plane - xy % CT_N - xy / CT_N);
            ivec3 p = mix(ivec3(CT_N - 1) - logical, logical, greaterThan(signV, ivec3(0)));
            int i = ct_index(p, CT_N);
            bool cellValid = logical.z >= 0 && logical.z < CT_N;
            vec3 radiance = vec3(0.0);
            vec3 mean = vec3(0.0);
            if (cellValid) {
                vec3 rates = abs(omega) / ct_size(p, 0);
                float extinction = ctCoefficient(i, 0).a * crossSection;
                #ifdef CT_UNCOLLIDED
                vec3 source = vec3(0.0);
                #elif defined(CT_CORRECTION)
                vec3 source = ctCoefficient(i, 0).a * CLOUDS_CU_SCATTERING_COEFF * (1.0 - ct_forwardFraction(angle)) * ct_delta(p);
                #else
                vec3 source = ctSource(angle * CT_CELLS + i).rgb;
                #endif
                vec3 incoming = vec3(0.0);
                for (int axis = 0; axis < 3; ++axis) {
                    if (logical[axis] > 0) {
                        int upstream = xy - (axis == 0 ? 1 : (axis == 1 ? CT_N : 0));
                        incoming += rates[axis] * vec3(planeRadiance[upstream], planeRadiance[CT_N * CT_N + upstream], planeRadiance[2 * CT_N * CT_N + upstream]);
                    }
                    #ifdef CT_UNCOLLIDED
                    else incoming += rates[axis] * boundary;
                    #endif
                }
                float sumRate = rates.x + rates.y + rates.z;
                incoming /= sumRate;
                float tau = extinction / sumRate;
                float tr = exp(-tau);
                float transfer = tau < 0.001 ? 1.0 - tau * 0.5 + tau * tau / 6.0 : (1.0 - tr) / tau;
                radiance = tr * incoming + transfer * (source / sumRate);
                mean = mix(radiance, incoming, ct_incomingWeight(tau));
            }
            barrier();
            if (cellValid) {
                planeRadiance[xy] = radiance.r;
                planeRadiance[CT_N * CT_N + xy] = radiance.g;
                planeRadiance[2 * CT_N * CT_N + xy] = radiance.b;
                #ifdef CT_UNCOLLIDED
                ctLifted(angle * CT_CELLS + i) = vec4(radiance, 0.0);
                ctSky(angle * CT_CELLS + i) = vec4(mean, 0.0);
                #elif defined(CT_CORRECTION)
                ctLifted(angle * CT_CELLS + i) = vec4(radiance, 0.0);
                #else
                ctRadiance(angle * CT_CELLS + i) = vec4(radiance, 0.0);
                ctLifted(angle * CT_CELLS + i) = vec4(mean, 0.0);
                #endif
            }
            barrier();
        }
    }
}
