#define CT_STORAGE restrict buffer
#include "/techniques/atmospherics/clouds/Transport.glsl"
#include "/util/Celestial.glsl"
#include "/util/Rand.glsl"

#if defined(CT_SETUP)
layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1, 1, 1);
shared uint phaseCounts[144];

void main() {
    uint lane = gl_LocalInvocationIndex;
    if (lane < 144u) phaseCounts[lane] = 0u;
    if (lane < 32u) ctStats(lane) = vec4(0.0);
    if (lane <= CT_N) {
        float t = float(lane) / float(CT_N);
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        float bottom = atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT - 2.5;
        float top = atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS + 0.15;
        ctEdges(lane) = vec4(ct_horizontalEdge(t), mix(bottom, top, t), 0.0, 0.0);
    }
    for (int i = int(lane); i < CT_ANGLES * CT_ANGLES; i += 256) {
        ctPhase(i) = mix(texelFetch(usam_cloudTransport, ivec2(i, 2), 0), texelFetch(usam_cloudTransport, ivec2(i, 3), 0), SETTING_CLOUDS_CU_PHASE_RATIO);
    }
    barrier();
    // Integrated direct-beam injection; inverse CDF sampling resolves the peak.
    // Fixed quadrature is independent of frame history and of cloud voxels.
    for (uint sampleIndex = lane; sampleIndex < 8192u; sampleIndex += 256u) {
        vec2 xi = rand_r2Seq2(sampleIndex);
        vec3 mu = ct_phaseCosines(xi.x);
        for (int light = 0; light < 2; ++light) {
            vec3 direction = -(light == 0 ? uval_sunDirWorld : uval_moonDirWorld);
            vec3 tangent = normalize(cross(direction, abs(direction.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
            vec3 bitangent = cross(direction, tangent);
            vec3 azimuth = tangent * cos(SPHERE_SOLID_ANGLE * 0.5 * xi.y) + bitangent * sin(SPHERE_SOLID_ANGLE * 0.5 * xi.y);
            for (int c = 0; c < 3; ++c) {
                vec3 omega = direction * mu[c] + azimuth * sqrt(1.0 - mu[c] * mu[c]);
                atomicAdd(phaseCounts[(light * 24 + ct_angle(omega)) * 3 + c], 1u);
            }
        }
    }
    barrier();
    if (lane < 24u) {
        int i = int(lane) * 3;
        ctSunPhase(lane) = vec4(vec3(phaseCounts[i], phaseCounts[i + 1], phaseCounts[i + 2]) / (8192.0 * CT_W), 0.0);
        i += 72;
        ctMoonPhase(lane) = vec4(vec3(phaseCounts[i], phaseCounts[i + 1], phaseCounts[i + 2]) / (8192.0 * CT_W), 0.0);
    }
}

#else
layout(local_size_x = 64) in;
#if CT_LEVEL == 0
const ivec3 workGroups = ivec3(CT_GROUPS, 1, 1);
#elif CT_LEVEL == 1
#if CT_N == 64
const ivec3 workGroups = ivec3(512, 1, 1);
#else
const ivec3 workGroups = ivec3(64, 1, 1);
#endif
#elif CT_LEVEL == 2
#if CT_N == 64
const ivec3 workGroups = ivec3(64, 1, 1);
#else
const ivec3 workGroups = ivec3(8, 1, 1);
#endif
#elif CT_LEVEL == 3 && CT_N == 64
const ivec3 workGroups = ivec3(8, 1, 1);
#else
const ivec3 workGroups = ivec3(1, 1, 1);
#endif

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int n = CT_N >> CT_LEVEL;
    if (i >= n * n * n) return;
    ivec3 p = ct_position(i, n);
    float density = 0.0;
    #if CT_LEVEL == 0
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec2 layerRange = atmosphere.bottom + vec2(SETTING_CLOUDS_CU_HEIGHT, SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS);
    vec3 start = ct_edge(p, 0) + ct_worldCenter();
    vec3 h = ct_size(p, 0);
    vec3 sun = vec3(0.0);
    vec3 moon = vec3(0.0);
    if (SETTING_CLOUDS_CU_THICKNESS > 0.0 && CU_DENSITY_SCALE > 0.0 && SETTING_CLOUDS_CU_COVERAGE > 0.0) {
        for (int s = 0; s < 4; ++s) {
            ivec3 bits = ivec3(s & 1, s >> 1, (s ^ (s >> 1)) & 1);
            vec3 pos = start + (vec3(bits) * 0.5 + 0.25) * h;
            float height = length(pos);
            float rho = clouds_cu_density(pos, (height - layerRange.x) / (layerRange.y - layerRange.x));
            density += rho * 0.25;
            if (rho == 0.0) continue;
            for (int light = 0; light < 2; ++light) {
                vec3 direction = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
                if (raySphereIntersectNearest(pos, direction, vec3(0.0), atmosphere.bottom) >= 0.0) continue;
                vec3 irradiance = light == 0 ? SUN_ILLUMINANCE : MOON_ILLUMINANCE;
                vec3 airTr = atmospherics_air_lut_sampleTransmittance(atmosphere, dot(pos / height, direction), height);
                vec3 incident = ct_toAP0(irradiance * airTr);
                vec3 tr = exp(-CT_EXTINCTION * clouds_cu_column(pos, direction, layerRange, 16, 6.0, 0.5));
                vec3 correlatedSource = incident * tr * (rho * 0.25);
                if (light == 0) sun += correlatedSource;
                else moon += correlatedSource;
            }
        }
    }
    ctMaterialSun(i) = vec4(sun, density);
    ctMaterialMoon(i) = vec4(moon, 0.0);
    #else
    for (int corner = 0; corner < 8; ++corner) {
        ivec3 q = p * 2 + (ivec3(corner) >> ivec3(0, 1, 2) & 1);
        int child = ct_levelOffset(CT_LEVEL - 1) + ct_index(q, n * 2);
        density += ctCoefficient(child, 0).a * ct_volume(q, CT_LEVEL - 1);
    }
    density /= ct_volume(p, CT_LEVEL);
    #endif
    CTCoefficients c = ct_coefficients(density, pow(ct_volume(p, CT_LEVEL), 1.0 / 3.0));
    int destination = ct_levelOffset(CT_LEVEL) + i;
    ctCoefficient(destination, 0) = c.slow;
    ctCoefficient(destination, 1) = c.fast;
    ctCoefficient(destination, 2) = c.weight;
    ctCoefficient(destination, 3) = c.exchange;
}
#endif
