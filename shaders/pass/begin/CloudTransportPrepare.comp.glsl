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
    if (lane < 26u) ctStats(lane) = vec4(0.0);
    if (lane <= CT_N) {
        float t = float(lane) / float(CT_N);
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        float bottom = atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT - 2.5;
        float top = atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS + 0.15;
        ctEdges(lane) = vec4(ct_horizontalEdge(t), mix(bottom, top, t), 0.0, 0.0);
    }
    if (lane < CT_ANGLES) {
        int row = int(lane) * CT_ANGLES;
        vec3 diagonal = mix(texelFetch(usam_cloudTransport, ivec2(row + int(lane), 2), 0).rgb,
            texelFetch(usam_cloudTransport, ivec2(row + int(lane), 3), 0).rgb, SETTING_CLOUDS_CU_PHASE_RATIO);
        float forward = mmin3(diagonal);
        float mass = 1.0 - forward;
        float cdf = 0.0;
        for (int j = 0; j < CT_ANGLES; ++j) {
            vec3 phase = mix(texelFetch(usam_cloudTransport, ivec2(row + j, 2), 0).rgb,
                texelFetch(usam_cloudTransport, ivec2(row + j, 3), 0).rgb, SETTING_CLOUDS_CU_PHASE_RATIO);
            vec3 reduced = phase - (j == int(lane) ? forward : 0.0);
            cdf += dot(reduced, vec3(1.0 / 3.0)) / mass;
            bool last = j == CT_ANGLES - 1 || (int(lane) == CT_ANGLES - 1 && mmax3(diagonal) == forward && j == CT_ANGLES - 2);
            ctPhase(row + j) = vec4(phase, last ? 1.0 : cdf);
        }
    }
    barrier();
    if (lane == 0u) {
        vec4 previousCenter = ctStats(26);
        vec4 previousSun = ctStats(27);
        vec3 previousMoon = ctStats(28).rgb;
        bool reuse = global_historyResetFactor > 0.0 && frameCounter > 0 && previousSun.a == float(frameCounter - 1);
        reuse = reuse && dot(previousSun.xyz, uval_sunDirWorld) > cos(SUN_ANGULAR_RADIUS);
        reuse = reuse && dot(previousMoon, uval_moonDirWorld) > cos(MOON_ANGULAR_RADIUS);
        vec3 displacement = vec3(0.0);
        if (reuse) displacement = ct_worldCenter() - previousCenter.xyz + clouds_cu_wind(frameTimeCounter) - clouds_cu_wind(previousCenter.a);
        ctStats(24) = vec4(0.0, float(reuse), 0.0, 0.0);
        ctStats(25) = vec4(displacement, 0.0);
        ctStats(26) = vec4(ct_worldCenter(), frameTimeCounter);
        ctStats(27) = vec4(uval_sunDirWorld, float(frameCounter));
        ctStats(28) = vec4(uval_moonDirWorld, 0.0);
        ctStats(29) = vec4(0.0);
        ctStats(30) = vec4(0.0);
        ctStats(31) = vec4(0.0);
    }
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

#elif defined(CT_HISTORY)
layout(local_size_x = 64) in;
const ivec3 workGroups = ivec3(CT_GROUPS, 1, 1);

void main() {
    int i = int(gl_GlobalInvocationID.x);
    bool reuse = ctStats(24).y > 0.0;
    vec3 displacement = ctStats(25).xyz;
    bool stationary = all(equal(displacement, vec3(0.0)));
    ivec3 base;
    vec3 f;
    bool inside = false;
    if (reuse && !stationary) inside = ct_interpolation(ct_center(ct_position(i, CT_N), 0) + ct_worldCenter() + displacement, base, f);
    for (int angle = 0; angle < CT_ANGLES; ++angle) {
        vec3 previous = vec3(0.0);
        float forward = ct_forwardFraction(angle);
        if (reuse && stationary) previous = ctSource(angle * CT_CELLS + i).rgb - forward * ctLifted(angle * CT_CELLS + i).rgb;
        else if (inside) {
            for (int corner = 0; corner < 8; ++corner) {
                ivec3 offset = ivec3(corner) >> ivec3(0, 1, 2) & 1;
                vec3 w = mix(1.0 - f, f, vec3(offset));
                int a = angle * CT_CELLS + ct_index(base + offset, CT_N);
                previous += (ctSource(a).rgb - forward * ctLifted(a).rgb) * (w.x * w.y * w.z);
            }
        }
        ctRadiance(angle * CT_CELLS + i) = vec4(previous, 0.0);
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
            for (int light = 0; light < 2; ++light) {
                vec3 direction = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
                vec3 f = pos - start;
                vec3 d = max(abs(direction), vec3(1e-20));
                float nearDistance = mmin3(mix(f, h - f, greaterThan(direction, vec3(0.0))) / d);
                float farDistance = mmin3(mix(h - f, f, greaterThan(direction, vec3(0.0))) / d);
                float rayLength = nearDistance + farDistance;
                vec3 entry = pos + direction * nearDistance;
                bool occluded = raySphereIntersectNearest(entry, direction, vec3(0.0), atmosphere.bottom) >= 0.0;
                if (light == 1 && occluded) continue;
                float column = 0.0;
                for (int tap = 0; tap < 8; ++tap) {
                    vec3 point = entry - direction * ((float(tap) + 0.5) * (rayLength / 8.0));
                    float heightFraction = (length(point) - layerRange.x) / (layerRange.y - layerRange.x);
                    column += clouds_cu_density(point, heightFraction) * (rayLength / 8.0);
                }
                if (light == 0) density += column / rayLength * 0.25;
                if (column == 0.0 || occluded) continue;
                vec3 irradiance = light == 0 ? SUN_ILLUMINANCE : MOON_ILLUMINANCE;
                float height = length(entry);
                vec3 airTr = atmospherics_air_lut_sampleTransmittance(atmosphere, dot(entry / height, direction), height);
                vec3 incident = ct_toAP0(irradiance * airTr);
                vec3 tr = exp(-CT_EXTINCTION * clouds_cu_column(entry, direction, layerRange, 16, 0.5));
                vec3 tau = CT_EXTINCTION * column;
                vec3 opacity = mix(1.0 - exp(-tau), tau * (1.0 - tau * 0.5 + tau * tau / 6.0), lessThan(tau, vec3(0.001)));
                // Integrate the first-scattering layer instead of point-sampling
                // rho*T, which misses almost all injection in optically thick cells.
                vec3 correlatedSource = incident * tr * opacity / (CT_EXTINCTION * rayLength) * 0.25;
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
