#ifndef INCLUDE_clouds_CumulusRay_glsl
#define INCLUDE_clouds_CumulusRay_glsl

#include "Cirrus.glsl"
#include "Cumulus.glsl"
#ifndef ATMOSPHERE_RAYMARCHING_SKY
#define ATMOSPHERE_RAYMARCHING_SKY
#include "../air/Raymarching.glsl"
#undef ATMOSPHERE_RAYMARCHING_SKY
#endif

void clouds_cu_lightColumn(vec3 receiver, vec3 direction, CuField field, CloudRenderParams light,
    float jitter, float lod, out vec3 opticalDepth, out vec3 isotropicMS) {
    opticalDepth = vec3(0.0);
    isotropicMS = vec3(0.0);
    vec2 interval;
    if (!clouds_cu_interval(receiver, direction, CU_RANGE, interval)) return;

    float top = mix(1.0, clamp((dot(field.normal, light.lightDir) + 0.5) / 1.5, 0.0, 1.0), 0.25);
    float boundary = top * (1.0 - exp(-field.height / (0.1 * mix(1.0, 4.0, field.development))));
    vec3 buildRate = max(1.0 - CLOUDS_CU_ASYM, vec3(0.0));
    // Keep the eight local, quadratic-distance bins used by the existing multiple-scattering estimator.
    float localEnd = min(interval.y, float(SETTING_CLOUDS_CU_THICKNESS));
    for (int i = 0; i < 8; ++i) {
        float begin = float(i * i) * (1.0 / 64.0) * localEnd;
        float ds = float(2 * i + 1) * (1.0 / 64.0) * localEnd;
        float t = begin + jitter * ds;
        float rho = clouds_cu_field(receiver + direction * t, direction, lod, true).density * CU_DENSITY;
        if (rho > 0.0) {
            vec3 sigmaT = CLOUDS_CU_EXTINCTION * rho;
            vec3 prefix = opticalDepth + sigmaT * (jitter * ds);
            vec3 weight = CLOUDS_CU_SCATTERING * rho * ds * sigmaT * boundary / max(t, 0.5 * ds);
            isotropicMS += weight * exp(-(0.001 + sqrt(0.003)) * prefix) * (1.0 - exp(-buildRate * prefix));
            opticalDepth += sigmaT * ds;
        }
    }
    isotropicMS = 1.0 - exp(-max(isotropicMS * SETTING_CLOUDS_CU_ISOTROPIC_MS_INTENSITY, vec3(0.0)));

    // A separate tower beyond the local diffusion column still casts a direct shadow.
    float t = localEnd;
    const int farBins = 16;
    for (int i = 0; i < farBins && t < interval.y && opticalDepth.y < 5.3; ++i) {
        float empty;
        float ds = clouds_cu_bin(receiver + direction * t, direction, 0.10 + 0.028 * t, interval.y - t, i, farBins, empty);
        if (ds > empty) {
            float probe = t + mix(empty, ds, jitter);
            CuField cloud = clouds_cu_field(receiver + direction * probe, direction, lod, true);
            opticalDepth += CLOUDS_CU_EXTINCTION * (cloud.density * CU_DENSITY * (ds - empty));
            t = max(t + ds, probe + cloud.bound);
        } else {
            t += ds;
        }
    }
}

ScatteringResult clouds_cu_airPrefix(AtmosphereParameters atmosphere, ScatteringParameters scattering,
    vec3 atmOrigin, vec3 atmOffset) {
    vec3 end = atmOrigin + atmOffset;
    if (all(equal(end, atmOrigin))) return scatteringResult_init();
    RaymarchParameters ray = RaymarchParameters(atmOrigin, end, 1u);
    return raymarchSky(atmosphere, ray, scattering, 8.0);
}

// Integrate extinction in every view bin. Select a source from those bins in
// proportion to green-channel opacity so that only one sun column is needed.
struct CuTrace {
    float column;
    float meanDistance;
    float sourceDistance;
    float sourceStep;
    float sourceDensity;
    float sourcePrefix;
    float sourceWeight;
};

CuTrace clouds_cu_trace(vec3 origin, vec3 direction, vec2 interval, vec2 jitter, int count) {
    CuTrace trace = CuTrace(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    float t = interval.x;
    float transmittance = 1.0;
    float u = jitter.y;
    int occupiedBins = 0;
    // Empty cell crossings get their own allowance. After that each advances the view budget too.
    const int emptyCrossings = 64;
    for (int i = 0; i < count + emptyCrossings && occupiedBins < count && t < interval.y && transmittance > 0.003; ++i) {
        float empty;
        int bin = max(occupiedBins, i - emptyCrossings);
        float target = 0.70 / float(SETTING_CLOUDS_LOW_STEP_MIN) + t * 0.0012;
        float ds = clouds_cu_bin(origin + direction * t, direction, target, interval.y - t, bin, count, empty);
        if (ds <= empty) {
            t += ds;
            continue;
        }
        ++occupiedBins;
        float span = ds - empty;
        int strata = clamp(int(ceil(span / (0.05 + 0.0009 * t))), 1, 4);
        float sampleStep = span / float(strata);
        float next = t + ds;
        for (int j = 0; j < strata && transmittance > 0.003; ++j) {
            float probe = t + empty + (float(j) + jitter.x) * sampleStep;
            float lod = smoothstep(5.0, 18.0, probe);
            CuField field = clouds_cu_field(origin + direction * probe, direction, lod, true);
            float column = field.density * CU_DENSITY * sampleStep;
            if (column > 0.0) {
                float nextT = exp(-CLOUDS_CU_EXTINCTION.y * (trace.column + column));
                float weight = transmittance - nextT;
                if (weight > 0.0) {
                    float probability = weight / (1.0 - nextT);
                    if (probability == 1.0 || u < probability) {
                        trace.sourceDistance = probe;
                        trace.sourceStep = sampleStep;
                        trace.sourceDensity = field.density;
                        trace.sourcePrefix = trace.column;
                        trace.sourceWeight = weight;
                        u /= probability;
                    } else {
                        u = (u - probability) / (1.0 - probability);
                    }
                    trace.meanDistance += probe * weight;
                }
                trace.column += column;
                transmittance = nextT;
            }
            next = max(next, probe + field.bound);
        }
        t = next;
    }
    if (trace.sourceWeight > 0.0) trace.meanDistance /= 1.0 - transmittance;
    return trace;
}

ScatteringResult clouds_cu_render(AtmosphereParameters atmosphere, vec3 origin, vec3 atmOrigin,
    vec3 direction, vec2 segment, vec4 jitter, bool environment, out float meanDistance) {
    ScatteringResult result = scatteringResult_init();
    meanDistance = 0.0;
    vec2 interval;
    if (!clouds_cu_interval(origin, direction, segment.y, interval)) return result;
    interval.x = max(interval.x, segment.x);
    if (interval.x >= interval.y) return result;

    int count = environment ? min(40, max(16, SETTING_CLOUDS_LOW_STEP_MAX / 4)) : SETTING_CLOUDS_LOW_STEP_MAX;
    CuTrace trace = clouds_cu_trace(origin, direction, interval, jitter.xw, count);
    result.transmittance = exp(-CLOUDS_CU_EXTINCTION * trace.column);
    meanDistance = trace.meanDistance;
    if (trace.sourceWeight == 0.0) return result;

    CloudRenderParams light = clouds_renderParams(direction, jitter.z);
    CloudParticpatingMedium medium = clouds_cu_medium(light.cosLightTheta);
    vec2 ambUV = cloods_amblut_uv(direction, jitter.yz);
    vec3 ambient = clouds_amblut_sample(ambUV, CLOUDS_AMBLUT_LAYER_CUMULUS);
    vec3 lightDir = rand_sampleInCone(light.lightDir, SUN_ANGULAR_RADIUS * 4.0, jitter.yz);
    float probe = trace.sourceDistance;
    float lod = smoothstep(5.0, 18.0, probe);
    vec3 pos = origin + direction * probe;
    CuField field = clouds_cu_field(pos, direction, lod, true);
    vec3 opticalDepth, isotropicMS;
    clouds_cu_lightColumn(pos, lightDir, field, light, jitter.y, lod, opticalDepth, isotropicMS);
    vec3 atmDelta = direction * (1000.0 / float(SETTING_ATM_D_SCALE));
    vec3 airStart = atmOrigin + atmDelta * segment.x;
    ScatteringResult bin = clouds_computeLighting(atmosphere, light, medium, ambient, atmOrigin + atmDelta * probe,
        trace.sourceStep, trace.sourceDensity * CU_DENSITY, field.bulk * CU_DENSITY, opticalDepth, isotropicMS);
    ScatteringParameters scattering = scatteringParameters_init(
        lightParameters_init(atmosphere, SUN_ILLUMINANCE, uval_sunDirWorld, direction),
        lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, direction), 1.0);

    ScatteringResult air = clouds_cu_airPrefix(atmosphere, scattering, airStart, atmDelta * (probe - segment.x));
    #ifdef SETTING_CLOUDS_CI
    float ciDistance = clouds_ci_distance(atmosphere, atmOrigin, normalize(atmDelta)) / length(atmDelta);
    if (ciDistance >= segment.x && ciDistance < probe) {
        vec3 offset = atmDelta * ciDistance;
        ScatteringResult ci = clouds_ci_layer(atmosphere, atmOrigin + offset, light, ambUV);
        ScatteringResult airToCirrus = clouds_cu_airPrefix(atmosphere, scattering, airStart, atmDelta * (ciDistance - segment.x));
        air.inScattering = airToCirrus.inScattering + airToCirrus.transmittance * ci.inScattering
            + ci.transmittance * (air.inScattering - airToCirrus.inScattering);
        air.transmittance *= ci.transmittance;
    }
    #endif

    vec3 prefix = exp(-CLOUDS_CU_EXTINCTION * trace.sourcePrefix);
    float inverseProbability = (1.0 - result.transmittance.y) / trace.sourceWeight;
    result.inScattering = prefix * inverseProbability
        * (air.transmittance * bin.inScattering + (1.0 - bin.transmittance) * air.inScattering);
    return result;
}

#endif
