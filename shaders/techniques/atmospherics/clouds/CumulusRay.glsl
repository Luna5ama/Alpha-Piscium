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
    while (t < interval.y && opticalDepth.y < 5.3) {
        bool empty;
        float ds = min(clouds_cu_bin(receiver + direction * t, direction, 0.10 + 0.028 * t, empty), interval.y - t);
        if (empty) {
            t += ds;
            continue;
        }
        float probe = t + jitter * ds;
        CuField cloud = clouds_cu_field(receiver + direction * probe, direction, lod, true);
        opticalDepth += CLOUDS_CU_EXTINCTION * (cloud.density * CU_DENSITY * ds);
        t = max(t + ds, probe + cloud.bound);
    }
}

ScatteringResult clouds_cu_airPrefix(AtmosphereParameters atmosphere, ScatteringParameters scattering,
    vec3 atmOrigin, vec3 atmOffset) {
    vec3 end = atmOrigin + atmOffset;
    if (all(equal(end, atmOrigin))) return scatteringResult_init();
    RaymarchParameters ray = RaymarchParameters(atmOrigin, end, 1u);
    return raymarchSky(atmosphere, ray, scattering, 8.0);
}

// Returns the cumulus operator C + T * background. Each occupied bin includes the air in front of
// its source. Its opacity replaces the same fraction of the already air-composited background.
ScatteringResult clouds_cu_render(AtmosphereParameters atmosphere, vec3 origin, vec3 atmOrigin,
    vec3 direction, vec2 segment, vec4 jitter, bool environment, out float meanDistance) {
    ScatteringResult result = scatteringResult_init();
    meanDistance = 0.0;
    vec2 interval;
    if (!clouds_cu_interval(origin, direction, segment.y, interval)) return result;
    interval.x = max(interval.x, segment.x);
    if (interval.x >= interval.y) return result;

    CloudRenderParams light = clouds_renderParams(direction, jitter.z);
    CloudParticpatingMedium medium = clouds_cu_medium(light.cosLightTheta);
    vec2 ambUV = cloods_amblut_uv(direction, jitter.zw);
    vec3 ambient = clouds_amblut_sample(ambUV, CLOUDS_AMBLUT_LAYER_CUMULUS);
    vec3 lightDir = rand_sampleInCone(light.lightDir, SUN_ANGULAR_RADIUS * 4.0, jitter.yw);
    vec3 atmDelta = direction * (1000.0 / float(SETTING_ATM_D_SCALE));
    vec3 airStart = atmOrigin + atmDelta * segment.x;
    ScatteringParameters scattering = scatteringParameters_init(
        lightParameters_init(atmosphere, SUN_ILLUMINANCE, uval_sunDirWorld, direction),
        lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, direction), 1.0);

    #ifdef SETTING_CLOUDS_CI
    float ciDistance = clouds_ci_distance(atmosphere, atmOrigin, normalize(atmDelta)) / length(atmDelta);
    ScatteringResult ci = scatteringResult_init();
    ScatteringResult airToCirrus = scatteringResult_init();
    if (ciDistance >= segment.x && ciDistance < interval.y) {
        vec3 offset = atmDelta * ciDistance;
        ci = clouds_ci_layer(atmosphere, atmOrigin + offset, light, ambUV);
        airToCirrus = clouds_cu_airPrefix(atmosphere, scattering, airStart, atmDelta * (ciDistance - segment.x));
    }
    #endif

    float t = interval.x;
    while (t < interval.y && result.transmittance.y > 0.01) {
        float ds = min(0.70 / float(SETTING_CLOUDS_LOW_STEP_MIN) + t * 0.009,
            CU_RANGE / float(SETTING_CLOUDS_LOW_STEP_MAX));
        if (environment) ds *= 1.6;
        bool empty;
        ds = min(clouds_cu_bin(origin + direction * t, direction, ds, empty), interval.y - t);
        if (empty) {
            t += ds;
            continue;
        }
        float probe = t + jitter.x * ds;
        float lod = smoothstep(5.0, 18.0, probe);
        vec3 pos = origin + direction * probe;
        CuField field = clouds_cu_field(pos, direction, lod, true);
        if (field.density > 0.0) {
            vec3 opticalDepth;
            vec3 isotropicMS;
            clouds_cu_lightColumn(pos, lightDir, field, light, jitter.y, lod, opticalDepth, isotropicMS);
            vec3 offset = atmDelta * probe;
            ScatteringResult bin = clouds_computeLighting(atmosphere, light, medium, ambient, atmOrigin + offset, ds,
                field.density * CU_DENSITY, field.bulk * CU_DENSITY, opticalDepth, isotropicMS);
            ScatteringResult air = clouds_cu_airPrefix(atmosphere, scattering, airStart, atmDelta * (probe - segment.x));
            #ifdef SETTING_CLOUDS_CI
            if (ciDistance >= segment.x && ciDistance < probe) {
                air.inScattering = airToCirrus.inScattering + airToCirrus.transmittance * ci.inScattering
                    + ci.transmittance * (air.inScattering - airToCirrus.inScattering);
                air.transmittance *= ci.transmittance;
            }
            #endif
            vec3 weight = result.transmittance * (1.0 - bin.transmittance);
            result.inScattering += result.transmittance * air.transmittance * bin.inScattering + weight * air.inScattering;
            meanDistance += weight.y * probe;
            result.transmittance *= bin.transmittance;
        }
        t = max(t + ds, probe + field.bound);
    }
    if (result.transmittance.y < 1.0) meanDistance /= 1.0 - result.transmittance.y;
    return result;
}

#endif
