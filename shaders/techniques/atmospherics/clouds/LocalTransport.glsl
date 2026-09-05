#ifndef INCLUDE_clouds_LocalTransport_glsl
#define INCLUDE_clouds_LocalTransport_glsl a

#include "Transport.glsl"
#include "/util/Celestial.glsl"

float ct_random(inout uint rng) {
    return hash_uintToFloat(hash_pcg_11(rng++));
}

struct CTReservoir {
    float distance;
    vec3 weight;
};

void ct_updateReservoir(inout CTReservoir reservoir, float distance, vec3 weight, float probabilityWeight, float total, inout uint rng) {
    if (ct_random(rng) * total < probabilityWeight) {
        reservoir.distance = distance;
        reservoir.weight = weight / probabilityWeight;
    }
}

// Sample the exact interpolated RGB matrix row, not the continuous phase.
// WRS normalizes the scalar proposal even with FP32 row-sum roundoff.
int ct_samplePhase(ivec4 angles, vec4 weights, inout uint rng, out vec3 phaseWeight) {
    float total = 0.0;
    int selected = 0;
    for (int incoming = 0; incoming < CT_ANGLES; ++incoming) {
        vec3 phase = vec3(0.0);
        for (int a = 0; a < 4; ++a) phase += weights[a] * ctPhase(angles[a] * CT_ANGLES + incoming).rgb;
        float proposal = dot(phase, vec3(1.0 / 3.0));
        total += proposal;
        if (ct_random(rng) * total < proposal) {
            selected = incoming;
            phaseWeight = phase / proposal;
        }
    }
    phaseWeight *= total;
    return selected;
}

vec3 ct_localRadiance(vec3 pos, int angle, vec2 layerRange, inout uint rng) {
    ivec3 base;
    vec3 f;
    if (!ct_interpolation(pos, base, f)) return vec3(0.0);
    vec3 omega = ct_direction(angle);
    // One upstream cell flight length. Exponential bins resolve the near field.
    vec3 cellSize = mix(ct_size(base, 0), ct_size(base + 1, 0), f);
    float radius = 1.0 / dot(abs(omega), 1.0 / cellSize);
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    float groundDistance = raySphereIntersectNearest(pos, -omega, vec3(0.0), atmosphere.bottom);
    bool hitsGround = groundDistance >= 0.0 && groundDistance < radius;
    if (hitsGround) radius = groundDistance;
    vec3 skyWeight;
    int skyAngle = ct_samplePhase(ivec4(angle), vec4(1.0, 0.0, 0.0, 0.0), rng, skyWeight);
    vec3 skyIncident = ct_boundary(skyAngle) * skyWeight;
    vec3 transmittance = vec3(1.0);
    vec3 result = vec3(0.0);
    float jitter = ct_random(rng);
    float shadowJitter = ct_random(rng);
    for (int step = 0; step < SETTING_CLOUDS_MS_LOCAL_STEPS; ++step) {
        float start = (exp2(float(step) * (4.0 / SETTING_CLOUDS_MS_LOCAL_STEPS)) - 1.0) * (radius / 15.0);
        float end = (exp2(float(step + 1) * (4.0 / SETTING_CLOUDS_MS_LOCAL_STEPS)) - 1.0) * (radius / 15.0);
        vec3 samplePos = pos - omega * mix(start, end, jitter);
        float height = length(samplePos);
        float density = clouds_cu_density(samplePos, (height - layerRange.x) / (layerRange.y - layerRange.x));
        if (density == 0.0) continue;
        vec3 source = ct_sampleDirection(samplePos, angle, CT_SOURCE_OFFSET);
        // Reuse one sky direction along this local ray; its expectation is P B_h.
        source += skyIncident * exp(-CT_EXTINCTION * clouds_cu_column(samplePos, -ct_direction(skyAngle), layerRange, 12, 6.0, shadowJitter));
        for (int light = 0; light < 2; ++light) {
            vec3 lightDir = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
            if (raySphereIntersectNearest(samplePos, lightDir, vec3(0.0), atmosphere.bottom) >= 0.0) continue;
            vec3 illuminance = light == 0 ? SUN_ILLUMINANCE : MOON_ILLUMINANCE;
            vec3 phase = light == 0 ? ctSunPhase(angle).rgb : ctMoonPhase(angle).rgb;
            vec3 airTr = atmospherics_air_lut_sampleTransmittance(atmosphere, dot(samplePos / height, lightDir), height);
            vec3 shadow = exp(-CT_EXTINCTION * clouds_cu_column(samplePos, lightDir, layerRange, 16, 6.0, shadowJitter));
            source += ct_toAP0(illuminance * airTr) * phase * shadow;
        }
        vec3 tau = CT_EXTINCTION * density * (end - start);
        vec3 segmentTr = exp(-tau);
        vec3 opacity = mix(1.0 - segmentTr, tau * (1.0 - 0.5 * tau + tau * tau / 6.0), lessThan(tau, vec3(0.001)));
        result += transmittance * opacity * (CT_SCATTERING / CT_EXTINCTION) * source;
        transmittance *= segmentTr;
    }
    if (!hitsGround) result += transmittance * ct_sampleDirection(pos - omega * radius, angle, CT_RADIANCE_OFFSET);
    return result;
}

vec3 ct_localResidual(CTReservoir reservoir, vec3 origin, vec3 direction, ivec4 angles, vec4 weights, vec2 layerRange, inout uint rng) {
    vec3 phaseWeight;
    int angle = ct_samplePhase(angles, weights, rng, phaseWeight);
    vec3 pos = origin + direction * reservoir.distance;
    vec3 predictor = ct_sampleDirection(pos, angle, CT_RADIANCE_OFFSET);
    vec3 local = ct_localRadiance(pos, angle, layerRange, rng);
    return reservoir.weight * colors2_constants_toWorkSpace(phaseWeight * (local - predictor));
}

#endif
