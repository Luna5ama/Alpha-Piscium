#ifndef INCLUDE_clouds_LocalTransport_glsl
#define INCLUDE_clouds_LocalTransport_glsl a

#include "Transport.glsl"
#include "/util/Celestial.glsl"
#include "/util/Rand.glsl"

float ct_random(inout uint rng) {
    return hash_uintToFloat(hash_pcg_11(rng++));
}

struct CTReservoir {
    float distance;
    vec3 weight;
    vec3 prediction;
};

void ct_updateReservoir(inout CTReservoir reservoir, float distance, vec3 weight, vec3 prediction, float probabilityWeight, float total, inout uint rng) {
    if (ct_random(rng) * total < probabilityWeight) {
        reservoir.distance = distance;
        reservoir.weight = weight / probabilityWeight;
        reservoir.prediction = prediction;
    }
}

// Sample the continuous RGB Mie/HG+Draine mixture, preserving the forward lobe.
vec3 ct_samplePhase(vec3 outgoing, vec2 xi, out vec3 phaseWeight) {
    float channelXi = xi.x * 3.0;
    vec3 cosines = ct_phaseCosines(fract(channelXi));
    float mu = cosines[int(channelXi)];
    vec3 tangent = normalize(cross(outgoing, abs(outgoing.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(outgoing, tangent);
    vec3 incoming = outgoing * mu + (tangent * cos(2.0 * PI * xi.y) + bitangent * sin(2.0 * PI * xi.y)) * sqrt(1.0 - mu * mu);
    vec3 phase = ct_toAP0(clouds_phase_cu(mu, SETTING_CLOUDS_CU_PHASE_RATIO));
    phaseWeight = phase / dot(phase, vec3(1.0 / 3.0));
    return incoming;
}

// Reduce the beam proposal with path depth so continuation likelihoods do not grow exponentially.
vec3 ct_sampleLightDirection(vec3 pos, vec3 outgoing, float beamProbability, inout uint rng, out vec3 phaseWeight) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 lightDir = uval_sunDirWorld;
    if (raySphereIntersectNearest(pos, lightDir, vec3(0.0), atmosphere.bottom) >= 0.0) lightDir = uval_moonDirWorld;
    bool visible = raySphereIntersectNearest(pos, lightDir, vec3(0.0), atmosphere.bottom) < 0.0;
    vec3 axis = visible && ct_random(rng) < beamProbability ? -lightDir : outgoing;
    vec3 incoming = ct_samplePhase(axis, vec2(ct_random(rng), ct_random(rng)), phaseWeight);
    if (visible) {
        vec3 phase = ct_toAP0(clouds_phase_cu(clamp(dot(outgoing, incoming), -1.0, 1.0), SETTING_CLOUDS_CU_PHASE_RATIO));
        vec3 lightPhase = ct_toAP0(clouds_phase_cu(clamp(dot(-lightDir, incoming), -1.0, 1.0), SETTING_CLOUDS_CU_PHASE_RATIO));
        phaseWeight = phase / dot(mix(phase, lightPhase, beamProbability), vec3(1.0 / 3.0));
    }
    return incoming;
}

struct CTLightReservoir {
    vec3 pos;
    vec3 omega;
    vec3 weight;
    vec3 prediction;
};

void ct_updateLightReservoir(inout CTLightReservoir reservoir, vec3 pos, vec3 omega, vec3 weight, vec3 prediction, float proposal, float total, inout uint rng) {
    if (ct_random(rng) * total < proposal) reservoir = CTLightReservoir(pos, omega, weight / proposal, prediction);
}

vec3 ct_localSource(CTLightReservoir reservoir, vec2 layerRange, inout uint rng) {
    vec3 pos = reservoir.pos;
    vec3 omega = reservoir.omega;
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    float height = length(pos);
    float shadowJitter = ct_random(rng);
    vec3 skyWeight;
    vec3 skyDir = ct_samplePhase(omega, vec2(ct_random(rng), ct_random(rng)), skyWeight);
    vec3 source = ct_boundary(pos, skyDir, layerRange, shadowJitter) * skyWeight * clouds_cu_transmittance(pos, -skyDir, layerRange, CT_EXTINCTION, rng);
    for (int light = 0; light < 2; ++light) {
        vec3 lightDir = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
        if (raySphereIntersectNearest(pos, lightDir, vec3(0.0), atmosphere.bottom) >= 0.0) continue;
        vec3 illuminance = light == 0 ? SUN_ILLUMINANCE : MOON_ILLUMINANCE;
        float radius = light == 0 ? SUN_ANGULAR_RADIUS : MOON_ANGULAR_RADIUS;
        vec3 shadowDir = rand_sampleInCone(lightDir, radius, vec2(ct_random(rng), ct_random(rng)));
        vec3 phase = ct_toAP0(clouds_phase_cu(clamp(dot(-omega, shadowDir), -1.0, 1.0), SETTING_CLOUDS_CU_PHASE_RATIO));
        vec3 airTr = atmospherics_air_lut_sampleTransmittance(atmosphere, dot(pos / height, lightDir), height);
        vec3 shadow = clouds_cu_transmittance(pos, shadowDir, layerRange, CT_EXTINCTION, rng);
        source += ct_toAP0(illuminance * airTr) * phase * shadow;
    }
    return reservoir.weight * (source - reservoir.prediction);
}

vec3 ct_localRadiance(vec3 pos, vec3 omega, vec2 layerRange, inout uint rng) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    float referenceExtinction = dot(CT_EXTINCTION, vec3(1.0 / 3.0));
    float majorant = 1.5 * mmax3(CT_EXTINCTION) * CU_DENSITY_MAX;
    vec3 throughput = vec3(1.0);
    vec3 result = vec3(0.0);
    vec3 sourceFloor = ct_toAP0(SUN_ILLUMINANCE + MOON_ILLUMINANCE) * UNIFORM_PHASE;
    float lightingWeight = 0.0;
    CTLightReservoir light0 = CTLightReservoir(pos, omega, vec3(0.0), vec3(0.0));
    CTLightReservoir light1 = light0;
    for (int bounce = 0; bounce < SETTING_CLOUDS_MS_LOCAL_BOUNCES; ++bounce) {
        ivec3 base;
        vec3 f;
        if (!ct_interpolation(pos, base, f)) break;
        vec3 cellSize = mix(ct_size(base, 0), ct_size(base + 1, 0), f);
        float radius = 1.0 / dot(abs(omega), 1.0 / cellSize);
        float groundDistance = raySphereIntersectNearest(pos, -omega, vec3(0.0), atmosphere.bottom);
        bool hitsGround = groundDistance >= 0.0 && groundDistance < radius;
        if (hitsGround) radius = groundDistance;
        float distance = 0.0;
        bool escaped = false;
        // A padded majorant keeps RGB null weights away from zero. A tight
        // majorant and least-extinct real events cause rare, huge blue path weights.
        while (true) {
            distance -= log(1.0 - ct_random(rng)) / majorant;
            if (distance >= radius) {
                if (!hitsGround) result += throughput * ct_sampleRadiance(pos - omega * radius, omega);
                escaped = true;
                break;
            }
            vec3 samplePos = pos - omega * distance;
            float height = length(samplePos);
            float density = clouds_cu_density(samplePos, (height - layerRange.x) / (layerRange.y - layerRange.x));
            float realRate = density * referenceExtinction;
            if (ct_random(rng) * majorant < realRate) {
                pos = samplePos;
                throughput *= CT_SCATTERING / referenceExtinction;
                break;
            }
            throughput *= (majorant - density * CT_EXTINCTION) / (majorant - realRate);
        }
        if (escaped) break;
        vec3 prediction = ct_sampleExternal(pos, omega);
        result += throughput * prediction;
        // A positive isotropic beam floor keeps unresolved coarse cells sampleable.
        // This proposal changes variance only; the entire source residual is retained.
        float proposal = mmax3(throughput * (prediction + sourceFloor));
        lightingWeight += proposal;
        ct_updateLightReservoir(light0, pos, omega, throughput, prediction, proposal, lightingWeight, rng);
        ct_updateLightReservoir(light1, pos, omega, throughput, prediction, proposal, lightingWeight, rng);
        vec3 phaseWeight;
        omega = ct_sampleLightDirection(pos, omega, 0.0625 / float(bounce + 1), rng, phaseWeight);
        throughput *= phaseWeight;
        if (bounce + 1 == SETTING_CLOUDS_MS_LOCAL_BOUNCES) {
            result += throughput * ct_sampleRadiance(pos, omega);
            break;
        }
    }
    if (lightingWeight > 0.0) {
        vec3 residual = ct_localSource(light0, layerRange, rng) + ct_localSource(light1, layerRange, rng);
        result += residual * (lightingWeight * 0.5);
    }
    return result;
}

vec3 ct_localResidual(CTReservoir reservoir, vec3 origin, vec3 direction, vec2 layerRange, inout uint rng) {
    vec3 phaseWeight;
    vec3 pos = origin + direction * reservoir.distance;
    vec3 omega = ct_sampleLightDirection(pos, -direction, 0.0625, rng, phaseWeight);
    vec3 local = ct_localRadiance(pos, omega, layerRange, rng);
    // Subtract the exact camera predictor stored by WRS. Its angular matrix need
    // not match the continuous local model: the full predictor cancels in expectation.
    return reservoir.weight * (colors2_constants_toWorkSpace(phaseWeight * local) - reservoir.prediction);
}

#endif
