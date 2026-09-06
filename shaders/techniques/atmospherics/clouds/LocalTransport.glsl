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

struct CTLightReservoir {
    int angle;
    vec3 pos;
    vec3 omega;
    vec3 weight;
};

void ct_updateLightReservoir(inout CTLightReservoir reservoir, int angle, vec3 pos, vec3 omega, vec3 weight, float proposal, float total, inout uint rng) {
    if (ct_random(rng) * total < proposal) reservoir = CTLightReservoir(angle, pos, omega, weight / proposal);
}

// Uniform solid-angle sampling within one cubic angular cell.
vec3 ct_sampleAngularCell(int angle, inout uint rng) {
    int axis = angle / 8;
    vec3 direction;
    float norm2;
    do {
        direction = vec3(0.0);
        direction[axis] = 1.0;
        direction[(axis + 1) % 3] = ct_random(rng);
        direction[(axis + 2) % 3] = ct_random(rng);
        norm2 = dot(direction, direction);
    } while (ct_random(rng) * norm2 * sqrt(norm2) > 1.0);
    return direction * inversesqrt(norm2) * (1.0 - 2.0 * vec3(ivec3(angle) >> ivec3(0, 1, 2) & 1));
}

vec3 ct_localSkySource(CTLightReservoir reservoir, vec2 layerRange, inout uint rng) {
    vec3 pos = reservoir.pos;
    vec3 omega = reservoir.angle < 0 ? reservoir.omega : ct_sampleAngularCell(reservoir.angle, rng);
    float shadowJitter = ct_random(rng);
    vec3 skyWeight;
    vec3 skyDir = ct_samplePhase(omega, vec2(ct_random(rng), ct_random(rng)), skyWeight);
    vec3 source = ct_boundary(pos, skyDir, layerRange, shadowJitter) * skyWeight * clouds_cu_transmittance(pos, -skyDir, layerRange);
    return reservoir.weight * source;
}

struct CTBeamReservoir {
    vec3 pos;
    vec3 direction;
    vec3 weight;
    int light;
};

vec3 ct_beamIncident(vec3 pos, int light) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 direction = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
    vec3 illuminance = light == 0 ? SUN_ILLUMINANCE : MOON_ILLUMINANCE;
    float height = length(pos);
    return ct_toAP0(illuminance * atmospherics_air_lut_sampleTransmittance(atmosphere, dot(pos / height, direction), height));
}

int ct_sampleChannelDirection(vec3 omega, inout uint rng) {
    ivec4 angles;
    vec4 weights;
    ct_angularWeights(omega, angles, weights);
    float xi = ct_random(rng);
    for (int i = 0; i < 3; ++i) {
        if (xi < weights[i]) return angles[i];
        xi -= weights[i];
    }
    return angles.w;
}

int ct_sampleReducedPhase(int angle, inout uint rng, out vec3 weight) {
    int row = angle * CT_ANGLES;
    float xi = ct_random(rng);
    int lo = 0, hi = CT_ANGLES - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (xi < ctPhase(row + mid).a) hi = mid;
        else lo = mid + 1;
    }
    float forward = ct_forwardFraction(angle);
    vec3 phase = ctPhase(row + lo).rgb - (lo == angle ? forward : 0.0);
    float probability = ctPhase(row + lo).a - (lo == 0 ? 0.0 : ctPhase(row + lo - 1).a);
    weight = phase / probability;
    return lo;
}

void ct_selectSource(vec3 pos, vec3 omega, int angle, vec3 throughput,
    inout CTLightReservoir sky, inout float skyWeight,
    inout CTBeamReservoir beam, inout float beamWeight, inout uint rng) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 sourceFloor = ct_toAP0(SUN_ILLUMINANCE + MOON_ILLUMINANCE) * UNIFORM_PHASE;
    float proposal = mmax3(throughput * sourceFloor);
    skyWeight += proposal;
    ct_updateLightReservoir(sky, angle, pos, omega, throughput, proposal, skyWeight, rng);
    // Beam and sky sources have different angular variation. Select one of
    // each, including the sharp beam phase in its proposal instead of a flat floor.
    for (int light = 0; light < 2; ++light) {
        vec3 direction = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
        if (raySphereIntersectNearest(pos, direction, vec3(0.0), atmosphere.bottom) >= 0.0) continue;
        float radius = light == 0 ? SUN_ANGULAR_RADIUS : MOON_ANGULAR_RADIUS;
        vec3 shadowDir = rand_sampleInCone(direction, radius, vec2(ct_random(rng), ct_random(rng)));
        vec3 phase = angle < 0
            ? ct_toAP0(clouds_phase_cu(clamp(dot(-omega, shadowDir), -1.0, 1.0), SETTING_CLOUDS_CU_PHASE_RATIO))
            : (light == 0 ? ctSunPhase(angle).rgb : ctMoonPhase(angle).rgb);
        vec3 weight = throughput * phase;
        float beamProposal = mmax3(weight * ct_toAP0(light == 0 ? SUN_ILLUMINANCE : MOON_ILLUMINANCE));
        beamWeight += beamProposal;
        if (ct_random(rng) * beamWeight < beamProposal) beam = CTBeamReservoir(pos, shadowDir, weight / beamProposal, light);
    }
}

vec3 ct_localRadiance(vec3 pos, vec3 omega, vec2 layerRange, inout uint rng) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    int angle = -1;
    vec3 throughput = vec3(1.0);
    vec3 result = vec3(0.0);
    float skyWeight = 0.0;
    float beamWeight = 0.0;
    CTLightReservoir sky = CTLightReservoir(-1, pos, omega, vec3(0.0));
    CTBeamReservoir beam = CTBeamReservoir(pos, omega, vec3(0.0), 0);
    for (int bounce = 0; bounce < SETTING_CLOUDS_MS_LOCAL_BOUNCES; ++bounce) {
        ivec3 base;
        vec3 f;
        if (!ct_interpolation(pos, base, f)) break;
        vec3 cellSize = mix(ct_size(base, 0), ct_size(base + 1, 0), f);
        float radius = 3.0 / dot(abs(omega), 1.0 / cellSize);
        float groundDistance = raySphereIntersectNearest(pos, -omega, vec3(0.0), atmosphere.bottom);
        bool hitsGround = groundDistance >= 0.0 && groundDistance < radius;
        if (hitsGround) radius = groundDistance;
        // Sum the common RGB within-channel mass f analytically; only the
        // nonnegative remainder P - f I needs explicit direction changes.
        float forward = angle < 0 ? 0.0 : ct_forwardFraction(angle);
        float extinction = CLOUDS_CU_EXTINCTION_COEFF - CLOUDS_CU_SCATTERING_COEFF * forward;
        float lengthScale = 0.25 / (extinction * CU_DENSITY_MAX);
        float spread = log2(1.0 + radius / lengthScale);
        bool terminal = angle >= 0 && bounce + 1 == SETTING_CLOUDS_MS_LOCAL_BOUNCES;
        float targetTau = -log(1.0 - ct_random(rng));
        float transmittance = 1.0;
        float collisionWeight = 0.0;
        vec3 collisionPos = pos;
        bool collided = false;
        for (int step = 0; step < 16; ++step) {
            float start = (exp2(float(step) * (spread / 16.0)) - 1.0) * lengthScale;
            float end = (exp2(float(step + 1) * (spread / 16.0)) - 1.0) * lengthScale;
            vec3 samplePos = pos - omega * ((start + end) * 0.5);
            float height = length(samplePos);
            float density = clouds_cu_density(samplePos, (height - layerRange.x) / (layerRange.y - layerRange.x));
            float segmentColumn = density * (end - start);
            float tau = extinction * segmentColumn;
            if (terminal) {
                if (tau == 0.0) continue;
                float opacity = tau < 0.001 ? tau * (1.0 - 0.5 * tau) : 1.0 - exp(-tau);
                float weight = transmittance * opacity;
                collisionWeight += weight;
                if (ct_random(rng) * collisionWeight < weight) collisionPos = samplePos;
                transmittance *= 1.0 - opacity;
                if (transmittance == 0.0) break;
                continue;
            }
            if (targetTau < tau) {
                // Grey extinction gives the same flight PDF in every color.
                throughput *= CLOUDS_CU_SCATTERING_COEFF / extinction;
                pos = samplePos;
                collided = true;
                break;
            }
            targetTau -= tau;
        }
        if (terminal && collisionWeight > 0.0) {
            vec3 weight = throughput * (collisionWeight * CLOUDS_CU_SCATTERING_COEFF / extinction);
            vec3 cached = ct_sampleChannel(collisionPos, angle, CT_LIFTED_OFFSET);
            vec3 convolution = ct_sampleChannel(collisionPos, angle, CT_SOURCE_OFFSET);
            result += weight * (convolution - forward * cached);
            ct_selectSource(collisionPos, omega, angle, weight, sky, skyWeight, beam, beamWeight, rng);
        }
        if (!collided) {
            if (!hitsGround) {
                vec3 endpoint = pos - omega * radius;
                vec3 cached = angle < 0 ? ct_sampleRadiance(endpoint, omega) : ct_sampleChannel(endpoint, angle, CT_LIFTED_OFFSET);
                result += throughput * transmittance * cached;
            }
            break;
        }
        ct_selectSource(pos, omega, angle, throughput, sky, skyWeight, beam, beamWeight, rng);
        vec3 phaseWeight;
        if (angle < 0) {
            // The first local event retains continuous low-order scattering.
            omega = ct_samplePhase(omega, vec2(ct_random(rng), ct_random(rng)), phaseWeight);
            throughput *= phaseWeight;
            vec3 prediction = ct_sampleRadiance(pos, omega);
            angle = ct_sampleChannelDirection(omega, rng);
            result += throughput * (prediction - ct_sampleChannel(pos, angle, CT_LIFTED_OFFSET));
        } else {
            // (P - f I) C is the exact conditional expectation of the cached
            // continuation. The scalar f <= min(P_jj) keeps every RGB weight positive.
            vec3 cached = ct_sampleChannel(pos, angle, CT_LIFTED_OFFSET);
            vec3 convolution = ct_sampleChannel(pos, angle, CT_SOURCE_OFFSET);
            result += throughput * (convolution - forward * cached);
            angle = ct_sampleReducedPhase(angle, rng, phaseWeight);
            throughput *= phaseWeight;
            result -= throughput * ct_sampleChannel(pos, angle, CT_LIFTED_OFFSET);
        }
        omega = ct_direction(angle);
        if (bounce + 1 == SETTING_CLOUDS_MS_LOCAL_BOUNCES) {
            result += throughput * ct_sampleChannel(pos, angle, CT_LIFTED_OFFSET);
            break;
        }
    }
    if (skyWeight > 0.0) result += ct_localSkySource(sky, layerRange, rng) * skyWeight;
    if (beamWeight > 0.0) {
        vec3 incident = ct_beamIncident(beam.pos, beam.light) * clouds_cu_transmittance(beam.pos, beam.direction, layerRange);
        result += beam.weight * incident * beamWeight;
    }
    return result;
}

vec3 ct_localResidual(CTReservoir reservoir, vec3 origin, vec3 direction, vec2 layerRange, vec2 xi, inout uint rng) {
    vec3 phaseWeight;
    vec3 pos = origin + direction * reservoir.distance;
    vec3 omega = ct_samplePhase(-direction, xi, phaseWeight);
    vec3 local = ct_localRadiance(pos, omega, layerRange, rng);
    // WRS subtracts the complete predictor stored at the selected camera depth.
    return reservoir.weight * (colors2_constants_toWorkSpace(phaseWeight * local) - reservoir.prediction);
}

#endif
