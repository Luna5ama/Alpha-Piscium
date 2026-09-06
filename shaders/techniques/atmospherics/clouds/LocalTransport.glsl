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

vec3 ct_localSkyResidual(CTLightReservoir reservoir, vec2 layerRange, inout uint rng) {
    vec3 pos = reservoir.pos;
    vec3 omega = reservoir.omega;
    float shadowJitter = ct_random(rng);
    vec3 skyWeight;
    vec3 skyDir = ct_samplePhase(omega, vec2(ct_random(rng), ct_random(rng)), skyWeight);
    vec3 source = ct_boundary(pos, skyDir, layerRange, shadowJitter) * skyWeight * clouds_cu_transmittance(pos, -skyDir, layerRange, CT_EXTINCTION);
    return reservoir.weight * (source - reservoir.prediction);
}

struct CTBeamReservoir {
    vec3 pos;
    vec3 direction;
    vec3 weight;
    vec3 prediction;
    int light;
};

vec3 ct_beamIncident(vec3 pos, int light) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 direction = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
    vec3 illuminance = light == 0 ? SUN_ILLUMINANCE : MOON_ILLUMINANCE;
    float height = length(pos);
    return ct_toAP0(illuminance * atmospherics_air_lut_sampleTransmittance(atmosphere, dot(pos / height, direction), height));
}

struct CTBoundaryClosure {
    vec3 center;
    vec3 normal;
    vec3 mean;
    vec3 asymmetry;
    float radius;
};

vec3 ct_boundaryIncident(vec3 pos, vec3 normal, vec2 layerRange) {
    ivec3 base;
    vec3 f;
    if (!ct_interpolation(pos, base, f)) return vec3(0.0);
    vec3 cellSize = mix(ct_size(base, 0), ct_size(base + 1, 0), f);
    float radius = 3.0 / dot(abs(normal), 1.0 / cellSize);
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    float ground = raySphereIntersectNearest(pos, normal, vec3(0.0), atmosphere.bottom);
    bool hitsGround = ground >= 0.0 && ground < radius;
    if (hitsGround) radius = ground;
    float lengthScale = 0.25 / (mmax3(CT_EXTINCTION) * CU_DENSITY_MAX);
    float spread = log2(1.0 + radius / lengthScale);
    ivec4 angles;
    vec4 weights;
    ct_angularWeights(-normal, angles, weights);
    vec3 sunPhase = ct_toAP0(clouds_phase_cu(clamp(dot(normal, uval_sunDirWorld), -1.0, 1.0), SETTING_CLOUDS_CU_PHASE_RATIO));
    vec3 moonPhase = ct_toAP0(clouds_phase_cu(clamp(dot(normal, uval_moonDirWorld), -1.0, 1.0), SETTING_CLOUDS_CU_PHASE_RATIO));
    vec3 tr = vec3(1.0);
    vec3 result = vec3(0.0);
    for (int step = 0; step < 12; ++step) {
        float start = (exp2(float(step) * (spread / 12.0)) - 1.0) * lengthScale;
        float end = (exp2(float(step + 1) * (spread / 12.0)) - 1.0) * lengthScale;
        vec3 point = pos + normal * ((start + end) * 0.5);
        float rho = clouds_cu_density(point, (length(point) - layerRange.x) / (layerRange.y - layerRange.x));
        if (rho == 0.0) continue;
        vec3 sky, sun, moon;
        ct_cachedExternal(point, -normal, sky, sun, moon);
        vec3 source = (ct_sampleField(point, angles, weights, CT_SOURCE_OFFSET) + sky + sun * sunPhase + moon * moonPhase) * (CT_SCATTERING / CT_EXTINCTION);
        vec3 tau = CT_EXTINCTION * rho * (end - start);
        vec3 segmentTr = exp(-tau);
        vec3 opacity = mix(1.0 - segmentTr, tau * (1.0 - 0.5 * tau), lessThan(tau, vec3(0.001)));
        result += tr * opacity * source;
        tr *= segmentTr;
        if (mmax3(tr) < 0.001) return result + tr * source;
    }
    if (!hitsGround) result += tr * ct_sampleRadiance(pos + normal * radius, -normal);
    return result;
}

CTBoundaryClosure ct_boundaryClosure(vec3 pos, vec2 layerRange) {
    CTBoundaryClosure c = CTBoundaryClosure(pos, vec3(0.0), vec3(0.0), vec3(0.0), 0.0);
    float rho = clouds_cu_density(pos, (length(pos) - layerRange.x) / (layerRange.y - layerRange.x));
    if (rho == 0.0) return c;
    vec3 beta1 = mix(texelFetch(usam_cloudTransport, ivec2(0, 4), 0).rgb, texelFetch(usam_cloudTransport, ivec2(3, 4), 0).rgb, SETTING_CLOUDS_CU_PHASE_RATIO);
    float transportExtinction = rho * dot(CT_EXTINCTION - CT_SCATTERING * beta1, vec3(1.0 / 3.0));
    float radius = 2.0 / (3.0 * transportExtinction);
    // A diffusion boundary closure applies only to optically thick details.
    if (radius > 1.0 / (3.7 * CU_DETAIL_FREQUENCY)) return c;
    float stepLength = 1.0 / (16.0 * CU_DETAIL_FREQUENCY);
    vec3 gradient = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        vec3 offset = vec3(1 - 2 * (i & 1), 1 - 2 * (i >> 1), 1 - 2 * ((i ^ (i >> 1)) & 1));
        vec3 point = pos + offset * (stepLength * 0.25);
        // The unclamped interface keeps a usable normal inside the opaque
        // surface ramp, where a density gradient would already be zero.
        float shape = clouds_cu_interface(point, (length(point) - layerRange.x) / (layerRange.y - layerRange.x));
        gradient += offset * shape;
    }
    if (dot(gradient, gradient) == 0.0) return c;
    vec3 normal = -normalize(gradient);
    float insideDistance = 0.0;
    float outsideDistance = 0.0;
    for (int i = 1; i <= 4; ++i) {
        float distance = float(i) * stepLength;
        vec3 point = pos + normal * distance;
        if (clouds_cu_density(point, (length(point) - layerRange.x) / (layerRange.y - layerRange.x)) == 0.0) {
            outsideDistance = distance;
            break;
        }
        insideDistance = distance;
    }
    if (outsideDistance == 0.0) return c;
    for (int i = 0; i < 4; ++i) {
        float distance = (insideDistance + outsideDistance) * 0.5;
        vec3 point = pos + normal * distance;
        if (clouds_cu_density(point, (length(point) - layerRange.x) / (layerRange.y - layerRange.x)) == 0.0) outsideDistance = distance;
        else insideDistance = distance;
    }
    c.center = pos + normal * outsideDistance;
    c.normal = normal;
    c.radius = radius;
    vec3 interior = ct_sampleMean(c.center - normal * radius);
    vec3 incident = ct_boundaryIncident(c.center, normal, layerRange);
    // Marshak matching at one extrapolation length: Phi_0=(Phi_in+Phi_bulk)/2,
    // J_n=(Phi_bulk-Phi_in)/4. A normalized positive HG field carries both moments.
    c.mean = (interior + incident) * 0.5;
    for (int channel = 0; channel < 3; ++channel) {
        if (c.mean[channel] > 0.0) c.asymmetry[channel] = (interior[channel] - incident[channel]) / (4.0 * c.mean[channel]);
    }
    return c;
}

vec3 ct_boundaryRadiance(CTBoundaryClosure c, vec3 pos, vec3 omega) {
    vec3 coarse = ct_sampleRadiance(pos, omega);
    if (c.radius == 0.0) return coarse;
    float distance = dot(c.center - pos, c.normal);
    if (distance < 0.0 || length(pos - c.center) > c.radius) return coarse;
    vec3 boundary = c.mean * SPHERE_SOLID_ANGLE * phasefunc_HenyeyGreenstein(dot(omega, c.normal), c.asymmetry);
    return mix(coarse, boundary, exp(-distance / c.radius));
}

vec3 ct_localRadiance(vec3 pos, vec3 omega, vec2 layerRange, inout uint rng) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 boundaryOrigin = pos;
    float referenceExtinction = dot(CT_EXTINCTION, vec3(1.0 / 3.0));
    float lengthScale = 0.25 / (referenceExtinction * CU_DENSITY_MAX);
    vec3 throughput = vec3(1.0);
    vec3 result = vec3(0.0);
    bool reachedDepth = false;
    vec3 sourceFloor = ct_toAP0(SUN_ILLUMINANCE + MOON_ILLUMINANCE) * UNIFORM_PHASE;
    float skyWeight = 0.0;
    float beamWeight = 0.0;
    CTLightReservoir sky = CTLightReservoir(pos, omega, vec3(0.0), vec3(0.0));
    CTBeamReservoir beam = CTBeamReservoir(pos, omega, vec3(0.0), vec3(0.0), 0);
    for (int bounce = 0; bounce < SETTING_CLOUDS_MS_LOCAL_BOUNCES; ++bounce) {
        ivec3 base;
        vec3 f;
        if (!ct_interpolation(pos, base, f)) break;
        vec3 cellSize = mix(ct_size(base, 0), ct_size(base + 1, 0), f);
        float radius = 3.0 / dot(abs(omega), 1.0 / cellSize);
        float groundDistance = raySphereIntersectNearest(pos, -omega, vec3(0.0), atmosphere.bottom);
        bool hitsGround = groundDistance >= 0.0 && groundDistance < radius;
        if (hitsGround) radius = groundDistance;
        float spread = log2(1.0 + radius / lengthScale);
        float targetTau = -log(1.0 - ct_random(rng));
        float column = 0.0;
        bool collided = false;
        for (int step = 0; step < 16; ++step) {
            float start = (exp2(float(step) * (spread / 16.0)) - 1.0) * lengthScale;
            float end = (exp2(float(step + 1) * (spread / 16.0)) - 1.0) * lengthScale;
            vec3 samplePos = pos - omega * ((start + end) * 0.5);
            float height = length(samplePos);
            float density = clouds_cu_density(samplePos, (height - layerRange.x) / (layerRange.y - layerRange.x));
            float segmentColumn = density * (end - start);
            float tau = referenceExtinction * segmentColumn;
            if (targetTau < tau) {
                // Sample the analytically integrated segment source. Spectral
                // opacity ratios preserve the same piecewise-constant equation.
                vec3 opticalDepth = CT_EXTINCTION * segmentColumn;
                vec3 opacity = mix(1.0 - exp(-opticalDepth), opticalDepth * (1.0 - 0.5 * opticalDepth), lessThan(opticalDepth, vec3(0.001)));
                float referenceOpacity = tau < 0.001 ? tau * (1.0 - 0.5 * tau) : 1.0 - exp(-tau);
                throughput *= exp((referenceExtinction - CT_EXTINCTION) * column) * (CT_SCATTERING / CT_EXTINCTION) * (opacity / referenceOpacity);
                pos = samplePos;
                collided = true;
                break;
            }
            targetTau -= tau;
            column += segmentColumn;
        }
        if (!collided) {
            if (!hitsGround) result += throughput * exp((referenceExtinction - CT_EXTINCTION) * column) * ct_sampleRadiance(pos - omega * radius, omega);
            break;
        }
        vec3 cachedSky, cachedSun, cachedMoon;
        ct_cachedExternal(pos, omega, cachedSky, cachedSun, cachedMoon);
        result += throughput * cachedSky;
        float proposal = mmax3(throughput * sourceFloor);
        skyWeight += proposal;
        ct_updateLightReservoir(sky, pos, omega, throughput, cachedSky, proposal, skyWeight, rng);
        // Beam and sky sources have different angular variation. Select one of
        // each, including the sharp beam phase in its proposal instead of a flat floor.
        for (int light = 0; light < 2; ++light) {
            vec3 direction = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
            if (raySphereIntersectNearest(pos, direction, vec3(0.0), atmosphere.bottom) >= 0.0) continue;
            float radius = light == 0 ? SUN_ANGULAR_RADIUS : MOON_ANGULAR_RADIUS;
            vec3 shadowDir = rand_sampleInCone(direction, radius, vec2(ct_random(rng), ct_random(rng)));
            vec3 phase = ct_toAP0(clouds_phase_cu(clamp(dot(-omega, shadowDir), -1.0, 1.0), SETTING_CLOUDS_CU_PHASE_RATIO));
            vec3 cached = light == 0 ? cachedSun : cachedMoon;
            vec3 weight = throughput * phase;
            result += weight * cached;
            float beamProposal = mmax3(weight * (cached + ct_beamIncident(pos, light)));
            beamWeight += beamProposal;
            if (ct_random(rng) * beamWeight < beamProposal) beam = CTBeamReservoir(pos, shadowDir, weight / beamProposal, cached, light);
        }
        vec3 phaseWeight;
        omega = ct_sampleLightDirection(pos, omega, 0.0625 / float(bounce + 1), rng, phaseWeight);
        throughput *= phaseWeight;
        if (bounce + 1 == SETTING_CLOUDS_MS_LOCAL_BOUNCES) {
            reachedDepth = true;
            break;
        }
    }
    if (skyWeight > 0.0) result += ct_localSkyResidual(sky, layerRange, rng) * skyWeight;
    if (beamWeight > 0.0) {
        vec3 incident = ct_beamIncident(beam.pos, beam.light) * clouds_cu_transmittance(beam.pos, beam.direction, layerRange, CT_EXTINCTION);
        result += beam.weight * (incident - beam.prediction) * beamWeight;
    }
    if (reachedDepth) {
        CTBoundaryClosure boundary = ct_boundaryClosure(boundaryOrigin, layerRange);
        result += throughput * ct_boundaryRadiance(boundary, pos, omega);
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
