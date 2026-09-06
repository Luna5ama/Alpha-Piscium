#extension GL_KHR_shader_subgroup_basic : enable

#include "LocalTransport.glsl"
#include "./ss/Common.glsl"
#include "/util/Celestial.glsl"
#include "/util/Morton.glsl"
#include "/techniques/HiZ.glsl"

/*const*/
#define ATMOSPHERE_RAYMARCHING_SKY a
/*const*/
#include "/techniques/atmospherics/air/Raymarching.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(RENDER_MULTIPLIER, RENDER_MULTIPLIER);

shared bool shared_worldGroupCheck;
layout(rgba32ui) uniform writeonly uimage2D uimg_rgba32ui;

vec3 singleScatteringAtPoint(vec3 pos, vec3 direction, vec2 layerRange, inout uint rng) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    float height = length(pos);
    vec2 ambientJitter = vec2(ct_random(rng), ct_random(rng));
    float lightJitter = ct_random(rng);
    vec3 phaseWeight;
    vec3 incoming = ct_samplePhase(-direction, ambientJitter, phaseWeight);
    float skyTransmittance = clouds_cu_transmittance(pos, -incoming, layerRange);
    vec3 sky = ct_boundary(pos, incoming, layerRange, lightJitter) * phaseWeight * skyTransmittance;
    vec3 source = colors2_constants_toWorkSpace(sky);
    for (int light = 0; light < 2; ++light) {
        vec3 lightDir = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
        if (raySphereIntersectNearest(pos, lightDir, vec3(0.0), atmosphere.bottom) >= 0.0) continue;
        vec3 illuminance = light == 0 ? SUN_ILLUMINANCE : MOON_ILLUMINANCE;
        float radius = light == 0 ? SUN_ANGULAR_RADIUS : MOON_ANGULAR_RADIUS;
        vec3 shadowDir = rand_sampleInCone(lightDir, radius, ambientJitter);
        float shadow = clouds_cu_transmittance(pos, shadowDir, layerRange);
        vec3 phase = clouds_phase_cu(clamp(dot(direction, shadowDir), -1.0, 1.0), SETTING_CLOUDS_CU_PHASE_RATIO);
        vec3 atmosphereTr = atmospherics_air_lut_sampleTransmittance(atmosphere, dot(pos / height, lightDir), height);
        source += illuminance * atmosphereTr * phase * shadow;
    }
    return source;
}

vec3 shadeCloudSample(float distance, vec3 origin, vec3 direction, vec2 layerRange, float directWeight, float multipleWeight, vec2 xi, inout uint rng) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    LightParameters sunParams = lightParameters_init(atmosphere, SUN_ILLUMINANCE, uval_sunDirWorld, direction);
    LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, direction);
    ScatteringParameters airScattering = scatteringParameters_init(sunParams, moonParams, 1.0);
    RaymarchParameters airParams = raymarchParameters_init();
    airParams.rayStart = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    airParams.rayEnd = airParams.rayStart + direction * distance;
    airParams.steps = uint(clamp(distance * 2.0, 4.0, 32.0));
    ScatteringResult air = raymarchSky(atmosphere, airParams, airScattering, 8.0);
    vec3 pos = origin + direction * distance;
    const float albedo = CLOUDS_CU_SCATTERING_COEFF / CLOUDS_CU_EXTINCTION_COEFF;
    vec3 result = directWeight * (air.inScattering + air.transmittance * albedo * singleScatteringAtPoint(pos, direction, layerRange, rng));
    #if SETTING_CLOUDS_MS_LOCAL_SAMPLES > 0
    if (multipleWeight > 0.0) {
        vec3 phaseWeight;
        vec3 omega = ct_samplePhase(-direction, xi, phaseWeight);
        vec3 light = ct_localRadiance(pos, omega, layerRange, rng);
        result += multipleWeight * air.transmittance * albedo * colors2_constants_toWorkSpace(phaseWeight * light);
    }
    #else
    ivec4 angles;
    vec4 angularWeights;
    ct_angularWeights(-direction, angles, angularWeights);
    vec3 source = colors2_constants_toWorkSpace(ct_sampleField(pos, angles, angularWeights, CT_SOURCE_OFFSET));
    result += directWeight * air.transmittance * albedo * source;
    #endif
    return result;
}

void render(ivec2 texelPos) {
    CloudSSHistoryData result = clouds_ss_historyData_init();
    result.hLen = 1.0;
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec2 screenPos = clouds_ss_upscaledTexelCenter(texelPos) * uval_mainImageSizeRcp;
    vec3 viewPos = coords_toViewCoord(screenPos, -65536.0, global_camProjInverse);
    vec3 direction = normalize(mat3(gbufferModelViewInverse) * viewPos);
    vec3 origin = atmosphere_viewToAtmNoClamping(atmosphere, vec3(0.0));
    vec2 layerRange = atmosphere.bottom + vec2(SETTING_CLOUDS_CU_HEIGHT, SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS);
    vec2 interval;
    bool intersects = clouds_layerInterval(origin, direction, layerRange, interval);
    float groundDistance = raySphereIntersectNearest(origin, direction, vec3(0.0), min(atmosphere.bottom, length(origin) - 0.001));
    if (groundDistance >= 0.0) interval.y = min(interval.y, groundDistance);
    interval.y = min(interval.y, min(interval.x + 60.0, 120.0));

    if (intersects && interval.y > interval.x && CU_DENSITY_SCALE > 0.0 && SETTING_CLOUDS_CU_COVERAGE > 0.0) {
        float cameraTransmittance = 1.0;
        float jitter = rand_stbnVec1(rand_newStbnPos(texelPos, 1), frameCounter);
        float rayLength = interval.y - interval.x;
        float opticalStep = 0.5 / (CU_DENSITY_MAX * CLOUDS_CU_EXTINCTION_COEFF);
        int steps = int(clamp(rayLength * 3.0, float(SETTING_CLOUDS_LOW_STEP_MIN), float(max(SETTING_CLOUDS_LOW_STEP_MIN, SETTING_CLOUDS_LOW_STEP_MAX))));

        float emptyDistance = interval.x;
        float integratedEnd = interval.x;
        bool inside = interval.x == 0.0 && clouds_cu_density(origin, (length(origin) - layerRange.x) / (layerRange.y - layerRange.x)) > 0.0;
        float depthWeight = 0.0;
        uint rng = hash_pcg3d_33(uvec3(texelPos, frameCounter)).x;
        float lightingWeight = 0.0;
        float reservoir0 = 0.0;
        float reservoir1 = 0.0;
        #if SETTING_CLOUDS_MS_LOCAL_SAMPLES == 4
        float reservoir2 = 0.0;
        float reservoir3 = 0.0;
        #endif

        for (int i = 0; i < steps; ++i) {
            float u0 = float(i) / float(steps);
            float u1 = float(i + 1) / float(steps);
            float start = u0 * (0.4 + 0.6 * u0) * rayLength;
            float end = u1 * (0.4 + 0.6 * u1) * rayLength;
            float distance = interval.x + mix(start, end, jitter);
            vec3 pos = origin + direction * distance;
            float height = length(pos);
            float heightFraction = (height - layerRange.x) / (layerRange.y - layerRange.x);
            float density = clouds_cu_density(pos, heightFraction);
            if (density <= 0.0) {
                emptyDistance = distance;
                inside = false;
                continue;
            }
            float occupiedStart = interval.x + start;
            if (!inside) {
                // Locate the interface before integrating the optically thin surface segments.
                float occupiedDistance = distance;
                for (int refine = 0; refine < 8; ++refine) {
                    float midpoint = (emptyDistance + occupiedDistance) * 0.5;
                    vec3 midpointPos = origin + direction * midpoint;
                    float midpointHeight = (length(midpointPos) - layerRange.x) / (layerRange.y - layerRange.x);
                    if (clouds_cu_density(midpointPos, midpointHeight) > 0.0) occupiedDistance = midpoint;
                    else emptyDistance = midpoint;
                }
                occupiedStart = max(occupiedDistance, integratedEnd);
            }
            inside = true;

            float occupiedEnd = interval.x + end;
            for (float segmentStart = occupiedStart; segmentStart < occupiedEnd; segmentStart += opticalStep) {
                float segmentEnd = min(segmentStart + opticalStep, occupiedEnd);
                integratedEnd = segmentEnd;
                distance = mix(segmentStart, segmentEnd, jitter);
                pos = origin + direction * distance;
                height = length(pos);
                heightFraction = (height - layerRange.x) / (layerRange.y - layerRange.x);
                density = clouds_cu_density(pos, heightFraction);
                if (density <= 0.0) {
                    emptyDistance = distance;
                    inside = false;
                    continue;
                }
                inside = true;

                float tau = CLOUDS_CU_EXTINCTION_COEFF * density * (segmentEnd - segmentStart);
                float transmittance = exp(-tau);
                float opacity = tau < 0.001 ? tau * (1.0 - 0.5 * tau + tau * tau / 6.0) : 1.0 - transmittance;
                float cameraOpacity = cameraTransmittance * opacity;
                cameraTransmittance *= transmittance;
                bool opaque = cameraTransmittance < 0.005;
                if (opaque) {
                    // Complete the opaque tail with the local source instead of leaking an arbitrary fraction of the sky.
                    cameraOpacity += cameraTransmittance;
                    cameraTransmittance = 0.0;
                }
                result.distance += cameraOpacity * distance;
                depthWeight += cameraOpacity;
                // Grey cloud opacity defines one camera-depth PDF for air, SS and MS.
                float proposal = cameraOpacity;
                lightingWeight += proposal;
                ct_updateReservoir(reservoir0, distance, proposal, lightingWeight, rng);
                ct_updateReservoir(reservoir1, distance, proposal, lightingWeight, rng);
                #if SETTING_CLOUDS_MS_LOCAL_SAMPLES == 4
                ct_updateReservoir(reservoir2, distance, proposal, lightingWeight, rng);
                ct_updateReservoir(reservoir3, distance, proposal, lightingWeight, rng);
                #endif
                if (opaque) break;
            }
            if (cameraTransmittance == 0.0) break;
        }
        result.transmittance = vec3(cameraTransmittance);
        if (depthWeight > 0.0) result.distance /= depthWeight;
        if (lightingWeight > 0.0) {
            #if SETTING_CLOUDS_MS_LOCAL_SAMPLES == 4
            const float directWeight = 0.25;
            #else
            const float directWeight = 0.5;
            #endif
            #if SETTING_CLOUDS_MS_LOCAL_SAMPLES > 0
            const float multipleWeight = 1.0 / float(SETTING_CLOUDS_MS_LOCAL_SAMPLES);
            #else
            const float multipleWeight = 0.0;
            #endif
            // UNORM blue noise includes 1.0; use bin centers for the half-open phase sampler.
            vec2 phaseJitter = (rand_stbnVec2(rand_newStbnPos(texelPos, 2u), frameCounter) * 255.0 + 0.5) / 256.0;
            result.inScattering = shadeCloudSample(reservoir0, origin, direction, layerRange, directWeight, multipleWeight, phaseJitter, rng);
            #if SETTING_CLOUDS_MS_LOCAL_SAMPLES == 1
            result.inScattering += shadeCloudSample(reservoir1, origin, direction, layerRange, directWeight, 0.0, fract(phaseJitter + 0.5), rng);
            #else
            result.inScattering += shadeCloudSample(reservoir1, origin, direction, layerRange, directWeight, multipleWeight, fract(phaseJitter + 0.5), rng);
            #endif
            #if SETTING_CLOUDS_MS_LOCAL_SAMPLES == 4
            result.inScattering += shadeCloudSample(reservoir2, origin, direction, layerRange, directWeight, multipleWeight, fract(phaseJitter + vec2(0.25, 0.75)), rng);
            result.inScattering += shadeCloudSample(reservoir3, origin, direction, layerRange, directWeight, multipleWeight, fract(phaseJitter + vec2(0.75, 0.25)), rng);
            #endif
            result.inScattering *= lightingWeight;
        }
    }

    uvec4 packedOutput;
    clouds_ss_historyData_pack(packedOutput, result);
    transient_lowCloudRender_store(texelPos, packedOutput);
}

void main() {
    if (gl_LocalInvocationIndex == 0) {
        vec2 groupCenter = vec2(gl_WorkGroupID.xy << 3u) + 4.0;
        shared_worldGroupCheck = hiz_groupSkyCheck4x4(groupCenter * UPSCALE_FACTOR / CHECK_MIP_FACTOR, CHECK_MIP_LEVEL);
    }
    barrier();

    if (shared_worldGroupCheck) {
        uint threadIndex = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
        ivec2 texelPos = ivec2((gl_WorkGroupID.xy << 3u) + morton_8bDecode(threadIndex));
        if (all(lessThan(texelPos, renderSize))) render(texelPos);
    }
}
