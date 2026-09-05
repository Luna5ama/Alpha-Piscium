#extension GL_KHR_shader_subgroup_basic : enable

#include "Transport.glsl"
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

vec3 skySingleScattering(vec3 pos, vec3 outgoing, vec2 layerRange, vec2 jitter, float lightJitter) {
    vec3 tangent = normalize(cross(outgoing, abs(outgoing.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(outgoing, tangent);
    vec3 result = vec3(0.0);
    for (uint s = 0u; s < 2u; ++s) {
        vec2 xi = fract(jitter + rand_r2Seq2(s));
        // An equal mixture of the three AP0 phase densities is the proposal.
        float channelXi = xi.x * 3.0;
        vec3 cosines = ct_phaseCosines(fract(channelXi));
        float mu = cosines[int(channelXi)];
        vec3 incoming = outgoing * mu + (tangent * cos(2.0 * PI * xi.y) + bitangent * sin(2.0 * PI * xi.y)) * sqrt(1.0 - mu * mu);
        vec3 phase = ct_toAP0(clouds_phase_cu(mu, SETTING_CLOUDS_CU_PHASE_RATIO));
        vec3 transmittance = exp(-CT_EXTINCTION * clouds_cu_column(pos, -incoming, layerRange, 12, 6.0, lightJitter));
        result += ct_boundary(ct_angle(incoming)) * transmittance * phase / dot(phase, vec3(1.0 / 3.0));
    }
    return colors2_constants_toWorkSpace(result * 0.5);
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
        CloudParticpatingMedium medium = clouds_cu_medium(0.0);
        vec3 albedo = medium.scattering / medium.extinction;
        ivec4 transportAngles;
        vec4 angularWeights;
        ct_angularWeights(-direction, transportAngles, angularWeights);
        vec2 ambientJitter = rand_stbnVec2(rand_newStbnPos(texelPos, 1), frameCounter);
        float jitter = rand_stbnVec1(rand_newStbnPos(texelPos, 1), frameCounter);
        float lightJitter = rand_stbnVec1(rand_newStbnPos(texelPos, 2), frameCounter);
        float rayLength = interval.y - interval.x;
        int steps = int(clamp(rayLength * 3.0, float(SETTING_CLOUDS_LOW_STEP_MIN), float(max(SETTING_CLOUDS_LOW_STEP_MIN, SETTING_CLOUDS_LOW_STEP_MAX))));

        vec3 airOrigin = atmosphere_viewToAtm(atmosphere, vec3(0.0));
        LightParameters sunParams = lightParameters_init(atmosphere, SUN_ILLUMINANCE, uval_sunDirWorld, direction);
        LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, direction);
        ScatteringParameters airScattering = scatteringParameters_init(sunParams, moonParams, 1.0);
        ScatteringResult air = scatteringResult_init();
        float airDistance = 0.0;
        float emptyDistance = interval.x;
        bool inside = interval.x == 0.0 && clouds_cu_density(origin, (length(origin) - layerRange.x) / (layerRange.y - layerRange.x)) > 0.0;

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
                // Resolve the first interface, then shade near the first scattering event in the occupied segment.
                float occupiedDistance = distance;
                for (int refine = 0; refine < 4; ++refine) {
                    float midpoint = (emptyDistance + occupiedDistance) * 0.5;
                    vec3 midpointPos = origin + direction * midpoint;
                    float midpointHeight = (length(midpointPos) - layerRange.x) / (layerRange.y - layerRange.x);
                    if (clouds_cu_density(midpointPos, midpointHeight) > 0.0) occupiedDistance = midpoint;
                    else emptyDistance = midpoint;
                }
                occupiedStart = occupiedDistance;
                float meanFreePath = 1.0 / (CU_DENSITY_SCALE * dot(medium.extinction, vec3(1.0 / 3.0)) * mix(0.55, 1.35, smoothstep(0.0, 0.32, heightFraction)));
                distance = min(occupiedDistance + meanFreePath, distance);
                pos = origin + direction * distance;
                height = length(pos);
                heightFraction = (height - layerRange.x) / (layerRange.y - layerRange.x);
                density = clouds_cu_density(pos, heightFraction);
            }
            inside = true;

            RaymarchParameters airParams = raymarchParameters_init();
            airParams.rayStart = airOrigin + direction * airDistance;
            airParams.rayEnd = airOrigin + direction * distance;
            airParams.steps = uint(clamp((distance - airDistance) * 2.0, 4.0, 32.0));
            air = scatteringResult_blendLayer(air, raymarchSky(atmosphere, airParams, airScattering, 8.0), false);
            airDistance = distance;

            vec3 up = pos / height;
            vec3 source = ct_sample(pos, transportAngles, angularWeights);
            source += skySingleScattering(pos, -direction, layerRange, fract(ambientJitter + rand_r2Seq2(uint(i))), lightJitter);
            for (int light = 0; light < 2; ++light) {
                vec3 lightDir = light == 0 ? uval_sunDirWorld : uval_moonDirWorld;
                if (raySphereIntersectNearest(pos, lightDir, vec3(0.0), atmosphere.bottom) >= 0.0) continue;
                vec3 illuminance = light == 0 ? SUN_ILLUMINANCE : MOON_ILLUMINANCE;
                float radius = light == 0 ? SUN_ANGULAR_RADIUS : MOON_ANGULAR_RADIUS;
                vec3 shadowDir = rand_sampleInCone(lightDir, radius, ambientJitter);
                vec3 opticalDepth = medium.extinction * clouds_cu_column(pos, shadowDir, layerRange, 16, 6.0, lightJitter);
                vec3 phase = clouds_phase_cu(clamp(dot(direction, lightDir), -1.0, 1.0), SETTING_CLOUDS_CU_PHASE_RATIO);
                vec3 atmosphereTr = atmospherics_air_lut_sampleTransmittance(atmosphere, dot(up, lightDir), height);
                source += illuminance * atmosphereTr * phase * exp(-opticalDepth);
            }
            vec3 tau = medium.extinction * density * (interval.x + end - occupiedStart);
            vec3 transmittance = exp(-tau);
            vec3 opacity = mix(1.0 - transmittance, tau * (1.0 - 0.5 * tau + tau * tau / 6.0), lessThan(tau, vec3(0.001)));
            // Opacity weights the clear-air radiance up to each actual cloud sample.
            vec3 radiance = albedo * source * air.transmittance + air.inScattering;
            result.inScattering += result.transmittance * opacity * radiance;
            result.transmittance *= transmittance;
            if (mmax3(result.transmittance) < 0.005) {
                // Complete the opaque tail with the local source instead of leaking an arbitrary fraction of the sky.
                result.inScattering += result.transmittance * radiance;
                result.transmittance = vec3(0.0);
                break;
            }
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
