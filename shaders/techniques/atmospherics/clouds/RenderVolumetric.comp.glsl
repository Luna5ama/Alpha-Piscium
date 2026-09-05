#extension GL_KHR_shader_subgroup_basic : enable

#include "Cumulus.glsl"
#include "./amblut/API.glsl"
#include "./ss/Common.glsl"
#include "/util/Celestial.glsl"
#include "/util/Morton.glsl"
#include "/techniques/HiZ.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(RENDER_MULTIPLIER, RENDER_MULTIPLIER);

shared bool shared_worldGroupCheck;
layout(rgba32ui) uniform writeonly uimage2D uimg_rgba32ui;

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
        float lightSelect = rand_stbnVec1(rand_newStbnPos(texelPos, 0), frameCounter);
        float sunAngleWarped = fract(sunAngle + 0.25);
        float sunFactor = smoothstep(0.23035, 0.24035, sunAngleWarped) * (1.0 - smoothstep(0.75965, 0.76965, sunAngleWarped));
        sunFactor = step(0.5, sunFactor + lightSelect * 0.2 - 0.1);
        vec3 lightDir = mix(uval_moonDirWorld, uval_sunDirWorld, sunFactor);
        vec3 illuminance = mix(MOON_ILLUMINANCE, SUN_ILLUMINANCE, sunFactor);
        float cosTheta = clamp(dot(direction, lightDir), -1.0, 1.0);
        CloudParticpatingMedium medium = clouds_cu_medium(cosTheta);
        vec3 albedo = medium.scattering / medium.extinction;

        vec2 ambientJitter = rand_stbnVec2(rand_newStbnPos(texelPos, 1), frameCounter);
        vec3 ambient = clouds_amblut_sample(cloods_amblut_uv(direction, ambientJitter), CLOUDS_AMBLUT_LAYER_CUMULUS);
        float jitter = rand_stbnVec1(rand_newStbnPos(texelPos, 1), frameCounter);
        float lightJitter = rand_stbnVec1(rand_newStbnPos(texelPos, 2), frameCounter);
        vec3 shadowDir = rand_sampleInCone(lightDir, SUN_ANGULAR_RADIUS, ambientJitter);
        float rayLength = interval.y - interval.x;
        int steps = int(clamp(rayLength * 3.0, float(SETTING_CLOUDS_LOW_STEP_MIN), float(max(SETTING_CLOUDS_LOW_STEP_MIN, SETTING_CLOUDS_LOW_STEP_MAX))));

        for (int i = 0; i < steps; ++i) {
            float u0 = float(i) / float(steps);
            float u1 = float(i + 1) / float(steps);
            float start = u0 * (0.4 + 0.6 * u0) * rayLength;
            float end = u1 * (0.4 + 0.6 * u1) * rayLength;
            vec3 pos = origin + direction * (interval.x + mix(start, end, jitter));
            float height = length(pos);
            float heightFraction = (height - layerRange.x) / (layerRange.y - layerRange.x);
            float density = clouds_cu_density(pos, heightFraction);
            if (density <= 0.0) continue;

            vec3 up = pos / height;
            vec3 sunOpticalDepth = medium.extinction * clouds_cu_column(pos, shadowDir, layerRange, 10, lightJitter);
            vec3 skyOpticalDepth = medium.extinction * clouds_cu_column(pos, up, layerRange, 4, 0.5);
            vec3 groundOpticalDepth = medium.extinction * clouds_cu_column(pos, -up, layerRange, 3, 0.5);
            vec3 atmosphereTransmittance = atmospherics_air_lut_sampleTransmittance(atmosphere, dot(up, lightDir), height);

            // Successive scattering orders lose directionality and traverse reduced optical depth.
            vec3 sunlight = medium.phase * exp(-sunOpticalDepth);
            float amplitude = 0.6;
            float opticalDepthScale = 0.45;
            float asymmetry = 0.45;
            for (int order = 1; order <= 4; ++order) {
                sunlight += amplitude * phasefunc_HenyeyGreenstein(cosTheta, asymmetry) * exp(-sunOpticalDepth * opticalDepthScale);
                amplitude *= 0.6;
                opticalDepthScale *= 0.45;
                asymmetry *= 0.5;
            }
            vec3 ambientVisibility = 0.5 * (exp(-skyOpticalDepth * 0.15) + exp(-groundOpticalDepth * 0.15));
            vec3 source = illuminance * atmosphereTransmittance * sunlight + ambient * ambientVisibility;
            vec3 transmittance = exp(-medium.extinction * density * (end - start));
            result.inScattering += result.transmittance * albedo * source * (1.0 - transmittance);
            result.transmittance *= transmittance;
            if (mmax3(result.transmittance) < 0.005) break;
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
