#extension GL_KHR_shader_subgroup_basic : enable

#include "Common.glsl"
#include "Cumulus.glsl"
#include "./amblut/API.glsl"
#include "./ss/Common.glsl"
#include "/util/Celestial.glsl"
#include "/util/Math.glsl"
#include "/util/Morton.glsl"
#include "/techniques/HiZ.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(RENDER_MULTIPLIER, RENDER_MULTIPLIER);

shared bool shared_worldGroupCheck;

layout(rgba32ui) uniform writeonly uimage2D uimg_rgba32ui;

const float TRANSMITTANCE_EPSILON = 0.01;

float EvalIGN(vec2 uv)
{
    uint frame = uint(frameCounter);

    //frame += WellonsHash2(WeylHash(uvec2(uv)/4u)) % 4u;

    if((frame & 2u) != 0u) uv = vec2(-uv.y, uv.x);
    if((frame & 1u) != 0u) uv.x = -uv.x;

    //return fract(52.9829189 * fract(dot(uv, vec2(0.06711056, 0.00583715))) + float(frame)*0.41421356);
    //return fract(52.9829189 * fract(dot(uv, vec2(0.06711056, 0.00583715))));
    //return fract(IGN(uv)+float(frame)*0.41421356*1.0);

    // http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/#dither
    return fract(uv.x*0.7548776662 + uv.y*0.56984029 + float(frame)*0.41421356*1.0);
}


void render(ivec2 texelPosDownScale) {
    vec2 texelPosF = clouds_ss_upscaledTexelCenter(texelPosDownScale);
    float viewZ = -65536.0;

    vec2 screenPos = texelPosF * uval_mainImageSizeRcp;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

    vec3 originView = vec3(0.0);
    vec3 endView = viewPos;

    mat3 vectorView2World = mat3(gbufferModelViewInverse);

    vec3 viewDirView = normalize(endView - originView);
    vec3 viewDirWorld = normalize(vectorView2World * viewDirView);

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    CloudMainRayParams mainRayParams;
    mainRayParams.rayStart = atmosphere_viewToAtmNoClamping(atmosphere, originView);
    const vec3 earthCenter = vec3(0.0);

    float maxRayLen = 0.0;

    {
        vec3 rayDir = viewDirWorld;
        if (endView.z == -65536.0) {
            // Check if ray origin is outside the atmosphere
            if (length(mainRayParams.rayStart) > atmosphere.top) {
                float tTop = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, atmosphere.top);
                if (tTop < 0.0) {
                    return;// No intersection with atmosphere: stop right away
                }
                mainRayParams.rayStart += rayDir * (tTop + 0.001);
            }

            float clampedBottom = min(atmosphere.bottom, length(mainRayParams.rayStart) - 0.001);
            float tBottom = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, clampedBottom);
            float tTop = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, atmosphere.top);
            float rayLen = 0.0;

            if (tBottom < 0.0) {
                if (tTop < 0.0) {
                    return;// No intersection with earth nor atmosphere: stop right away
                } else {
                    rayLen = tTop;
                    maxRayLen = tTop;
                }
            } else {
                if (tTop > 0.0) {
                    rayLen = min(tTop, tBottom);
                    maxRayLen = max(tTop, tBottom);
                }
            }

            mainRayParams.rayEnd = mainRayParams.rayStart + rayDir * rayLen;
        } else {
            return;
        }
    }

    mainRayParams.rayDir = normalize(mainRayParams.rayEnd - mainRayParams.rayStart);
    mainRayParams.rayStartHeight = length(mainRayParams.rayStart);
    mainRayParams.rayEndHeight = length(mainRayParams.rayEnd);

    float lightSelectRand = rand_stbnVec1(rand_newStbnPos(texelPosDownScale, 0), frameCounter);

    float sunAngleWarped = fract(sunAngle + 0.25);
    float sunLightFactor = smoothstep(0.23035, 0.24035, sunAngleWarped);
    sunLightFactor *= smoothstep(0.76965, 0.75965, sunAngleWarped);
    sunLightFactor += lightSelectRand * 0.2 - 0.1;
    sunLightFactor = step(0.5, sunLightFactor);
    vec3 lightDir = mix(uval_moonDirWorld, uval_sunDirWorld, sunLightFactor);
    vec3 lightIlluminance = mix(MOON_ILLUMINANCE, SUN_ILLUMINANCE, sunLightFactor);
    CloudRenderParams renderParams = cloudRenderParams_init(mainRayParams, lightDir, lightIlluminance);
    CloudRaymarchAccumState cuAccum = clouds_raymarchAccumState_init();

    vec2 ambLutJitterRand = rand_stbnVec2(rand_newStbnPos(texelPosDownScale, 1), frameCounter);
    vec3 viewDir = mainRayParams.rayDir;
    vec2 ambLutUV = cloods_amblut_uv(viewDir, ambLutJitterRand);

    {
        float cuMinHeight = atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT;
        float cuMaxHeight = cuMinHeight + SETTING_CLOUDS_CU_THICKNESS;
        float cuMidHeight = cuMinHeight + SETTING_CLOUDS_CU_THICKNESS * 0.5;
        float cuHeightDiff = cuMidHeight - mainRayParams.rayStartHeight;

        float cuRayLenBot = raySphereIntersectNearest(mainRayParams.rayStart, mainRayParams.rayDir, earthCenter, cuMinHeight);
        float cuRayLenTop = raySphereIntersectNearest(mainRayParams.rayStart, mainRayParams.rayDir, earthCenter, cuMaxHeight);

        bool inLayer = abs(cuHeightDiff) < SETTING_CLOUDS_CU_THICKNESS * 0.5;
        float cuOrigin2RayStart = inLayer ? 0.0 : (cuHeightDiff < 0.0 ? cuRayLenTop : cuRayLenBot);

        uint cuFlag = uint(sign(cuHeightDiff) == sign(mainRayParams.rayDir.y)) | uint(inLayer);
        cuFlag &= uint(cuOrigin2RayStart >= 0.0);

        if (bool(cuFlag)) {
            #define CLOUDS_CU_DENSITY (256.0 * SETTING_CLOUDS_CU_DENSITY)

            const float CLOUDS_CU_MAX_RAY_LENGTH = 50.0;
            float cuRayLen = mainRayParams.rayDir.y < 0.0 ? cuRayLenBot : cuRayLenTop;
            cuRayLen -= cuOrigin2RayStart;
            cuRayLen = cuRayLen <= 0.0 ? CLOUDS_CU_MAX_RAY_LENGTH : cuRayLen;
            cuRayLen = min(cuRayLen, CLOUDS_CU_MAX_RAY_LENGTH);
            float cuRaySteps = cuRayLen / CLOUDS_CU_MAX_RAY_LENGTH * float(SETTING_CLOUDS_LOW_STEP_MAX);
            cuRaySteps = max(cuRaySteps, SETTING_CLOUDS_LOW_STEP_MIN);
            uint cuRayStepsI = uint(cuRaySteps);

            vec3 ambientIrradiance = clouds_amblut_sample(ambLutUV, CLOUDS_AMBLUT_LAYER_CUMULUS);
            CloudParticpatingMedium cuMedium = clouds_cu_medium(renderParams.cosLightTheta);
            CloudRaymarchLayerParam layerParam = clouds_raymarchLayerParam_init(
                mainRayParams,
                cuMedium,
                ambientIrradiance,
                vec2(cuMinHeight, cuMaxHeight),
                cuOrigin2RayStart,
                cuRayLen,
                cuRayStepsI
            );
            CloudRaymarchStepState stepState = clouds_raymarchStepState_init(layerParam);

            vec2 lightRayDirJitterRand = rand_stbnVec2(rand_newStbnPos(texelPosDownScale, 2), frameCounter);
            vec3 lightRayDir = renderParams.lightDir;
            // Scale cone radius by 4 to simulate subsurface scattering
            lightRayDir = rand_sampleInCone(lightRayDir, SUN_ANGULAR_RADIUS * 4.0, lightRayDirJitterRand);

            float mainRayJitterRand = rand_stbnVec1(rand_newStbnPos(texelPosDownScale, 1), frameCounter);
            float lightRayJitterRand = rand_stbnVec1(rand_newStbnPos(texelPosDownScale, 2), frameCounter);

            for (uint stepIndex = 0; stepIndex < cuRayStepsI; ++stepIndex) {
                if (stepState.position.w > cuRayLen) break;

                float heightFraction = linearStep(cuMinHeight, cuMaxHeight, stepState.height);
                float sampleDensity = 0.0;
                float sampleDensityLod = 0.0;
                if (clouds_cu_density(stepState.position.xyz, heightFraction, true, sampleDensity, sampleDensityLod)) {
                    sampleDensity *= CLOUDS_CU_DENSITY;
                    sampleDensityLod *= CLOUDS_CU_DENSITY;

                    #define CLOUDS_CU_LIGHT_RAYMARCH_STEP 8
                    #define CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP rcp(float(CLOUDS_CU_LIGHT_RAYMARCH_STEP))
                    const float C = 0.5 * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP;

                    float lightRayTotalDensity = 0.0;
                    {
                        float lightRayLen = SETTING_CLOUDS_CU_THICKNESS * 1.0;
                        vec3 lightRayTotalDelta = lightRayDir * lightRayLen;
                        for (uint lightStepIndex = 0; lightStepIndex < CLOUDS_CU_LIGHT_RAYMARCH_STEP; ++lightStepIndex) {
                            // Use x^2 curve to distribute more samples near the starting point
                            float indexF = float(lightStepIndex);
                            float x = (indexF + lightRayJitterRand) * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP;
                            vec3 lightRaySamplePos = stepState.position.xyz + lightRayTotalDelta * pow2(x);

                            float lightSampleHeight = length(lightRaySamplePos);
                            if (lightSampleHeight > cuMaxHeight) break;
                            float lightHeightFraction = linearStep(cuMinHeight, cuMaxHeight, lightSampleHeight);
                            float lightSampleDensity = 0.0;
                            float lightSampleDensityLod = 0.0;
                            if (clouds_cu_density(lightRaySamplePos, lightHeightFraction, true, lightSampleDensity, lightSampleDensityLod)) {
                                // (x + c)^2 - (x - c)^2 = 4xc
                                float x = (indexF + 0.5) * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP;
                                float lightRayStepLength = 4.0 * x * C * lightRayLen;
                                lightRayTotalDensity += lightSampleDensity * lightRayStepLength;
                            }
                        }
                    }
                    lightRayTotalDensity *= CLOUDS_CU_DENSITY;
                    vec3 lightRayOpticalDepth = cuMedium.extinction * lightRayTotalDensity;

                    clouds_computeLighting(
                        atmosphere,
                        renderParams,
                        layerParam,
                        stepState,
                        sampleDensity,
                        sampleDensityLod,
                        lightRayOpticalDepth,
                        cuAccum
                    );
                }

                if (cuAccum.totalTransmittance.y < TRANSMITTANCE_EPSILON) {
                    break;
                }

                clouds_raymarchStepState_update(stepState, float(stepIndex) + mainRayJitterRand);
            }
        }
    }

    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    historyData.inScattering = clamp(cuAccum.totalInSctr, 0.0, FP16_MAX);
    historyData.transmittance = cuAccum.totalTransmittance;
    historyData.hLen = 1.0;
    uvec4 packedOutput = uvec4(0u);
    clouds_ss_historyData_pack(packedOutput, historyData);
    transient_lowCloudRender_store(texelPosDownScale, packedOutput);
}

void main() {
    if (gl_LocalInvocationIndex == 0) {
        vec2 groupCenter = vec2(gl_WorkGroupID.xy << 3u) + 4.0;
        vec2 hizCheckPos = groupCenter * UPSCALE_FACTOR / CHECK_MIP_FACTOR;
        shared_worldGroupCheck = hiz_groupSkyCheck4x4(hizCheckPos, CHECK_MIP_LEVEL);
    }
    barrier();

    if (shared_worldGroupCheck) {
        uvec2 workGroupOrigin = gl_WorkGroupID.xy << 3;
        uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
        uvec2 mortonPos = morton_8bDecode(threadIdx);
        uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
        ivec2 texelPos = ivec2(mortonGlobalPosU);
        if (all(lessThan(texelPos, renderSize))) {
            render(texelPos);
        }
    }
}