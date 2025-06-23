#extension GL_KHR_shader_subgroup_basic : enable

#include "Common.glsl"
#include "Cumulus.glsl"
#include "./amblut/API.glsl"
#include "./ss/Common.glsl"
#include "/util/Celestial.glsl"
#include "/util/Math.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(RENDER_MULTIPLIER, RENDER_MULTIPLIER);

layout(rgba32ui) uniform writeonly uimage2D uimg_csrgba32ui;

const float TRANSMITTANCE_EPSILON = 0.01;

void render(ivec2 texelPosDownScale) {
    vec2 texelPosF = clouds_ss_upscaledTexelCenter(texelPosDownScale);
    float viewZ = -65536.0;

    vec2 screenPos = texelPosF * global_mainImageSizeRcp;
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
            mainRayParams.rayEnd = atmosphere_viewToAtm(atmosphere, endView);
            return;
        }
    }

    mainRayParams.rayDir = normalize(mainRayParams.rayEnd - mainRayParams.rayStart);
    mainRayParams.rayStartHeight = length(mainRayParams.rayStart);
    mainRayParams.rayEndHeight = length(mainRayParams.rayEnd);

    float sunAngleWarped = fract(sunAngle + 0.25);
    float sunLightFactor = smoothstep(0.23035, 0.24035, sunAngleWarped);
    sunLightFactor *= smoothstep(0.76965, 0.75965, sunAngleWarped);
    sunLightFactor *= step(0.5, sunLightFactor);
    vec3 lightDir = mix(uval_moonDirWorld, uval_sunDirWorld, sunLightFactor);
    vec3 lightIlluminance = mix(MOON_ILLUMINANCE, SUN_ILLUMINANCE * PI, sunLightFactor);
    CloudRenderParams renderParams = cloudRenderParams_init(mainRayParams, lightDir, lightIlluminance);

    CloudRaymarchAccumState accumState = clouds_raymarchAccumState_init();

    vec2 jitters = rand_stbnVec2(texelPosDownScale, frameCounter);
    vec3 viewDir = mainRayParams.rayDir;
    vec2 ambLutUV = cloods_amblut_uv(viewDir, jitters);

    {
        float cuHeight = atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT;
        float cuMinHeight = cuHeight - SETTING_CLOUDS_CU_THICKNESS * 0.5;
        float cuMaxHeight = cuHeight + SETTING_CLOUDS_CU_THICKNESS * 0.5;
        float cuHeightDiff = cuHeight - mainRayParams.rayStartHeight;

        float cuRayLenBot = raySphereIntersectNearest(mainRayParams.rayStart, mainRayParams.rayDir, earthCenter, cuMinHeight);
        float cuRayLenTop = raySphereIntersectNearest(mainRayParams.rayStart, mainRayParams.rayDir, earthCenter, cuMaxHeight);

        bool inLayer = abs(cuHeightDiff) < SETTING_CLOUDS_CU_THICKNESS * 0.5;
        float cuOrigin2RayStart = inLayer ? 0.0 : min(cuRayLenBot, cuRayLenTop);

        uint cuFlag = uint(sign(cuHeightDiff) == sign(mainRayParams.rayDir.y)) | uint(inLayer);
        cuFlag &= uint(cuOrigin2RayStart >= 0.0);

        if (bool(cuFlag)) {
            float cuRaySteps = mix(
                SETTING_CLOUDS_LOW_STEP_MIN,
                SETTING_CLOUDS_LOW_STEP_MAX,
                pow(1.0 - abs(mainRayParams.rayDir.y), SETTING_CLOUDS_LOW_STEP_CURVE)
            );
            cuRaySteps = 64.0;
            cuRaySteps = round(cuRaySteps);

            #define CLOUDS_CU_DENSITY (72.0 * SETTING_CLOUDS_CU_DENSITY)

            float cuRayLen = inLayer ? (cuRayLenBot > 0.0 ? cuRayLenBot : min(cuRayLenTop, cuOrigin2RayStart + cuRaySteps)) : max(cuRayLenBot, cuRayLenTop);
            cuRayLen -= cuOrigin2RayStart;

            vec3 ambientIrradiance = clouds_amblut_sample(ambLutUV, CLOUDS_AMBLUT_LAYER_CUMULUS);
            CloudParticpatingMedium cuMedium = clouds_cu_medium(renderParams.cosLightTheta);
            CloudRaymarchLayerParam layerParam = clouds_raymarchLayerParam_init(
                mainRayParams,
                cuMedium,
                ambientIrradiance,
                vec2(cuMinHeight, cuMaxHeight),
                cuOrigin2RayStart,
                cuRayLen,
                rcp(cuRaySteps)
            );
            CloudRaymarchStepState stepState = clouds_raymarchStepState_init(layerParam, jitters.x);
            CloudRaymarchAccumState cuAccum = clouds_raymarchAccumState_init();

            uint cuRayStepsI = uint(cuRaySteps);

            for (uint stepIndex = 0; stepIndex < cuRayStepsI; ++stepIndex) {
                if (stepState.position.w > cuRayLen) break;
                float heightFraction = linearStep(cuMinHeight, cuMaxHeight, stepState.height);
                float sampleDensity = 0.0;
                if (clouds_cu_density(stepState.position.xyz, heightFraction, sampleDensity)) {
                    sampleDensity *= CLOUDS_CU_DENSITY;

                    #define CLOUDS_CU_LIGHT_RAYMARCH_STEP 8
                    #define CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP rcp(float(CLOUDS_CU_LIGHT_RAYMARCH_STEP))
                    const float C = 0.5 * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP;

                    float lightRayTotalDensity = 0.0;
                    {
                        float lightRayLen = SETTING_CLOUDS_CU_THICKNESS * 1.0;
                        vec3 lightRayTotalDelta = renderParams.lightDir * lightRayLen;
                        for (uint lightStepIndex = 0; lightStepIndex < CLOUDS_CU_LIGHT_RAYMARCH_STEP; ++lightStepIndex) {
                            // Use x^2 curve to distribute more samples near the starting point
                            float indexF = float(lightStepIndex);
                            float x = (indexF + jitters.y) * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP;
                            vec3 lightRaySamplePos = stepState.position.xyz + lightRayTotalDelta * pow2(x);

                            float lightSampleHeight = length(lightRaySamplePos);
                            if (lightSampleHeight > cuMaxHeight) break;
                            float lightHeightFraction = linearStep(cuMinHeight, cuMaxHeight, lightSampleHeight);
                            float lightSampleDensity = 0.0;
                            if (clouds_cu_density(lightRaySamplePos, lightHeightFraction, lightSampleDensity)) {
                                // (x + c)^2 - (x - c)^2 = 4xc
                                float x = (indexF + 0.5) * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP;
                                float lightRayStepLength = 4.0 * x * C * lightRayLen;
                                lightRayTotalDensity += lightSampleDensity * lightRayStepLength;
                            }
                        }
                    }
                    lightRayTotalDensity *= CLOUDS_CU_DENSITY;
                    vec3 lightRayOpticalDepth = cuMedium.extinction * lightRayTotalDensity;
                    vec3 lightRayTransmittance = exp(-lightRayOpticalDepth);

                    clouds_computeLighting(
                        atmosphere,
                        renderParams,
                        layerParam,
                        stepState,
                        sampleDensity,
                        lightRayTransmittance,
                        cuAccum
                    );
                }

                if (cuAccum.totalTransmittance.y < TRANSMITTANCE_EPSILON) {
                    break;
                }

                clouds_raymarchStepState_update(stepState);
            }

            const float TRANSMITTANCE_DECAY = 10.0;
            cuAccum.totalTransmittance = pow(cuAccum.totalTransmittance, vec3(exp2(-cuOrigin2RayStart * 0.1)));
            cuAccum.totalInSctr *= exp2(-pow2(cuOrigin2RayStart) * 0.002);

            float aboveFlag = float(cuHeightDiff < 0.0);
            accumState.totalInSctr = mix(
                accumState.totalInSctr + cuAccum.totalInSctr * accumState.totalTransmittance, // Below
                cuAccum.totalInSctr * cuAccum.totalTransmittance + cuAccum.totalInSctr, // Above
                aboveFlag
            );
            accumState.totalTransmittance *= cuAccum.totalTransmittance;
        }
    }

    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    historyData.inScattering = accumState.totalInSctr;
    historyData.transmittance = accumState.totalTransmittance;
    historyData.hLen = 1.0;
    uvec4 packedOutput = uvec4(0u);
    clouds_ss_historyData_pack(packedOutput, historyData);
    imageStore(uimg_csrgba32ui, gi_diffuseHistory_texelToTexel(texelPosDownScale), packedOutput);
}

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 3;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);
    if (all(lessThan(texelPos, renderSize))) {
        render(texelPos);
    }
}