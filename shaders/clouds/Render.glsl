#include "Common.glsl"
#include "Cirrus.glsl"
#include "Cumulus.glsl"
#include "./amblut/API.glsl"
#include "/util/Celestial.glsl"
#include "/util/Math.glsl"

const float DENSITY_EPSILON = 0.0001;
const float TRANSMITTANCE_EPSILON = 0.01;

void renderCloud(ivec2 texelPos, sampler2D viewZTex, inout vec4 outputColor) {
    float viewZ = texelFetch(viewZTex, texelPos, 0).r;
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * global_mainImageSizeRcp;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

    vec3 originView = vec3(0.0);
    vec3 endView = viewPos;

    mat3 vectorView2World = mat3(gbufferModelViewInverse);

    vec3 viewDirView = normalize(endView - originView);
    vec3 viewDirWorld = normalize(vectorView2World * viewDirView);

    float ignValue = rand_IGN(texelPos, frameCounter);
    AtmosphereParameters atmosphere = getAtmosphereParameters();

    CloudMainRayParams mainRayParams;
    mainRayParams.rayStart = atmosphere_viewToAtm(atmosphere, originView);
    const vec3 earthCenter = vec3(0.0);

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

            float tBottom = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, atmosphere.bottom);
            float tTop = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, atmosphere.top);
            float rayLen = 0.0;

            if (tBottom < 0.0) {
                if (tTop < 0.0) {
                    return;// No intersection with earth nor atmosphere: stop right away
                } else {
                    rayLen = tTop;
                }
            } else {
                if (tTop > 0.0) {
                    rayLen = min(tTop, tBottom);
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

    vec2 jitters = rand_stbnVec2(texelPos, frameCounter);
    vec3 viewDir = -mainRayParams.rayDir;
    vec2 ambLutUV = cloods_amblut_uv(viewDir, jitters);

    #ifdef SETTING_CLOUDS_CU
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
            cuRaySteps = round(cuRaySteps);

            #define CLOUDS_CU_LIGHT_RAYMARCH_STEP 4
            #define CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP rcp(float(CLOUDS_CU_LIGHT_RAYMARCH_STEP))
            #define CLOUDS_CU_DENSITY (64.0 * SETTING_CLOUDS_CU_DENSITY)

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

                    float lightRayTotalDensity = 0.0;
                    {
                        float lightRayLen = 0.5;
                        float lightRayStepLength = lightRayLen * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP;
                        vec3 lightRayPos = stepState.position.xyz;
                        for (uint lightStepIndex = 0; lightStepIndex < CLOUDS_CU_LIGHT_RAYMARCH_STEP; ++lightStepIndex) {
                            vec3 lightRayStepDelta = lightRayStepLength * renderParams.lightDir;
                            lightRayPos += lightRayStepDelta;
                            vec3 lightRaySamplePos = lightRayPos + lightRayStepDelta * (jitters.y - 0.5);
                            float lightSampleHeight = length(lightRaySamplePos);
                            if (lightSampleHeight > cuMaxHeight) break;
                            float lightHeightFraction = linearStep(cuMinHeight, cuMaxHeight, lightSampleHeight);
                            float lightSampleDensity = 0.0;
                            if (clouds_cu_density(lightRaySamplePos, lightHeightFraction, lightSampleDensity)) {
                                lightRayTotalDensity += lightSampleDensity * lightRayStepLength;
                            }
                            lightRayStepLength *= 1.5;
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

                    if (colors_sRGB_luma(cuAccum.totalTransmittance) < TRANSMITTANCE_EPSILON) {
                        break;
                    }
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
    #endif

    #ifdef SETTING_CLOUDS_CI
    {
        float ciHeight = atmosphere.bottom + SETTING_CLOUDS_CI_HEIGHT;
        float ciMinHeight = ciHeight - 0.5;
        float ciMaxHeight = ciHeight + 0.5;

        float ciHeighDiff = ciHeight - mainRayParams.rayStartHeight;
        float ciOrigin2RayOffset = raySphereIntersectNearest(mainRayParams.rayStart, mainRayParams.rayDir, earthCenter, ciHeight);
        uint ciFlag = uint(sign(ciHeighDiff) == sign(mainRayParams.rayDir.y)) & uint(ciOrigin2RayOffset > 0.0);

        if (bool(ciFlag)) {
            vec3 ambientIrradiance = clouds_amblut_sample(ambLutUV, CLOUDS_AMBLUT_LAYER_CIRRUS);
            CloudParticpatingMedium ciMedium = clouds_ci_medium(renderParams.cosLightTheta);
            CloudRaymarchLayerParam layerParam = clouds_raymarchLayerParam_init(
                mainRayParams,
                ciMedium,
                ambientIrradiance,
                vec2(ciMinHeight, ciMaxHeight),
                ciOrigin2RayOffset,
                1.0,
                1.0
            );
            CloudRaymarchStepState stepState = clouds_raymarchStepState_init(layerParam, 0.0);
            float sampleDensity = clouds_ci_density(stepState.position.xyz);
            CloudRaymarchAccumState ciAccum = clouds_raymarchAccumState_init();
            if (sampleDensity > DENSITY_EPSILON) {
                clouds_computeLighting(
                    atmosphere,
                    renderParams,
                    layerParam,
                    stepState,
                    sampleDensity,
                    vec3(1.0),
                    ciAccum
                );
            }

            const float TRANSMITTANCE_DECAY = 10.0;
            ciAccum.totalTransmittance = pow(ciAccum.totalTransmittance, vec3(exp2(-ciOrigin2RayOffset * 0.1)));
            ciAccum.totalInSctr *= exp2(-pow2(ciOrigin2RayOffset) * 0.002);

            float aboveFlag = float(ciHeighDiff < 0.0);
            accumState.totalInSctr = mix(
                accumState.totalInSctr + ciAccum.totalInSctr * accumState.totalTransmittance, // Below
                accumState.totalInSctr * ciAccum.totalTransmittance + ciAccum.totalInSctr, // Above
                aboveFlag
            );
            accumState.totalTransmittance *= ciAccum.totalTransmittance;
        }
    }
    #endif

    outputColor.rgb *= accumState.totalTransmittance;
    outputColor.rgb += accumState.totalInSctr;
}