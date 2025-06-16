#include "Common.glsl"
#include "Cirrus.glsl"
#include "Cumulus.glsl"
#include "/atmosphere/Common.glsl"
#include "/util/Celestial.glsl"

uniform sampler3D usam_cloudsAmbLUT;

const float DENSITY_EPSILON = 0.0001;

void renderCloud(ivec2 texelPos, sampler2D viewZTex, inout vec4 outputColor) {
    float viewZ = texelFetch(viewZTex, texelPos, 0).r;
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * global_mainImageSizeRcp;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);

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
            mainRayParams.rayStart.y = max(mainRayParams.rayStart.y, atmosphere.bottom + 0.5);

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

    vec3 viewDir = -mainRayParams.rayDir;
    vec2 ambLutUV = coords_equirectanglarForwardHorizonBoost(viewDir);

    #ifdef SETTING_CLOUDS_CU
    {
        float cuHeight = atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT;
        float cuMinHeight = cuHeight - SETTING_CLOUDS_CU_THICKNESS * 0.5;
        float cuMaxHeight = cuHeight + SETTING_CLOUDS_CU_THICKNESS * 0.5;
        float cuHeightDiff = cuHeight - mainRayParams.rayStartHeight;

        float cuRayLenBottom = raySphereIntersectNearest(mainRayParams.rayStart, mainRayParams.rayDir, earthCenter, cuMinHeight);
        float cuRayLenTop = raySphereIntersectNearest(mainRayParams.rayStart, mainRayParams.rayDir, earthCenter, cuMaxHeight);

        float cuOrigin2RayStart = abs(cuHeightDiff) < SETTING_CLOUDS_CU_THICKNESS * 0.5 ? 0.0 : min(cuRayLenBottom, cuRayLenTop);

        uint cuFlag = uint(sign(cuHeightDiff) == sign(mainRayParams.rayDir.y)) & uint(cuOrigin2RayStart >= 0.0);

        if (bool(cuFlag)) {
            #define CLOUDS_CU_RAYMARCH_STEP 64
            #define CLOUDS_CU_RAYMARCH_STEP_RCP rcp(float(CLOUDS_CU_RAYMARCH_STEP))
            #define CLOUDS_CU_LIGHT_RAYMARCH_STEP 4
            #define CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP rcp(float(CLOUDS_CU_LIGHT_RAYMARCH_STEP))

            float cuRayLen = max(cuRayLenBottom, cuRayLenTop) - cuOrigin2RayStart;
            cuRayLen = min(cuRayLen, SETTING_CLOUDS_CU_THICKNESS * 4.0); // Limit raymarching length to avoid artifacts

            vec3 ambientIrradiance = texture(usam_cloudsAmbLUT, vec3(ambLutUV, 1.5 / 6.0)).rgb;
            CloudParticpatingMedium cuMedium = clouds_cu_medium(renderParams.LDotV);
            CloudRaymarchLayerParam layerParam = clouds_raymarchLayerParam_init(
                mainRayParams,
                cuMedium,
                ambientIrradiance,
                vec2(cuMinHeight, cuMaxHeight),
                cuOrigin2RayStart,
                cuRayLen,
                CLOUDS_CU_RAYMARCH_STEP_RCP
            );

            CloudRaymarchAccumState cuAccum = clouds_raymarchAccumState_init();
            float jitter = rand_stbnVec1(texelPos, frameCounter);

            for (uint stepIndex = 0; stepIndex < CLOUDS_CU_RAYMARCH_STEP; ++stepIndex) {
                float stepIndexF = float(stepIndex);
                vec3 rayPosCenter = layerParam.rayStart + (stepIndexF + 0.5) * layerParam.rayStep.xyz;
                vec3 rayPosJittered = layerParam.rayStart + (stepIndexF + jitter) * layerParam.rayStep.xyz;
                float coverage = clouds_cu_coverage(rayPosJittered);

                if (coverage > DENSITY_EPSILON) {
                    float density = clouds_cu_density(rayPosJittered);
                    float sampleDensity = linearStep(density * 0.2, 1.0, coverage) * 4.0;

                    if (sampleDensity > DENSITY_EPSILON) {
                        float lightRayLen = raySphereIntersectNearest(rayPosCenter, renderParams.lightDir, earthCenter, cuMaxHeight);
                        lightRayLen = min(lightRayLen, 0.5);
                        float lightRayStepDelta = lightRayLen * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP;

                        float lightRayTotalDensity = 0.0;
//                        for (uint lightStepIndex = 0; lightStepIndex < CLOUDS_CU_LIGHT_RAYMARCH_STEP; ++lightStepIndex) {
//                            float lightStepIndexF = float(lightStepIndex) + jitter;
//                            vec3 lightSamplePos = rayPosCenter + renderParams.lightDir * lightStepIndexF * lightRayStepDelta;
//                            float lightCoverage = clouds_cu_coverage(lightSamplePos);
//                            if (lightCoverage > DENSITY_EPSILON) {
//                                float lightDensity = clouds_cu_density(lightSamplePos);
//                                float lightSampleDensity = linearStep(lightDensity * 0.25, 1.0, lightCoverage) * 192.0;
//                                lightRayTotalDensity += lightSampleDensity * lightRayStepDelta;
//                            }
//                        }
                        vec3 lightRayOpticalDepth = cuMedium.extinction * lightRayTotalDensity;
                        vec3 lightRayTransmittance = exp(-lightRayOpticalDepth);

                        CloudRaymarchStepState stepState = clouds_raymarchStepState_init(
                            layerParam.rayStep.w,
                            rayPosCenter,
                            sampleDensity,
                            lightRayTransmittance
                        );
                        clouds_computeLighting(
                            atmosphere,
                            renderParams,
                            layerParam,
                            stepState,
                            cuAccum
                        );
                    }
                }
            }


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
            CloudParticpatingMedium ciMedium = clouds_ci_medium(renderParams.LDotV);
            CloudRaymarchLayerParam layerParam = clouds_raymarchLayerParam_init(
                mainRayParams,
                ciMedium,
                ambientIrradiance,
                vec2(ciMinHeight, ciMaxHeight),
                ciOrigin2RayOffset,
                1.0,
                1.0
            );

            float sampleDensity = clouds_ci_density(layerParam.rayStart);
            vec3 ambientIrradiance = texture(usam_cloudsAmbLUT, vec3(ambLutUV, 3.5 / 6.0)).rgb;

            CloudRaymarchAccumState ciAccum = clouds_raymarchAccumState_init();

            if (sampleDensity > DENSITY_EPSILON) {
                CloudRaymarchStepState stepState = clouds_raymarchStepState_init(
                    1.0,
                    layerParam.rayStart,
                    sampleDensity
                );
                clouds_computeLighting(
                    atmosphere,
                    renderParams,
                    layerParam,
                    stepState,
                    ciAccum
                );
            }
            float aboveFlag = float(ciHeighDiff < 0.0);
            accumState.totalInSctr = mix(
                accumState.totalInSctr + ciAccum.totalInSctr * accumState.totalTransmittance, // Below
                ciAccum.totalInSctr * ciAccum.totalTransmittance + ciAccum.totalInSctr, // Above
                aboveFlag
            );
            accumState.totalTransmittance *= ciAccum.totalTransmittance;
        }
    }
    #endif

    outputColor.rgb += accumState.totalInSctr;
    outputColor.rgb *= accumState.totalTransmittance;
}