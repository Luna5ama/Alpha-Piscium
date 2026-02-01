#include "Common.glsl"
#include "clouds/Common.glsl"
#include "clouds/amblut/API.glsl"
#include "clouds/Cirrus.glsl"
#include "clouds/Cumulus.glsl"
#include "clouds/ss/Common.glsl"
#include "air/lut/API.glsl"
#include "air/UnwarpEpipolar.glsl"
#include "/util/BitPacking.glsl"
#include "/util/Celestial.glsl"
#include "/util/Math.glsl"

layout(rgba32ui) uniform restrict writeonly uimage2D uimg_rgba32ui;

const float DENSITY_EPSILON = 0.0001;

ScatteringResult atmospherics_skyComposite(ivec2 texelPos) {
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * uval_mainImageSizeRcp;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

    vec3 originView = vec3(0.0);
    vec3 endView = viewPos;

    mat3 vectorView2World = mat3(gbufferModelViewInverse);

    vec3 viewDirView = normalize(endView - originView);
    vec3 viewDirWorld = normalize(vectorView2World * viewDirView);

    float ignValue = rand_IGN(texelPos, frameCounter);
    AtmosphereParameters atmosphere = getAtmosphereParameters();

    CloudMainRayParams mainRayParams;
    mainRayParams.rayStart = atmosphere_viewToAtmNoClamping(atmosphere, originView);
    const vec3 earthCenter = vec3(0.0);

    ScatteringResult compositeResult = scatteringResult_init();

    vec3 rayDir = viewDirWorld;

    if (viewZ == -65536.0) {
        float rayLen = 0.0;

        // Check if ray origin is outside the atmosphere
        if (length(mainRayParams.rayStart) > atmosphere.top) {
            float tTop = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, atmosphere.top);
            if (tTop < 0.0) {
                rayLen = -1.0; // No intersection with atmosphere: stop right away
            }
            mainRayParams.rayStart += rayDir * (tTop + 0.001);
        }

        float clampedBottom = min(atmosphere.bottom, length(mainRayParams.rayStart) - 0.001);
        float tBottom = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, clampedBottom);
        float tTop = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, atmosphere.top);

        if (tBottom < 0.0) {
            if (tTop < 0.0) {
                rayLen = -1.0;// No intersection with earth nor atmosphere: stop right away
            } else {
                rayLen = tTop;
            }
        } else {
            if (tTop > 0.0) {
                rayLen = min(tTop, tBottom);
            }
        }

        if (rayLen >= 0.0) {
            mainRayParams.rayEnd = mainRayParams.rayStart + rayDir * rayLen;
            mainRayParams.rayDir = normalize(mainRayParams.rayEnd - mainRayParams.rayStart);
            mainRayParams.rayStartHeight = length(mainRayParams.rayStart);
            mainRayParams.rayEndHeight = length(mainRayParams.rayEnd);

            float sunAngleWarped = fract(sunAngle + 0.25);
            float sunLightFactor = smoothstep(0.23035, 0.24035, sunAngleWarped);
            sunLightFactor *= smoothstep(0.76965, 0.75965, sunAngleWarped);
            sunLightFactor *= step(0.5, sunLightFactor);
            vec3 lightDir = mix(uval_moonDirWorld, uval_sunDirWorld, sunLightFactor);
            vec3 lightIlluminance = mix(MOON_ILLUMINANCE, SUN_ILLUMINANCE, sunLightFactor);
            CloudRenderParams renderParams = cloudRenderParams_init(mainRayParams, lightDir, lightIlluminance);

            vec2 jitters = rand_stbnVec2(texelPos, frameCounter);
            vec3 viewDir = mainRayParams.rayDir;
            vec2 ambLutUV = cloods_amblut_uv(viewDir, jitters);

            vec3 rayEndView = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
            vec3 rayDir = normalize(mat3(gbufferModelViewInverse) * rayEndView);
            SkyViewLutParams skyViewLutParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, rayDir);
            #ifdef SETTING_CLOUDS_CU
            uvec4 packedData = transient_lowCloudAccumulated_fetch(texelPos);
            history_lowCloud_store(texelPos, packedData);
            #endif
            {
                ScatteringResult layerResult = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyViewLutParams, 1.0);
                compositeResult = scatteringResult_blendLayer(compositeResult, layerResult, true);
            }

            #ifdef SETTING_CLOUDS_CU
            {
                float cuMinHeight = atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT;
                float cuMaxHeight = cuMinHeight + SETTING_CLOUDS_CU_THICKNESS;
                float cuMidHeight = cuMinHeight + SETTING_CLOUDS_CU_THICKNESS * 0.5;
                float cuHeightDiff = cuMidHeight - mainRayParams.rayStartHeight;

                float cuRayLenBot = raySphereIntersectNearest(mainRayParams.rayStart, mainRayParams.rayDir, earthCenter, cuMinHeight);
                float cuRayLenTop = raySphereIntersectNearest(mainRayParams.rayStart, mainRayParams.rayDir, earthCenter, cuMaxHeight);

                bool inLayer = abs(cuHeightDiff) < SETTING_CLOUDS_CU_THICKNESS * 0.5;
                float cuOrigin2RayStart = inLayer ? 0.0 : min(cuRayLenBot, cuRayLenTop);

                uint cuFlag = uint(sign(cuHeightDiff) == sign(mainRayParams.rayDir.y)) | uint(inLayer);
                cuFlag &= uint(cuOrigin2RayStart >= 0.0);

                if (bool(cuFlag)) {
                    CloudSSHistoryData historyData = clouds_ss_historyData_init();
                    clouds_ss_historyData_unpack(packedData, historyData);
                    bool above = inLayer || cuHeightDiff < 0.0;
                    ScatteringResult layerResult = ScatteringResult(
                        historyData.transmittance,
                        historyData.inScattering
                    );
                    compositeResult = scatteringResult_blendLayer(
                        compositeResult,
                        layerResult,
                        above
                    );
                }
            }
            #endif

            {
                ScatteringResult layerResult = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyViewLutParams, 2.0);
                bool above = skyViewLutParams.viewHeight >= atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS * 0.5;
                compositeResult = scatteringResult_blendLayer(compositeResult, layerResult, above);
            }

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
                        0u
                    );
                    CloudRaymarchStepState stepState = clouds_raymarchStepState_init(layerParam);
                    float sampleDensity = clouds_ci_density(stepState.position.xyz);
                    CloudRaymarchAccumState ciAccum = clouds_raymarchAccumState_init();
                    if (sampleDensity > DENSITY_EPSILON) {
                        clouds_computeLighting(
                            atmosphere,
                            renderParams,
                            layerParam,
                            stepState,
                            sampleDensity,
                            sampleDensity * 0.5,
                            vec3(0.0),
                            ciAccum
                        );
                    };

                    ScatteringResult layerResult = ScatteringResult(
                        ciAccum.totalTransmittance,
                        ciAccum.totalInSctr
                    );
                    bool aboveFlag = ciHeighDiff < 0.0;
                    compositeResult = scatteringResult_blendLayer(compositeResult, layerResult, aboveFlag);
                }
            }
            #endif

            {
                ScatteringResult layerResult = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyViewLutParams, 3.0);
                bool above = skyViewLutParams.viewHeight >= atmosphere.bottom + SETTING_CLOUDS_CI_HEIGHT;
                compositeResult = scatteringResult_blendLayer(compositeResult, layerResult, above);
            }
        } else {
            CloudSSHistoryData newHistoryData = clouds_ss_historyData_init();
            uvec4 packedData = uvec4(0u);
            clouds_ss_historyData_pack(packedData, newHistoryData);
            history_lowCloud_store(texelPos, packedData);
        }
    } else {
        CloudSSHistoryData newHistoryData = clouds_ss_historyData_init();
        uvec4 packedData = uvec4(0u);
        clouds_ss_historyData_pack(packedData, newHistoryData);
        history_lowCloud_store(texelPos, packedData);
    }

    return compositeResult;
}