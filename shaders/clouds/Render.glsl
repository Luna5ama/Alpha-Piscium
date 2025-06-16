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

    vec3 rayDir = viewDirWorld;

    float ignValue = rand_IGN(texelPos, frameCounter);
    AtmosphereParameters atmosphere = getAtmosphereParameters();

    CloudRayParams params;
    params.rayStart = atmosphere_viewToAtm(atmosphere, originView);

    vec3 earthCenter = vec3(0.0);
    if (endView.z == -65536.0) {
        params.rayStart.y = max(params.rayStart.y, atmosphere.bottom + 0.5);

        // Check if ray origin is outside the atmosphere
        if (length(params.rayStart) > atmosphere.top) {
            float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
            if (tTop < 0.0) {
                return;// No intersection with atmosphere: stop right away
            }
            params.rayStart += rayDir * (tTop + 0.001);
        }

        float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom);
        float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
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

        params.rayEnd = params.rayStart + rayDir * rayLen;
    } else {
        params.rayEnd = atmosphere_viewToAtm(atmosphere, endView);
        return;
    }

    params.rayDir = normalize(params.rayEnd - params.rayStart);
    params.rayStartHeight = length(params.rayStart);
    params.rayEndHeight = length(params.rayEnd);

    float sunAngleWarped = fract(sunAngle + 0.25);
    float sunLightFactor = smoothstep(0.23035, 0.24035, sunAngleWarped);
    sunLightFactor *= smoothstep(0.76965, 0.75965, sunAngleWarped);
    sunLightFactor *= step(0.5, sunLightFactor);
    vec3 lightDir = mix(uval_moonDirWorld, uval_sunDirWorld, sunLightFactor);
    vec3 lightIlluminance = mix(MOON_ILLUMINANCE, SUN_ILLUMINANCE * PI, sunLightFactor);
    CloudRenderParams renderParams = cloudRenderParams_init(params, lightDir, lightIlluminance);

    CloudRaymarchAccumState accumState = clouds_raymarchAccumState_init();

    vec3 viewDir = -params.rayDir;
    vec2 ambLutUV = coords_equirectanglarForwardHorizonBoost(viewDir);

    #ifdef SETTING_CLOUDS_CU
    {
        float cuHeight = atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT;
        float cuMinHeight = cuHeight - SETTING_CLOUDS_CU_THICKNESS * 0.5;
        float cuMaxHeight = cuHeight + SETTING_CLOUDS_CU_THICKNESS * 0.5;
        float cuHeightDiff = cuHeight - params.rayStartHeight;
        float cuRayLenBottom = raySphereIntersectNearest(params.rayStart, params.rayDir, earthCenter, cuMinHeight);
        float cuRayLenTop = raySphereIntersectNearest(params.rayStart, params.rayDir, earthCenter, cuMaxHeight);
        float cuStartRayLen = abs(cuHeightDiff) < SETTING_CLOUDS_CU_THICKNESS * 0.5 ? 0.0 :
            min(cuRayLenBottom, cuRayLenTop);
        float cuEndRayLen = max(cuRayLenBottom, cuRayLenTop);
        uint cuFlag = uint(sign(cuHeightDiff) == sign(rayDir.y)) & uint(cuStartRayLen >= 0.0);

        if (bool(cuFlag)) {
            #define CLOUDS_CU_RAYMARCH_STEP 64
            #define CLOUDS_CU_RAYMARCH_STEP_RCP rcp(float(CLOUDS_CU_RAYMARCH_STEP))
            vec3 cuRayStart = params.rayStart + rayDir * cuStartRayLen;
            vec3 cuRayEnd = params.rayStart + rayDir * cuEndRayLen;
            vec3 cuRayStepDelta = (cuRayEnd - cuRayStart) * CLOUDS_CU_RAYMARCH_STEP_RCP;
            float cuRayStepLength = length(cuRayStepDelta);

            vec3 ambientIrradiance = texture(usam_cloudsAmbLUT, vec3(ambLutUV, 1.5 / 6.0)).rgb;
            CloudParticpatingMedium cuMedium = clouds_cu_medium(renderParams.LDotV);
            CloudRaymarchLayerParam layerParam = clouds_raymarchLayerParam_init(
                cuMedium,
                ambientIrradiance
            );

            CloudRaymarchAccumState cuAccum = clouds_raymarchAccumState_init();

            for (uint stepIndex = 0; stepIndex < CLOUDS_CU_RAYMARCH_STEP; ++stepIndex) {
                float stepIndexF = float(stepIndex) + 0.5;
                vec3 rayPos = cuRayStart + stepIndexF * cuRayStepDelta;
                float sampleDensity = clouds_cu_density(rayPos) * 5.0;

                if (sampleDensity > DENSITY_EPSILON) {
                    CloudRaymarchStepState stepState = clouds_raymarchStepState_init(
                        cuRayStepLength,
                        rayPos,
                        sampleDensity
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
        float ciHeighDiff = ciHeight - params.rayStartHeight;
        float ciRayLen = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, ciHeight);
        uint ciFlag = uint(sign(ciHeighDiff) == sign(rayDir.y)) & uint(ciRayLen > 0.0);

        if (bool(ciFlag)) {
            vec3 rayPos = params.rayStart + rayDir * ciRayLen;
            float sampleDensity = clouds_ci_density(rayPos);
            vec3 ambientIrradiance = texture(usam_cloudsAmbLUT, vec3(ambLutUV, 3.5 / 6.0)).rgb;
            CloudParticpatingMedium ciMedium = clouds_ci_medium(renderParams.LDotV);
            CloudRaymarchLayerParam layerParam = clouds_raymarchLayerParam_init(
                ciMedium,
                ambientIrradiance
            );

            CloudRaymarchAccumState ciAccum = clouds_raymarchAccumState_init();

            if (sampleDensity > DENSITY_EPSILON) {
                CloudRaymarchStepState stepState = clouds_raymarchStepState_init(
                    1.0,
                    rayPos,
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