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

layout(rgba32ui) uniform restrict writeonly uimage2D uimg_csrgba32ui;

const float DENSITY_EPSILON = 0.0001;

struct SkyViewLutParams {
    bool intersectGround;
    float viewZenithCosAngle;
    float sunViewCosAngle;
    float moonViewCosAngle;
    float viewHeight;
};

ScatteringResult _atmospherics_sampleSkyViewLUT(AtmosphereParameters atmosphere, SkyViewLutParams params, float layerIndex) {
    return atmospherics_air_lut_sampleSkyView(
        atmosphere,
        params.intersectGround,
        params.viewZenithCosAngle,
        params.sunViewCosAngle,
        params.moonViewCosAngle,
        params.viewHeight,
        layerIndex
    );
}

ScatteringResult atmospherics_skyComposite(ivec2 texelPos) {
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
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
    mainRayParams.rayStart = atmosphere_viewToAtmNoClamping(atmosphere, originView);
    const vec3 earthCenter = vec3(0.0);

    ScatteringResult compositeResult = scatteringResult_init();

    {
        vec3 rayDir = viewDirWorld;
        if (endView.z == -65536.0) {
            // Check if ray origin is outside the atmosphere
            if (length(mainRayParams.rayStart) > atmosphere.top) {
                float tTop = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, atmosphere.top);
                if (tTop < 0.0) {
                    return compositeResult;// No intersection with atmosphere: stop right away
                }
                mainRayParams.rayStart += rayDir * (tTop + 0.001);
            }

            float clampedBottom = min(atmosphere.bottom, length(mainRayParams.rayStart) - 0.001);
            float tBottom = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, clampedBottom);
            float tTop = raySphereIntersectNearest(mainRayParams.rayStart, rayDir, earthCenter, atmosphere.top);
            float rayLen = 0.0;

            if (tBottom < 0.0) {
                if (tTop < 0.0) {
                    return compositeResult;// No intersection with earth nor atmosphere: stop right away
                } else {
                    rayLen = tTop;
                }
            } else {
                if (tTop > 0.0) {
                    rayLen = min(tTop, tBottom);
                }
            }

            mainRayParams.rayEnd = mainRayParams.rayStart + rayDir * rayLen;
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

    vec2 jitters = rand_stbnVec2(texelPos, frameCounter);
    vec3 viewDir = mainRayParams.rayDir;
    vec2 ambLutUV = cloods_amblut_uv(viewDir, jitters);

    SkyViewLutParams skyViewLutParams;
    {
        vec3 rayStart = atmosphere_viewToAtm(atmosphere, vec3(0.0));
        float viewHeight = length(rayStart);
        vec3 upVector = rayStart / viewHeight;

        vec3 rayEndView = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
        vec3 rayDir = normalize(mat3(gbufferModelViewInverse) * rayEndView);

        float viewZenithCosAngle = dot(rayDir, upVector);

        const vec3 earthCenter = vec3(0.0);
        float tBottom = raySphereIntersectNearest(rayStart, rayDir, earthCenter, atmosphere.bottom);

        vec3 sideVector = normalize(cross(upVector, rayDir));		// assumes non parallel vectors
        vec3 forwardVector = normalize(cross(sideVector, upVector));	// aligns toward the sun light but perpendicular to up vector

        vec2 sunOnPlane = vec2(dot(uval_sunDirWorld, forwardVector), dot(uval_sunDirWorld, sideVector));
        sunOnPlane = normalize(sunOnPlane);
        float sunViewCosAngle = sunOnPlane.x;

        vec2 moonOnPlane = vec2(dot(uval_moonDirWorld, forwardVector), dot(uval_moonDirWorld, sideVector));
        moonOnPlane = normalize(moonOnPlane);
        float moonViewCosAngle = moonOnPlane.x;

        float horizonZenthCosAngle = -sqrt(1.0 - pow2(atmosphere.bottom / viewHeight));
        bool intersectGround = viewZenithCosAngle < (horizonZenthCosAngle);

        skyViewLutParams = SkyViewLutParams(
            intersectGround,
            viewZenithCosAngle,
            sunViewCosAngle,
            moonViewCosAngle,
            viewHeight
        );
    }

    if (viewZ == -65536.0) {
        {
            ScatteringResult layerResult = _atmospherics_sampleSkyViewLUT(atmosphere, skyViewLutParams, 1.0);
            compositeResult = scatteringResult_blendLayer(compositeResult, layerResult, true);
        }

        #ifdef SETTING_CLOUDS_CU
        {
            uvec4 packedData = texelFetch(usam_csrgba32ui, csrgba32ui_temp2_texelToTexel(texelPos), 0);
            imageStore(uimg_csrgba32ui, clouds_ss_history_texelToTexel(texelPos), packedData);

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
            ScatteringResult layerResult = _atmospherics_sampleSkyViewLUT(atmosphere, skyViewLutParams, 2.0);
            bool above = skyViewLutParams.viewHeight >= atmosphere.bottom + SETTING_CLOUDS_CU_HEIGHT;
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
                    1.0
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
                        vec3(1.0),
                        ciAccum
                    );
                }

                const float TRANSMITTANCE_DECAY = 10.0;
                ciAccum.totalTransmittance = pow(ciAccum.totalTransmittance, vec3(exp2(-ciOrigin2RayOffset * 0.1)));
                ciAccum.totalInSctr *= exp2(-pow2(ciOrigin2RayOffset) * 0.002);

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
            ScatteringResult layerResult = _atmospherics_sampleSkyViewLUT(atmosphere, skyViewLutParams, 3.0);
            bool above = skyViewLutParams.viewHeight >= atmosphere.bottom + SETTING_CLOUDS_CI_HEIGHT;
            compositeResult = scatteringResult_blendLayer(compositeResult, layerResult, above);
        }
    }

//    {
//        ScatteringResult layerResult;
//        #ifndef SETTING_DEPTH_BREAK_CORRECTION
//        unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, layerResult);
//        #else
//        bool isDepthBreak = !unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, layerResult);
//        uvec4 balllot = subgroupBallot(isDepthBreak);
//        uint correctionCount = subgroupBallotBitCount(balllot);
//        uint writeIndexBase = 0u;
//        if (subgroupElect()) {
//            writeIndexBase = atomicAdd(global_dispatchSize1.w, correctionCount);
//            uint totalCount = writeIndexBase + correctionCount;
//            atomicMax(global_dispatchSize1.x, (totalCount | 0x3Fu) >> 6u);
//        }
//        writeIndexBase = subgroupBroadcastFirst(writeIndexBase);
//        if (isDepthBreak) {
//            uint writeIndex = writeIndexBase + subgroupBallotExclusiveBitCount(balllot);
//            uint texelPosEncoded = packUInt2x16(uvec2(texelPos));
//            indirectComputeData[writeIndex] = texelPosEncoded;
//            layerResult = scatteringResult_init();
//        }
//        #endif
//        compositeResult = scatteringResult_blendLayer(compositeResult, layerResult, true);
//    }

    return compositeResult;
}