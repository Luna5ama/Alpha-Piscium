#extension GL_KHR_shader_subgroup_basic : enable

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

#include "Common.glsl"
#include "Cirrus.glsl"
#include "Cumulus.glsl"
#include "ss/Common.glsl"
#include "/util/Celestial.glsl"
#include "/util/Math.glsl"
#include "/util/Morton.glsl"

#define UPSCALE_FACTOR 4

#if UPSCALE_FACTOR == 1
ivec2 renderSize = global_mipmapSizesI[0];
#define UPSCALE_BLOCK_SIZE 1
#elif UPSCALE_FACTOR == 2
ivec2 renderSize = global_mipmapSizesI[1];
#define UPSCALE_BLOCK_SIZE 4
#elif UPSCALE_FACTOR == 4
ivec2 renderSize = global_mipmapSizesI[2];
#define UPSCALE_BLOCK_SIZE 16
#elif UPSCALE_FACTOR == 8
ivec2 renderSize = global_mipmapSizesI[3];
#define UPSCALE_BLOCK_SIZE 64
#endif

uniform sampler2D usam_gbufferViewZ;
uniform sampler3D usam_cloudsAmbLUT;

layout(rgba32ui) uniform restrict uimage2D uimg_csrgba32ui;

vec2 getTexelPos1x1(ivec2 texelPosDownScale) {
    vec2 texelPos1x1F = vec2(texelPosDownScale * UPSCALE_FACTOR);
    vec2 offset = rand_r2Seq2(frameCounter);
    offset *= UPSCALE_FACTOR;
    offset = mod(offset, vec2(UPSCALE_FACTOR));
    return clamp(texelPos1x1F + offset, vec2(0.5), global_mainImageSize - 0.5);
}

void writePixel(ivec2 outputTexelPos, CloudRaymarchAccumState accumState, float weight) {
    if (weight > 0.0) {
        ivec2 actualTexelPos = clouds_ss_history_texelToTexel(outputTexelPos);
        CloudSSHistoryData prevData = clouds_ss_historyData_init();
        clouds_ss_historyData_unpack(imageLoad(uimg_csrgba32ui, actualTexelPos), prevData);

        CloudSSHistoryData newData = clouds_ss_historyData_init();
        newData.hLen = min(prevData.hLen + weight, CLOUDS_SS_MAX_ACCUM);

        float alpha = weight / newData.hLen;
        newData.inScattering = mix(prevData.inScattering, accumState.totalInSctr, alpha);
        newData.transmittance = mix(prevData.transmittance, accumState.totalTransmittance, alpha);
        newData.viewZ = mix(prevData.viewZ, accumState.viewZ, alpha);

        uvec4 packedData = uvec4(0);
        clouds_ss_historyData_pack(packedData, newData);
        imageStore(uimg_csrgba32ui, actualTexelPos, packedData);
    }
}

void render(ivec2 texelPosDownScale) {
    vec2 texelPosF = getTexelPos1x1(texelPosDownScale);
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
    mainRayParams.rayStart = atmosphere_viewToAtm(atmosphere, originView);
    const vec3 earthCenter = vec3(0.0);

    float maxRayLen = 0.0;

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
    accumState.viewZ = maxRayLen;

    vec3 viewDir = -mainRayParams.rayDir;
    vec2 ambLutUV = coords_equirectanglarForwardHorizonBoost(viewDir);
    vec2 jitters = rand_stbnVec2(ivec2(texelPosF), frameCounter / UPSCALE_BLOCK_SIZE);

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
            #define CLOUDS_CU_DENSITY (256.0 * SETTING_CLOUDS_CU_DENSITY)

            float cuRayLen = max(cuRayLenBottom, cuRayLenTop) - cuOrigin2RayStart;

            vec3 ambientIrradiance = texture(usam_cloudsAmbLUT, vec3(ambLutUV, 1.5 / 6.0)).rgb;
            CloudParticpatingMedium cuMedium = clouds_cu_medium(renderParams.cosLightTheta);
            CloudRaymarchLayerParam layerParam = clouds_raymarchLayerParam_init(
                mainRayParams,
                cuMedium,
                ambientIrradiance,
                vec2(cuMinHeight, cuMaxHeight),
                cuOrigin2RayStart,
                cuRayLen,
                CLOUDS_CU_RAYMARCH_STEP_RCP
            );
            CloudRaymarchStepState stepState = clouds_raymarchStepState_init(layerParam, jitters.x);
            CloudRaymarchAccumState cuAccum = clouds_raymarchAccumState_init();

            for (uint stepIndex = 0; stepIndex < CLOUDS_CU_RAYMARCH_STEP; ++stepIndex) {
                if (stepState.position.w > cuRayLen) break;
                float heightFraction = linearStep(cuMinHeight, cuMaxHeight, stepState.height);
                float sampleDensity = 0.0;
                if (clouds_cu_density(stepState.position.xyz, heightFraction, sampleDensity)) {
                    sampleDensity *= CLOUDS_CU_DENSITY;

                    float lightRayTotalDensity = 0.0;
                    {
                        float lightRayLen = 0.5;
                        float lightRayStepDelta = lightRayLen * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP;
                        vec3 lightRayPos = stepState.position.xyz;
                        for (uint lightStepIndex = 0; lightStepIndex < CLOUDS_CU_LIGHT_RAYMARCH_STEP; ++lightStepIndex) {
                            lightRayPos += renderParams.lightDir * lightRayStepDelta;
                            vec3 lightRaySamplePos = lightRayPos + renderParams.lightDir * (jitters.y - 0.5);
                            float lightSampleHeight = length(lightRaySamplePos);
                            if (lightSampleHeight > cuMaxHeight) break;
                            float lightHeightFraction = linearStep(cuMinHeight, cuMaxHeight, lightSampleHeight);
                            float lightSampleDensity = 0.0;
                            if (clouds_cu_density(lightRaySamplePos, lightHeightFraction, lightSampleDensity)) {
                                lightRayTotalDensity += lightSampleDensity * lightRayStepDelta;
                            }
                            lightRayStepDelta *= 1.5;
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

                clouds_raymarchStepState_update(stepState);
            }

            float aboveFlag = float(cuHeightDiff < 0.0);
            accumState.totalInSctr = mix(
                accumState.totalInSctr + cuAccum.totalInSctr * accumState.totalTransmittance, // Below
                cuAccum.totalInSctr * cuAccum.totalTransmittance + cuAccum.totalInSctr, // Above
                aboveFlag
            );
            accumState.totalTransmittance *= cuAccum.totalTransmittance;
            accumState.viewZ = min(accumState.viewZ, cuAccum.viewZ);
        }
    }

    vec2 centerPixel = texelPosF - 0.5;
    vec2 centerPixelOrigin = floor(centerPixel);
    vec2 gatherTexelPos = centerPixelOrigin + 1.0;
    vec2 pixelPosFract = centerPixel - centerPixelOrigin;

    vec2 bilinearWeights2 = pixelPosFract;
    vec4 blinearWeights4;
    blinearWeights4.yz = bilinearWeights2.xx;
    blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
    blinearWeights4.xy *= bilinearWeights2.yy;
    blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;
    blinearWeights4 *= 4.0;

    ivec2 centerPixelOriginI = ivec2(centerPixelOrigin);

    writePixel(centerPixelOriginI, accumState, blinearWeights4.w);
    writePixel(centerPixelOriginI + ivec2(1, 0), accumState, blinearWeights4.z);
    writePixel(centerPixelOriginI + ivec2(0, 1), accumState, blinearWeights4.x);
    writePixel(centerPixelOriginI + ivec2(1, 1), accumState, blinearWeights4.y);
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