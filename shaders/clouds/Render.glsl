#include "Common.glsl"
#include "Cirrus.glsl"
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

    float cirrusHeight = atmosphere.bottom + SETTING_CLOUDS_CI_HEIGHT;
    float cirrusCloudHeightDiff = cirrusHeight - params.rayStartHeight;
    float cirrusRayLen = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom + SETTING_CLOUDS_CI_HEIGHT);
    uint cirrusFlag = uint(sign(cirrusCloudHeightDiff) == sign(rayDir.y)) & uint(cirrusRayLen > 0.0);

    if (bool(cirrusFlag)) {
        vec3 rayPos = params.rayStart + rayDir * cirrusRayLen;
        float sampleDensity = clouds_cirrus_density(rayPos);
        vec3 ambientIrradiance = texture(usam_cloudsAmbLUT, vec3(ambLutUV, 0.5 / 3.0)).rgb;
        CloudRaymarchLayerParam layerParam = clouds_raymarchLayerParam_init(
            clouds_cirrus_medium(renderParams.LDotV),
            1.0,
            ambientIrradiance
        );

        if (sampleDensity > DENSITY_EPSILON) {
            CloudRaymarchStepState stepState = clouds_raymarchStepState_init(rayPos, sampleDensity);
            CloudParticpatingMedium cirrusMedium = clouds_cirrus_medium(renderParams.LDotV);
            clouds_computeLighting(
                atmosphere,
                renderParams,
                layerParam,
                stepState,
                accumState
            );
        }
    }

    outputColor.rgb += accumState.totalInSctr;
    outputColor.rgb *= accumState.totalTransmittance;
}