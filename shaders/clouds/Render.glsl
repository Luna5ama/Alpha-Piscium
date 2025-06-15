#include "Common.glsl"
#include "Cirrus.glsl"
#include "/atmosphere/Common.glsl"
#include "/util/Celestial.glsl"

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

    CloudRenderParams renderParams = cloudRenderParams_init(params, uval_shadowLightDirWorld);

    CloudRaymarchAccumState accumState = clouds_raymarchAccumState_init();

    float cirrusHeight = atmosphere.bottom + CIRRUS_CLOUD_HEIGHT;
    float cirrusCloudHeightDiff = cirrusHeight - params.rayStartHeight;
    float cirrusRayLen = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom + CIRRUS_CLOUD_HEIGHT);
    uint cirrusFlag = uint(sign(cirrusCloudHeightDiff) == sign(rayDir.y)) & uint(cirrusRayLen > 0.0);

    if (bool(cirrusFlag)) {
        vec3 rayPos = params.rayStart + rayDir * cirrusRayLen;
        float sampleDensity = clouds_cirrus_density(rayPos);

        if (sampleDensity > DENSITY_EPSILON) {
            CloudRaymarchStepState stepState = clouds_raymarchStepState_init(rayPos, sampleDensity);
            CloudParticpatingMedium cirrusMedium = clouds_cirrus_medium(renderParams);
            clouds_computeLighting(
                atmosphere,
                renderParams,
                cirrusMedium,
                stepState,
                0.5,
                accumState
            );
        }
    }

    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));
    vec3 lightIlluminance = mix(MOON_ILLUMINANCE, SUN_ILLUMINANCE * PI, shadowIsSun);

    outputColor.rgb += accumState.totalInSctr * lightIlluminance;
    outputColor.rgb *= accumState.totalTransmittance;
}