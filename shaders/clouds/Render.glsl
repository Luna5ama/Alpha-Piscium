#include "Common.glsl"
#include "Cirrus.glsl"
#include "/atmosphere/Common.glsl"

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

    CloudRaymarchParameters params;
    params.rayStart = atmosphere_viewToAtm(atmosphere, originView);

    if (endView.z == -65536.0) {
        params.rayStart.y = max(params.rayStart.y, atmosphere.bottom + 0.5);
        vec3 earthCenter = vec3(0.0);

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

    vec3 intergralResult = vec3(0.0, 0.0, 1.0); // Sun scattering, sky scattering, transmittance

    float cirrusCloudHeightDiff = atmosphere.bottom + CIRRUS_CLOUD_HEIGHT - params.rayStartHeight;
    if (cirrusCloudHeightDiff > 0.0 && rayDir.y > 0.0) {
        float rayLen = cirrusCloudHeightDiff / rayDir.y;
        vec3 rayPos = params.rayStart + rayDir * rayLen;

        float density = clouds_cirrus_density(rayPos);
        float opticalDepth = density * 0.1;
        float transmittance = exp(-opticalDepth);

        intergralResult.x += density;
        intergralResult.z *= transmittance;
    }

    intergralResult.x *= smoothstep(0.0, 0.1, rayDir.y);

//    outputColor.rgb *= intergralResult.z;
    outputColor.rgb += intergralResult.x * 100.0;
}