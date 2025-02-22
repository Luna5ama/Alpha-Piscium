#include "/util/Coords.glsl"
#include "/util/GBuffers.glsl"
#include "/post/Dithering.glsl"

uniform sampler2D usam_main;
uniform usampler2D usam_gbufferData;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_taaLast;
uniform sampler2D usam_projReject;

in vec2 frag_texCoord;

/* RENDERTARGETS:0,15 */
layout(location = 0) out vec4 rt_out;
layout(location = 1) out vec4 rt_taaLast;

void updateNearMinMax(vec3 currColor, inout vec3 nearMin, inout vec3 nearMax) {
    nearMin = min(nearMin, currColor);
    nearMax = max(nearMax, currColor);
}

void main() {
    ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
    vec2 unjitteredTexCoord = frag_texCoord + global_taaJitterMat[3].xy;

    GBufferData gData;
    gbuffer_unpack(texelFetch(usam_gbufferData, intTexCoord, 0), gData);
    float isHand = float(gData.isHand);

    float viewZ = texelFetch(usam_gbufferViewZ, intTexCoord, 0).r;
    vec3 viewCoord = coords_toViewCoord(frag_texCoord, viewZ, gbufferProjectionInverse);
    vec4 worldCoord = gbufferModelViewInverse * vec4(viewCoord, 1.0);
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    vec4 prevWorldCoord = worldCoord;
    prevWorldCoord.xyz += cameraDelta;
    vec4 prevViewCoord = gbufferPrevModelView * prevWorldCoord;
    vec4 prevClipCoord = gbufferProjection * prevViewCoord;
    prevClipCoord /= prevClipCoord.w;
    vec2 prevTexCoord = prevClipCoord.xy * 0.5 + 0.5;
    prevTexCoord = mix(prevTexCoord, frag_texCoord, isHand);

    vec3 currColor = texture(usam_main, frag_texCoord).rgb;
    currColor = saturate(currColor);
    vec3 nearMin1 = currColor;
    vec3 nearMax1 = currColor;

    {
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(-1, 0)).rgb, nearMin1, nearMax1);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(1, 0)).rgb, nearMin1, nearMax1);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(0, -1)).rgb, nearMin1, nearMax1);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(0, 1)).rgb, nearMin1, nearMax1);
    }

    vec3 nearMin2 = nearMin1;
    vec3 nearMax2 = nearMax1;

    {
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(-1, -1)).rgb, nearMin2, nearMax2);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(1, 1)).rgb, nearMin2, nearMax2);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(1, -1)).rgb, nearMin2, nearMax2);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(-1, 1)).rgb, nearMin2, nearMax2);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(-2, 0)).rgb, nearMin2, nearMax2);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(2, 0)).rgb, nearMin2, nearMax2);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(0, -2)).rgb, nearMin2, nearMax2);
        updateNearMinMax(textureOffset(usam_main, unjitteredTexCoord, ivec2(0, 2)).rgb, nearMin2, nearMax2);
    }

    vec2 projReject = texelFetch(usam_projReject, intTexCoord, 0).rg;
    projReject = max(projReject, texelFetchOffset(usam_projReject, intTexCoord, 0, ivec2(-1, 0)).rg);
    projReject = max(projReject, texelFetchOffset(usam_projReject, intTexCoord, 0, ivec2(1, 0)).rg);
    projReject = max(projReject, texelFetchOffset(usam_projReject, intTexCoord, 0, ivec2(0, -1)).rg);
    projReject = max(projReject, texelFetchOffset(usam_projReject, intTexCoord, 0, ivec2(0, 1)).rg);

    float frustumTest = float(projReject.x > 0.0);
    float newPixel = float(projReject.y > 0.0);

    vec2 pixelPosDiff = (frag_texCoord - prevTexCoord) * textureSize(usam_main, 0).xy;
    float cameraSpeed = length(cameraDelta);
    float prevCameraSpeed = length(global_prevCameraDelta);
    float cameraSpeedDiff = abs(cameraSpeed - prevCameraSpeed);
    float pixelSpeed = length(pixelPosDiff);

    vec4 lastResult = texture(usam_taaLast, prevTexCoord);
    vec3 lastColor = saturate(lastResult.rgb);

    float clampRatio1 = 0.1;
    clampRatio1 += saturate(1.0 - lastResult.a);
    clampRatio1 += newPixel * 0.5;
    clampRatio1 += frustumTest * 0.5;
    clampRatio1 += pixelSpeed * 0.05;
    clampRatio1 += cameraSpeed * 0.1;
    clampRatio1 += cameraSpeedDiff * 8.0;
    clampRatio1 = saturate(clampRatio1);

    float clampRatio2 = 0.2;
    clampRatio2 += newPixel * 1.0;
    clampRatio2 += frustumTest * 1.0;
    clampRatio2 += pixelSpeed * 0.1;
    clampRatio2 += cameraSpeed * 0.5;
    clampRatio1 += cameraSpeedDiff * 32.0;
    clampRatio2 = saturate(clampRatio2);

    #ifndef SETTING_SCREENSHOT_MODE
    lastColor = mix(lastColor, clamp(lastColor, nearMin2, nearMax2), clampRatio2);
    lastColor = mix(lastColor, clamp(lastColor, nearMin1, nearMax1), clampRatio1);
    #endif

    float lastMixWeight = texture(usam_taaLast, frag_texCoord).a;

    float mixWeight = 0.95;
    mixWeight = mix(lastMixWeight, mixWeight, 0.5);

    #ifdef SETTING_SCREENSHOT_MODE
    float mixDecrease = 1.0;
    mixDecrease *= (1.0 - saturate(cameraSpeedDiff * 114.0));
    mixDecrease *= (1.0 - saturate(cameraSpeed * 69.0));
    mixDecrease *= (1.0 - saturate(pixelSpeed * 69.0));
    #else
    float mixDecrease = 1.0;
    mixDecrease *= (1.0 - saturate(cameraSpeedDiff * 16.0));
    mixDecrease *= (1.0 - saturate(cameraSpeed * 0.5));
    mixDecrease *= (1.0 - saturate(pixelSpeed * 0.05));
    mixDecrease = max(mixDecrease, 0.75);
    #endif

    mixWeight = mixWeight * mixDecrease;

    float finalMixWeight = mixWeight;
    finalMixWeight *= (1.0 - min(cameraSpeedDiff * 1.0, 0.5));

    #ifdef SETTING_SCREENSHOT_MODE
    finalMixWeight = clamp(finalMixWeight, 0.0, 0.99);
    #else
    finalMixWeight = clamp(finalMixWeight, 0.5, 0.99);
    #endif

    rt_out.rgb = mix(currColor, lastColor, finalMixWeight);
    rt_out.a = 1.0;
    dithering(gl_FragCoord.xy, rt_out.rgb);

    mixWeight = mix(lastMixWeight + 0.01, mixWeight, 0.05);

    #ifndef SETTING_SCREENSHOT_MODE
    mixWeight = saturate(mixWeight - isHand * 0.2);
    #endif

    rt_taaLast = vec4(rt_out.rgb, mixWeight);
}