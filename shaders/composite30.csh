#version 460 compatibility

#include "/atmosphere/UnwarpEpipolar.glsl"
#include "/atmosphere/Scattering.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp2;

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_translucentColor;

layout(rgba16f) restrict uniform image2D uimg_main;

void applyAtmosphere(vec2 screenPos, vec3 viewPos, float viewZ, inout vec4 outputColor) {
    ScatteringResult sctrResult;

    #ifndef SETTING_DEPTH_BREAK_CORRECTION
    unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, sctrResult);
    #else
    if (!unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, sctrResult)) {
        float ignValue = rand_IGN(texelPos, frameCounter);
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        sctrResult = computeSingleScattering(atmosphere, vec3(0.0), viewPos, ignValue);
    }
    #endif

    outputColor.rgb *= sctrResult.transmittance;
    outputColor.rgb += sctrResult.inScattering;
}

void updateMoments(vec3 colorSRGB, inout vec3 sum, inout vec3 sqSum) {
    vec3 color = colors_SRGBToYCoCg(colorSRGB);
    sum += color;
    sqSum += color * color;
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);

        GBufferData gData;
        gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), gData);
        gbufferData2_unpack(texelFetch(usam_gbufferData8UN, texelPos, 0), gData);
        Material material = material_decode(gData);

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);

        vec3 curr3x3Avg = vec3(0.0);
        vec3 curr3x3SqAvg = vec3(0.0);
        updateMoments(texelFetchOffset(usam_temp2, texelPos, 0, ivec2(-1, 0)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp2, texelPos, 0, ivec2(1, 0)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp2, texelPos, 0, ivec2(0, -1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp2, texelPos, 0, ivec2(0, 1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp2, texelPos, 0, ivec2(-1, -1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp2, texelPos, 0, ivec2(1, -1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp2, texelPos, 0, ivec2(-1, 1)).rgb, curr3x3Avg, curr3x3SqAvg);
        updateMoments(texelFetchOffset(usam_temp2, texelPos, 0, ivec2(1, 1)).rgb, curr3x3Avg, curr3x3SqAvg);
        curr3x3Avg /= 9.0;
        curr3x3SqAvg /= 9.0;

        vec3 centerGI = texelFetch(usam_temp2, texelPos, 0).rgb;

        // Ellipsoid intersection clipping by Marty
        vec3 centerGIYCoCg = colors_SRGBToYCoCg(centerGI);
        vec3 stddev = sqrt(curr3x3SqAvg - curr3x3Avg * curr3x3Avg);
        vec3 delta = centerGIYCoCg - curr3x3Avg;
        const float clippingEps = 0.00001;
        delta /= max(1.0, length(delta / (stddev + clippingEps)));
        centerGIYCoCg = curr3x3Avg + delta;
        centerGI.rgb = colors_YCoCgToSRGB(centerGIYCoCg);

        outputColor.rgb += centerGI.rgb * material.albedo;
        applyAtmosphere(screenPos, viewPos, viewZ, outputColor);

        float albedoLuminance = all(equal(gData.albedo, vec3(0.0))) ? 0.1 : colors_srgbLuma(material.albedo);
        float luminanceC = colors_srgbLuma(outputColor.rgb) / albedoLuminance;
        vec4 translucentColorSample = texelFetch(usam_translucentColor, texelPos, 0);
        outputColor.rgb = mix(outputColor.rgb, translucentColorSample.rgb * luminanceC, translucentColorSample.a);

        imageStore(uimg_main, texelPos, outputColor);
    }
}