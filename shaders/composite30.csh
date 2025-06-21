#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/atmosphere/UnwarpEpipolar.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp2;

uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_translucentColor;

layout(rgba16f) restrict uniform image2D uimg_main;

void applyAtmosphere(vec2 screenPos, vec3 viewPos, float viewZ, inout vec4 outputColor) {
    ScatteringResult sctrResult;

    #ifndef SETTING_DEPTH_BREAK_CORRECTION
    unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, sctrResult);
    #else
    bool isDepthBreak = !unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, sctrResult);
    uvec4 balllot = subgroupBallot(isDepthBreak);
    uint correctionCount = subgroupBallotBitCount(balllot);
    uint writeIndexBase = 0u;
    if (subgroupElect()) {
        writeIndexBase = atomicAdd(global_dispatchSize1.w, correctionCount);
        uint totalCount = writeIndexBase + correctionCount;
        atomicMax(global_dispatchSize1.x, (totalCount | 0x3Fu) >> 6u);
    }
    writeIndexBase = subgroupBroadcastFirst(writeIndexBase);
    if (isDepthBreak) {
        uint writeIndex = writeIndexBase + subgroupBallotExclusiveBitCount(balllot);
        uint texelPosEncoded = packUInt2x16(uvec2(texelPos));
        indirectComputeData[writeIndex] = texelPosEncoded;
        sctrResult = scatteringResult_init();
    }
    #endif

    outputColor.rgb *= sctrResult.transmittance;
    outputColor.rgb += sctrResult.inScattering;
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);

        vec3 albedo = colors_srgbToLinear(texelFetch(usam_gbufferData8UN, texelPos, 0).rgb);

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

        vec3 giRadiance = texelFetch(usam_temp2, texelPos, 0).rgb;

        outputColor.rgb += giRadiance.rgb * albedo;
        applyAtmosphere(screenPos, viewPos, viewZ, outputColor);

        imageStore(uimg_main, texelPos, outputColor);
    }
}