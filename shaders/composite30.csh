#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/atmospherics/air/UnwarpEpipolar.glsl"
#include "/techniques/atmospherics/clouds//Render.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) restrict uniform image2D uimg_main;

ScatteringResult sampleSkyViewLUT(vec3 viewPos, float viewZ) {
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * global_mainImageSizeRcp;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

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
    vec2 sampleUV;
    ScatteringResult result = atmospherics_air_lut_sampleSkyView(
        atmosphere,
        intersectGround,
        viewZenithCosAngle,
        sunViewCosAngle,
        moonViewCosAngle,
        viewHeight,
        0.0
    );

    return result;
}

void applyAtmosphere(vec2 screenPos, vec3 viewPos, float viewZ, inout vec4 outputColor) {
    ScatteringResult sctrResult;

    if (viewZ == -65536.0f) {
        ScatteringResult skyView = sampleSkyViewLUT(viewPos, viewZ);
        outputColor.rgb *= skyView.transmittance;
        outputColor.rgb += skyView.inScattering;
    }
    renderCloud(texelPos, usam_gbufferViewZ, outputColor);

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

        vec3 albedo = colors_sRGB_decodeGamma(texelFetch(usam_gbufferData8UN, texelPos, 0).rgb);

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

        vec3 giRadiance = texelFetch(usam_temp2, texelPos, 0).rgb;

        outputColor.rgb += giRadiance.rgb * albedo;
        applyAtmosphere(screenPos, viewPos, viewZ, outputColor);

        imageStore(uimg_main, texelPos, outputColor);
    }
}