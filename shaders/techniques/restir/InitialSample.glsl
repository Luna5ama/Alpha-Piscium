#ifndef INCLUDE_techniques_restir_InitialSample_glsl
#define INCLUDE_techniques_restir_InitialSample_glsl a

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/Common.glsl"
#include "Irradiance.glsl"

struct restir_InitialSampleData {
    vec4 directionAndLength;
    vec3 hitRadiance;
};

restir_InitialSampleData restir_initialSampleData_init() {
    restir_InitialSampleData data;
    data.directionAndLength = vec4(0.0, 0.0, 0.0, -1.0);
    data.hitRadiance = vec3(0.0);
    return data;
}

uvec4 restir_initialSampleData_pack(restir_InitialSampleData data) {
    uvec4 packedData;
//    packedData.x = packSnorm2x16(data.directionAndLength.xy);
//    packedData.y = (packSnorm2x16(vec2(data.directionAndLength.z, 0.0)) & 0xFFFFu) | (packHalf2x16(vec2(0.0, data.hitRadiance.r)) & 0xFFFF0000u);
//    packedData.z = packHalf2x16(data.hitRadiance.gb);
//    packedData.w = floatBitsToUint(data.directionAndLength.w);
    packedData.x = nzpacking_packNormalOct32(data.directionAndLength.xyz);
    packedData.y = floatBitsToUint(data.directionAndLength.w);
    packedData.zw = packHalf4x16(vec4(data.hitRadiance, 0.0));
    return packedData;
}

restir_InitialSampleData restir_initialSampleData_unpack(uvec4 packedData) {
    restir_InitialSampleData data;
    data.directionAndLength.xyz = nzpacking_unpackNormalOct32(packedData.x);
    data.directionAndLength.w = uintBitsToFloat(packedData.y);
    vec4 hitRadianceAndPadding = unpackHalf4x16(packedData.zw);
    data.hitRadiance = hitRadianceAndPadding.rgb;
//    data.directionAndLength.xy = unpackSnorm2x16(packedData.x);
//    data.directionAndLength.z = unpackSnorm2x16(packedData.y & 0xFFFFu).x;
//    data.hitRadiance.r = unpackHalf2x16(packedData.y & 0xFFFF0000u).y;
//    data.hitRadiance.gb = unpackHalf2x16(packedData.z);
//    data.directionAndLength.w = uintBitsToFloat(packedData.w);
    return data;
}

restir_InitialSampleData restir_initialSample_handleRayResult(SSTRay sstRay) {
    vec3 rayEndScreen = sstRay.pRayStart + sstRay.pRayDir * (sstRay.pRayVecLen * abs(sstRay.currT));
    vec3 rayOriginView = coords_screenToView(sstRay.pRayStart, global_camProjInverse);
    vec3 rayEndView = coords_screenToView(rayEndScreen, global_camProjInverse);
    vec3 rayDiffView = rayEndView - rayOriginView;
    float rayLengthView = length(rayDiffView);
    vec3 rayDirView = rayDiffView * rcp(rayLengthView);

    restir_InitialSampleData initialSample = restir_initialSampleData_init();
    initialSample.directionAndLength.xyz = rayDirView;
    initialSample.directionAndLength.w = rayLengthView;

    if (sstRay.currT <= -1.0) {
        // Miss
        vec3 rayWorldDir = coords_dir_viewToWorld(rayDirView);
        initialSample.hitRadiance = restir_irradiance_sampleIrradianceMiss(rayWorldDir);
        initialSample.directionAndLength.w = -1.0;
    } else {
        // Assert hit
        vec2 hitTexelPosF = floor(rayEndScreen.xy * uval_mainImageSize);
        ivec2 hitTexelPos = ivec2(hitTexelPosF);
        vec2 hitTexelCenter = hitTexelPosF + 0.5;
        vec2 hitScreenPos = hitTexelCenter * uval_mainImageSizeRcp;
        initialSample.hitRadiance = restir_irradiance_sampleIrradiance(sstRay.pRayOriginTexelPos, hitTexelPos, -rayDirView);
    }

    return initialSample;
}

#endif