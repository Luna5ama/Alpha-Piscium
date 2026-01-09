#ifndef INCLUDE_techniques_restir_InitialSample_glsl
#define INCLUDE_techniques_restir_InitialSample_glsl a

#include "Irradiance.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/Common.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

struct restir_InitialSampleData {
    vec4 directionAndLength;
    vec3 hitRadiance;
};

vec3 restir_initialSample_generateRayDir(ivec2 texelPos, vec3 geomNormal, mat3 tbn) {
    uvec4 randKey = uvec4(texelPos, 1919810u, RANDOM_FRAME);

    vec2 rand2 = hash_uintToFloat(hash_44_q3(randKey).zw);
    // vec2 rand2 = rand_stbnVec2(texelPos, RANDOM_FRAME);

    // vec4 sampleDirTangentAndPdf = rand_sampleInHemisphere(rand2);
    vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
    vec3 sampleDirView = normalize(tbn * sampleDirTangentAndPdf.xyz);

    // ivec2 stbnPos = texelPos + ivec2(rand_r2Seq2(RANDOM_FRAME / 64u) * vec2(128, 128));
    // vec3 sampleDirTangent = rand_stbnUnitVec3Cosine(stbnPos, RANDOM_FRAME);
    // vec3 sampleDirView = normalize(material.tbn * sampleDirTangent);

    if (dot(sampleDirView, geomNormal) < 0.0) {
        sampleDirView = reflect(sampleDirView, geomNormal);
    }

    return sampleDirView;
}

restir_InitialSampleData restir_initalSample_restoreData(ivec2 texelPos, float viewZ, vec3 geomNormal, Material selfMaterial, float hitDistance) {
    restir_InitialSampleData initialSampleData;
    vec3 rayDirView = restir_initialSample_generateRayDir(texelPos, geomNormal, selfMaterial.tbn);
    initialSampleData.directionAndLength.xyz = rayDirView;
    initialSampleData.directionAndLength.w = hitDistance;
    vec2 rayOriginScreenXY = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 rayOriginView = coords_toViewCoord(rayOriginScreenXY, viewZ, global_camProjInverse);

    if (hitDistance <= -1.0) {
        // Miss
        vec3 rayOriginScene = coords_pos_viewToWorld(rayOriginView, gbufferModelViewInverse);
        vec3 rayWorldDir = coords_dir_viewToWorld(rayDirView);
        initialSampleData.hitRadiance = restir_irradiance_sampleIrradianceMiss(texelPos, rayOriginScene, rayWorldDir);
    } else {
        vec3 rayEndView = rayOriginView + rayDirView * hitDistance;
        vec3 rayEndScreen = coords_viewToScreen(rayEndView, global_camProj);

        // Assert hit
        vec2 hitTexelPosF = floor(rayEndScreen.xy * uval_mainImageSize);
        ivec2 hitTexelPos = ivec2(hitTexelPosF);
        vec2 hitTexelCenter = hitTexelPosF + 0.5;
        vec2 hitScreenPos = hitTexelCenter * uval_mainImageSizeRcp;
        initialSampleData.hitRadiance = restir_irradiance_sampleIrradiance(texelPos, selfMaterial, hitTexelPos, -rayDirView);
    }

    return initialSampleData;
}

float restir_initialSample_handleRayResult(SSTRay sstRay) {
    float hitDistance = -1.0;
    if (sstRay.currT > -1.0) {
        vec3 rayEndScreen = sstRay.pRayStart + sstRay.pRayDir * (sstRay.pRayVecLen * abs(sstRay.currT));
        vec3 rayOriginView = coords_screenToView(sstRay.pRayStart, global_camProjInverse);
        vec3 rayEndView = coords_screenToView(rayEndScreen, global_camProjInverse);
        vec3 rayDiffView = rayEndView - rayOriginView;
        float rayLengthView = length(rayDiffView);
        hitDistance = rayLengthView;
    }
    return hitDistance;
}

#endif