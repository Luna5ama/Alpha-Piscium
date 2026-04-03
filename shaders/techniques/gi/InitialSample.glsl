#ifndef INCLUDE_techniques_restir_InitialSample_glsl
#define INCLUDE_techniques_restir_InitialSample_glsl a

#include "Irradiance.glsl"
#include "/util/BSDF.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/Common.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

struct restir_InitialSampleData {
    vec4 directionAndLength;
    vec3 hitRadiance;
    float pdf;
};

// Stochastic VNDF/cosine sampling.
// Slot 0 (RANDOM_FRAME/64u)   → choice random (stbnVec1)
// Slot 1 (RANDOM_FRAME/64u+1) → direction random (stbnVec2 or stbnUnitVec3Cosine)
vec3 restir_initialSample_generateRayDir(ivec2 texelPos, vec3 geomNormal, vec3 V, Material material, out float pdf) {
    float roughness = material.roughness;
    vec3 wiTangent = material.tbnInv * V;
    float NdotV = max(wiTangent.z, 1e-5);

    // Specular bounce probability: F / (albedo*(1-F) + F)
    vec3 fresnelV = fresnel_evalMaterial(material, NdotV);
    float p_spec;
    if (material.metallic > 0.5) {
        p_spec = 1.0;
    } else {
        vec3 fresnelT = vec3(1.0) - fresnelV;
        vec3 totalEnergy = material.albedo * fresnelT + fresnelV;
        p_spec = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, fresnelV / max(totalEnergy, vec3(1e-5)));
    }
    p_spec = clamp(p_spec, 0.0, 1.0);

    float choiceRand = rand_stbnVec1(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u), RANDOM_FRAME);

    if (choiceRand < p_spec) {
        // VNDF specular sample
        vec2 xi = rand_stbnVec2(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 1u), RANDOM_FRAME);
        float pdf_ratio;
        vec3 wmTangent = bsdf_SphericalCapBoundedWithPDFRatio(xi, wiTangent, vec2(roughness), pdf_ratio);
        vec3 wrTangent = reflect(-wiTangent, wmTangent);
        vec3 sampleDirView = normalize(material.tbn * wrTangent);
        if (dot(sampleDirView, geomNormal) < 0.0) {
            sampleDirView = reflect(sampleDirView, geomNormal);
        }
        // VNDF reflection PDF with bounded-cap correction
        float NdotH = max(wmTangent.z, 1e-5);
        float a2 = roughness * roughness;
        float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D = a2 / (PI * d * d);
        float G1 = bsdf_smithG1(NdotV, roughness);
        float vndf_pdf = G1 * D / (4.0 * NdotV * max(pdf_ratio, 1e-5));
        pdf = p_spec * max(vndf_pdf, 1e-6);
        return sampleDirView;
    } else {
        // Cosine-weighted diffuse sample around shading normal
        vec3 sampleDirTangent = rand_stbnUnitVec3Cosine(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 1u), RANDOM_FRAME);
        vec3 sampleDirView = normalize(material.tbn * sampleDirTangent);
        if (dot(sampleDirView, geomNormal) < 0.0) {
            sampleDirView = reflect(sampleDirView, geomNormal);
        }
        float NDotL = max(sampleDirTangent.z, 0.0);
        pdf = (1.0 - p_spec) * max(NDotL * RCP_PI, 1e-6);
        return sampleDirView;
    }
}

restir_InitialSampleData restir_initalSample_restoreData(ivec2 texelPos, float viewZ, vec3 geomNormal, Material selfMaterial, float hitDistance) {
    restir_InitialSampleData initialSampleData;
    vec2 rayOriginScreenXY = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 rayOriginView = coords_toViewCoord(rayOriginScreenXY, viewZ, global_camProjInverse);
    vec3 V = normalize(-rayOriginView);

    float pdf;
    vec3 rayDirView = restir_initialSample_generateRayDir(texelPos, geomNormal, V, selfMaterial, pdf);
    initialSampleData.directionAndLength.xyz = rayDirView;
    initialSampleData.directionAndLength.w = hitDistance;
    initialSampleData.pdf = pdf;

    if (hitDistance <= -1.0) {
        // Miss
        vec3 rayOriginScene = coords_pos_viewToWorld(rayOriginView, gbufferModelViewInverse);
        vec3 rayWorldDir = coords_dir_viewToWorld(rayDirView);
        initialSampleData.hitRadiance = restir_irradiance_sampleIrradianceMiss(texelPos, rayOriginScene, rayWorldDir);
    } else {
        vec3 rayEndView = rayOriginView + rayDirView * hitDistance;
        vec3 rayEndScreen = coords_viewToScreen(rayEndView, global_camProj);
        vec2 hitTexelPosF = floor(rayEndScreen.xy * uval_mainImageSize);
        ivec2 hitTexelPos = ivec2(hitTexelPosF);
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