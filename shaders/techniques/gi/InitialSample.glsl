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

// Stochastic VNDF/cosine sampling with MIS balance heuristic pdf.
// Slot 0 (RANDOM_FRAME/64u)   → choice random (stbnVec1)
// Slot 1 (RANDOM_FRAME/64u+1) → direction random (stbnVec2 or stbnUnitVec3Cosine)
vec3 restir_initialSample_generateRayDir(ivec2 texelPos, vec3 geomNormal, vec3 normal, vec3 V, Material material, out float pdf) {
    const float RESTIR_VNDF_TRIM = 0.25;

    float roughness = material.roughness;
    vec3 wiTangent = normalize(material.tbnInv * V);

    // Specular bounce probability: F / (albedo*(1-F) + F)
    float pSpec = 1.0;
    if (material.dielectric > 0.0) {
        float NdotV = saturate(wiTangent.z);
        vec3 fresnelV = saturate(fresnel_evalMaterial(material, NdotV));
        vec3 fresnelT = vec3(1.0) - fresnelV;
        vec3 totalEnergy = material.albedo * fresnelT + fresnelV;
        pSpec = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, fresnelV * safeRcp(totalEnergy));
        // Clamping this to avoid dead locks that causes fireflies
        pSpec = clamp(pSpec, 0.01, 0.99);
    }

    ivec2 sampleDirRandKey = rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 1u);
    vec2 xi = rand_stbnVec2(sampleDirRandKey, RANDOM_FRAME);
    float choiceRand = rand_stbnVec1(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u), RANDOM_FRAME);

    vec3 sampleDirTangent;

    if (choiceRand < pSpec) {
        // VNDF specular sample
        vec3 wmTangent = bsdf_VNDFSphericalCapTrimmed(wiTangent, roughness, xi, RESTIR_VNDF_TRIM);
        sampleDirTangent = reflect(-wiTangent, wmTangent.xyz);
    } else {
        // Cosine-weighted diffuse sample around shading normal
        sampleDirTangent = rand_stbnUnitVec3Cosine(sampleDirRandKey, RANDOM_FRAME);
    }

    vec3 sampleDirView = normalize(material.tbn * sampleDirTangent);

    // Discard the sample if it's below the geometric normal.
    // This can happen with VNDF sampling or normal mapping.
    pdf = 0.0;
    if (dot(sampleDirView, geomNormal) > 0.0&& sampleDirTangent.z > 0.0) {
        // Compute full MIS balance heuristic pdf for the chosen direction.
        // Both VNDF and cosine pdfs are evaluated for the ACTUAL sampled direction,
        // regardless of which branch was taken.
        vec3 LTangent = sampleDirTangent;
        float NDotL = max(LTangent.z, 1e-7);

        // Cosine-hemisphere pdf
        float cosinePdf = NDotL * RCP_PI;

        float vndfPdf = 0.0;
        vec3 H = normalize(LTangent + wiTangent);

        if (H.z > 0.0) {
            float NdotH2 = pow2(H.z);
            float a2 = pow2(roughness);
            float VdotH = saturate(dot(wiTangent, H));

            float d = a2 / max(PI * pow2(NdotH2 * (a2 - 1.0) + 1.0), 1e-16);
            float g1V = bsdf_smithG1(wiTangent.z, roughness);
            vec3 V_stretch = normalize(vec3(roughness * wiTangent.xy, wiTangent.z));
            float yMax = saturate(1.0 - RESTIR_VNDF_TRIM / (1.0 + V_stretch.z));
            float pdfH = (d * g1V * VdotH) / wiTangent.z / yMax;
            vndfPdf = pdfH / max(4.0 * VdotH, 1e-5);
        }

        // Combined mixture pdf (balance heuristic)
        pdf = pSpec * vndfPdf + (1.0 - pSpec) * cosinePdf;
    }

    return sampleDirView;
}

restir_InitialSampleData restir_initalSample_restoreData(ivec2 texelPos, float viewZ, vec3 geomNormal, vec3 normal, Material selfMaterial, float hitDistance) {
    restir_InitialSampleData initialSampleData;
    vec2 rayOriginScreenXY = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 rayOriginView = coords_toViewCoord(rayOriginScreenXY, viewZ, global_camProjInverse);
    vec3 V = normalize(-rayOriginView);

    float pdf;
    vec3 rayDirView = restir_initialSample_generateRayDir(texelPos, geomNormal, normal, V, selfMaterial, pdf);
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
        if (all(lessThan(abs(rayEndScreen * 2.0 - 1.0), vec3(0.999)))) {
            vec3 rayOriginView = coords_screenToView(sstRay.pRayStart, global_camProjInverse);
            vec3 rayEndView = coords_screenToView(rayEndScreen, global_camProjInverse);
            vec3 rayDiffView = rayEndView - rayOriginView;
            float rayLengthView = length(rayDiffView);
            hitDistance = rayLengthView;
        }
    }
    return hitDistance;
}

#endif