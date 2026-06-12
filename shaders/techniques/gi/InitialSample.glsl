#ifndef INCLUDE_techniques_restir_InitialSample_glsl
#define INCLUDE_techniques_restir_InitialSample_glsl a

#include "/techniques/voxel/VoxelTrace.glsl"
#include "Irradiance.glsl"
#include "/util/BSDF.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/RadianceCacheSample.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"
#include "/techniques/voxel/SurfaceData.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

struct restir_InitialSampleData {
    vec4 directionAndLength;
    vec3 hitRadiance;
    float pdf;
};

const float RESTIR_INITIAL_CANDIDATE_SKY_MISS = -1.0;
const float RESTIR_INITIAL_CANDIDATE_NEEDS_VOXEL = -2.0;
const float RESTIR_INITIAL_CANDIDATE_INVALID = -3.0;

struct restir_InitialCandidate {
    vec3 radiance;
    float hitDistance;
    vec3 rayDirView;
    float pdf;
    vec3 hitNormalView;
};

restir_InitialCandidate restir_initialCandidate_init() {
    restir_InitialCandidate candidate;
    candidate.radiance = vec3(0.0);
    candidate.hitDistance = RESTIR_INITIAL_CANDIDATE_INVALID;
    candidate.rayDirView = vec3(0.0, 0.0, 1.0);
    candidate.pdf = 0.0;
    candidate.hitNormalView = vec3(0.0);
    return candidate;
}

void restir_initialCandidate_store(ivec2 texelPos, restir_InitialCandidate candidate) {
    transient_restir_initialCandidate_store(texelPos, vec4(candidate.radiance, candidate.hitDistance));
    transient_restir_initialCandidateDirection_store(texelPos, vec4(candidate.rayDirView, candidate.pdf));
    transient_restir_initialCandidateNormal_store(texelPos, vec4(candidate.hitNormalView * 0.5 + 0.5, 1.0));
}

restir_InitialCandidate restir_initialCandidate_load(ivec2 texelPos) {
    restir_InitialCandidate candidate = restir_initialCandidate_init();
    vec4 radianceAndDistance = transient_restir_initialCandidate_fetch(texelPos);
    vec4 directionAndPdf = transient_restir_initialCandidateDirection_fetch(texelPos);
    vec4 hitNormalData = transient_restir_initialCandidateNormal_fetch(texelPos);
    candidate.radiance = radianceAndDistance.rgb;
    candidate.hitDistance = radianceAndDistance.w;
    candidate.rayDirView = normalize(directionAndPdf.xyz);
    candidate.pdf = directionAndPdf.w;
    candidate.hitNormalView = hitNormalData.rgb * 2.0 - 1.0;
    return candidate;
}

restir_InitialCandidate restir_initialCandidate_makeInvalid(vec3 rayDirView) {
    restir_InitialCandidate candidate = restir_initialCandidate_init();
    candidate.rayDirView = rayDirView;
    return candidate;
}

restir_InitialCandidate restir_initialCandidate_makeVoxelFallback(vec3 rayDirView, float pdf) {
    restir_InitialCandidate candidate = restir_initialCandidate_init();
    candidate.rayDirView = rayDirView;
    candidate.pdf = pdf;
    candidate.hitDistance = RESTIR_INITIAL_CANDIDATE_NEEDS_VOXEL;
    return candidate;
}

bool restir_initialCandidate_needsVoxelFallback(restir_InitialCandidate candidate) {
    return candidate.hitDistance == RESTIR_INITIAL_CANDIDATE_NEEDS_VOXEL;
}

vec3 restir_initialSample_sampleSky(ivec2 texelPos, vec3 worldDirection) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, worldDirection);
    vec3 skyRadiance = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
    #ifdef SETTING_GI_MC_SKYLIGHT_ATTENUATION
    float lmCoordSky = transient_lmCoord_fetch(texelPos).y;
    float skyLightFactor = max(lmCoordSky, linearStep(0.0, 240.0, float(eyeBrightnessSmooth.y)));
    skyRadiance *= skyLightFactor;
    #endif
    return skyRadiance;
}

bool restir_initialSample_screenHitQuery(
    ivec2 centerTexelPos,
    vec3 centerGeomNormalView,
    vec3 rayOriginView,
    vec3 rayDirView,
    float pdf,
    float hitDistance,
    out restir_InitialCandidate candidate
) {
    candidate = restir_initialCandidate_makeVoxelFallback(rayDirView, pdf);

    if (hitDistance <= 0.0) {
        return false;
    }

    vec3 hitViewPos = rayOriginView + rayDirView * hitDistance;
    vec3 hitScreenPos = coords_viewToScreen(hitViewPos, global_camProj);
    ivec2 hitTexelPos = ivec2(hitScreenPos.xy * uval_mainImageSize);

    if (any(lessThan(hitTexelPos, ivec2(0))) || any(greaterThanEqual(hitTexelPos, uval_mainImageSizeI))) {
        return false;
    }

    float hitViewZ = texelFetch(usam_gbufferSolidViewZ, hitTexelPos, 0).x;
    if (hitViewZ <= -65536.0) {
        return false;
    }

    GBufferData hitData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, hitTexelPos, 0), hitData);
    gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, hitTexelPos, 0), hitData);

    if (dot(hitData.geomNormal, hitData.geomNormal) <= 1e-6 || dot(hitData.normal, hitData.normal) <= 1e-6) {
        return false;
    }

    float geomNormalDot = dot(hitData.geomNormal, centerGeomNormalView);
    if (geomNormalDot > 0.99) {
        return false;
    }

    Material hitMaterial = material_decode(hitData);
    vec3 queryWorldPos = coords_pos_viewToWorld(hitViewPos - hitData.geomNormal * 0.02, gbufferModelViewInverse) + cameraPosition;
    vec3 queryWorldNormal = coords_dir_viewToWorld(hitData.normal);
    vec3 queryWorldGeomNormal = coords_dir_viewToWorld(hitData.geomNormal);
    vec3 V = coords_dir_viewToWorld(normalize(rayOriginView - hitViewPos));
    RCLookupResult rcLookup = rc_lookupDiffuseGI(V, queryWorldPos, queryWorldNormal, queryWorldGeomNormal);

    candidate = restir_initialCandidate_init();
    candidate.rayDirView = rayDirView;
    candidate.pdf = pdf;
    candidate.hitDistance = hitDistance;
    candidate.hitNormalView = hitData.normal;
    candidate.radiance = hitMaterial.emissive;

    if (rcLookup.weight > 0.0 && !any(isnan(rcLookup.radiance))) {
        candidate.radiance += rcLookup.radiance * hitMaterial.albedo;
    }

    candidate.radiance = clamp(candidate.radiance, 0.0, FP16_MAX);
    return true;
}

restir_InitialCandidate restir_initialSample_buildVoxelCandidate(
    ivec2 texelPos,
    vec3 rayOriginView,
    vec3 rayDirView,
    float pdf,
    VoxelHit hit
) {
    restir_InitialCandidate candidate = restir_initialCandidate_init();
    candidate.rayDirView = rayDirView;
    candidate.pdf = pdf;

    vec3 rayOriginWorld = coords_pos_viewToWorld(rayOriginView, gbufferModelViewInverse) + cameraPosition;
    vec3 rayWorldDir = coords_dir_viewToWorld(rayDirView);

    if (!hit.hit) {
        candidate.hitDistance = RESTIR_INITIAL_CANDIDATE_SKY_MISS;
        candidate.radiance = clamp(restir_initialSample_sampleSky(texelPos, rayWorldDir), 0.0, FP16_MAX);
        return candidate;
    }

    candidate.hitDistance = distance(hit.hitPos, rayOriginWorld);
    candidate.hitNormalView = coords_dir_worldToView(hit.normal);

    voxel_SurfaceData surface = voxel_sampleVoxelSurface(hit, 0.0);
    if (!surface.valid) {
        return candidate;
    }

    vec3 V = normalize(rayOriginWorld - hit.hitPos);
    RCLookupResult rcLookup = rc_lookupDiffuseGI(V, hit.hitPos, hit.normal, hit.normal);
    candidate.radiance = surface.material.emissive;
    if (rcLookup.weight > 0.0 && !any(isnan(rcLookup.radiance))) {
        candidate.radiance += rcLookup.radiance * surface.material.albedo;
    }
    candidate.radiance = clamp(candidate.radiance, 0.0, FP16_MAX);
    return candidate;
}

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
        pSpec = sqrt(clamp(pSpec, 0.01, 0.99));
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
    vec2 rayOriginScreenXY = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
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
        // 0.0000007629 = 0.05 (near plane) / 65536
        if (all(lessThan(vec3(abs(rayEndScreen.xy * 2.0 - 1.0), rayEndScreen.z), vec3(0.99999, 0.99999, 1.0))) && rayEndScreen.z > 0.0000015259) {
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
