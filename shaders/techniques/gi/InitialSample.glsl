#ifndef INCLUDE_techniques_restir_InitialSample_glsl
#define INCLUDE_techniques_restir_InitialSample_glsl a

#include "Irradiance.glsl"
#include "/util/BSDF.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/HitDirectLighting.glsl"
#include "/techniques/gi/RadianceCacheSample.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"
#include "/techniques/voxel/SurfaceData.glsl"
#include "/techniques/voxel/VoxelHit.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"
#include "/util/NZPacking.glsl"

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

bool restir_initialSample_isFinite(vec3 value) {
    return all(lessThanEqual(abs(value), vec3(FLT_MAX)));
}

vec3 restir_initialSample_sanitizeRadiance(vec3 radiance) {
    return restir_initialSample_isFinite(radiance)
        ? clamp(radiance, 0.0, FP16_MAX)
        : vec3(0.0);
}

#ifdef RESTIR_INITIAL_CANDIDATE_WRITE
void restir_initialCandidate_storeResult(ivec2 texelPos, restir_InitialCandidate candidate) {
    transient_restir_initialCandidate_store(texelPos, vec4(candidate.radiance, candidate.hitDistance));
    uint packedHitNormal = candidate.hitDistance > 0.0
        ? nzpacking_packNormalOct32(candidate.hitNormalView)
        : 0u;
    transient_restir_initialCandidateNormal_store(texelPos, uvec4(packedHitNormal));
}

void restir_initialCandidate_store(ivec2 texelPos, restir_InitialCandidate candidate) {
    restir_initialCandidate_storeResult(texelPos, candidate);
    transient_restir_initialCandidateDirection_store(texelPos, uvec4(
        nzpacking_packNormalOct32(candidate.rayDirView),
        floatBitsToUint(candidate.pdf),
        0u,
        0u
    ));
}
#endif

restir_InitialCandidate restir_initialCandidate_load(ivec2 texelPos) {
    restir_InitialCandidate candidate = restir_initialCandidate_init();
    vec4 radianceAndDistance = transient_restir_initialCandidate_fetch(texelPos);
    uvec4 directionAndPdf = transient_restir_initialCandidateDirection_fetch(texelPos);
    uint packedHitNormal = transient_restir_initialCandidateNormal_fetch(texelPos).x;
    candidate.radiance = radianceAndDistance.rgb;
    candidate.hitDistance = radianceAndDistance.w;
    candidate.rayDirView = nzpacking_unpackNormalOct32(directionAndPdf.x);
    candidate.pdf = uintBitsToFloat(directionAndPdf.y);
    candidate.hitNormalView = candidate.hitDistance > 0.0
        ? nzpacking_unpackNormalOct32(packedHitNormal)
        : vec3(0.0);
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
    vec3 hitWorldPos = coords_pos_viewToWorld(hitViewPos, gbufferModelViewInverse) + cameraPosition;
    vec3 queryWorldNormal = coords_dir_viewToWorld(hitData.normal);
    vec3 queryWorldGeomNormal = coords_dir_viewToWorld(hitData.geomNormal);
    vec3 queryWorldPos = hitWorldPos - queryWorldGeomNormal * 0.02;
    vec3 V = coords_dir_viewToWorld(normalize(rayOriginView - hitViewPos));
    RCLookupResult rcLookup = rc_lookupDiffuseGI(V, queryWorldPos, queryWorldNormal, queryWorldGeomNormal);

    candidate = restir_initialCandidate_init();
    candidate.rayDirView = rayDirView;
    candidate.pdf = pdf;
    candidate.hitDistance = hitDistance;
    candidate.hitNormalView = hitData.geomNormal;
    candidate.radiance = hitMaterial.emissive
        + gi_hitDirectLighting(hitMaterial, hitWorldPos, V, queryWorldNormal, queryWorldGeomNormal);

    if (rcLookup.weight > 0.0 && restir_initialSample_isFinite(rcLookup.radiance)) {
        candidate.radiance += rcLookup.radiance * hitMaterial.albedo;
    }

    candidate.radiance = restir_initialSample_sanitizeRadiance(candidate.radiance);
    return true;
}

restir_InitialCandidate restir_initialSample_buildVoxelCandidate(
    ivec2 texelPos,
    vec3 rayOriginWorld,
    vec3 rayDirView,
    vec3 rayWorldDir,
    float pdf,
    VoxelHit hit
) {
    restir_InitialCandidate candidate = restir_initialCandidate_init();
    candidate.rayDirView = rayDirView;
    candidate.pdf = pdf;

    if (!hit.hit) {
        candidate.hitDistance = RESTIR_INITIAL_CANDIDATE_SKY_MISS;
        candidate.radiance = restir_initialSample_sanitizeRadiance(
            restir_initialSample_sampleSky(texelPos, rayWorldDir)
        );
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
    candidate.radiance = surface.material.emissive
        + gi_hitDirectLighting(surface.material, hit.hitPos, V, hit.normal, hit.normal);
    if (rcLookup.weight > 0.0 && restir_initialSample_isFinite(rcLookup.radiance)) {
        candidate.radiance += rcLookup.radiance * surface.material.albedo;
    }
    candidate.radiance = restir_initialSample_sanitizeRadiance(candidate.radiance);
    return candidate;
}

float restir_initialSample_specularProbability(vec3 wiTangent, Material material) {
    if (material.dielectric <= 0.0) {
        return 1.0;
    }

    vec3 fresnelV = saturate(fresnel_evalMaterial(material, wiTangent.z));
    vec3 totalEnergy = material.albedo * (vec3(1.0) - fresnelV) + fresnelV;
    float pSpec = colors2_colorspaces_luma(
        COLORS2_WORKING_COLORSPACE,
        fresnelV * safeRcp(totalEnergy)
    );
    float discreteBin = clamp(floor(pSpec * 256.0), 13.0, 243.0);
    return discreteBin * (1.0 / 256.0);
}

bool restir_initialSample_useGeomSamplingFrame(
    vec3 V,
    Material material,
    out vec3 wiTangent
) {
    wiTangent = normalize(material.tbnInv * V);
    bool useGeomFrame = wiTangent.z <= 0.0
        && restir_initialSample_isFinite(wiTangent);
    if (useGeomFrame) {
        wiTangent = normalize(material.geomTbnInv * V);
    }
    return useGeomFrame;
}

float restir_initialSample_evaluateRayPdf(
    vec3 rayDirView,
    vec3 geomNormal,
    vec3 V,
    Material material
) {
    vec3 wiTangent;
    bool useGeomFrame = restir_initialSample_useGeomSamplingFrame(V, material, wiTangent);
    mat3 samplingTbnInv = useGeomFrame ? material.geomTbnInv : material.tbnInv;

    vec3 lightTangent = normalize(samplingTbnInv * rayDirView);
    if (
        wiTangent.z <= 0.0
        || lightTangent.z <= 0.0
        || dot(rayDirView, geomNormal) <= 0.0
        || !restir_initialSample_isFinite(wiTangent)
        || !restir_initialSample_isFinite(lightTangent)
    ) {
        return 0.0;
    }

    float pSpec = restir_initialSample_specularProbability(wiTangent, material);
    float cosinePdf = lightTangent.z * RCP_PI;
    float vndfPdf = 0.0;
    vec3 halfVector = lightTangent + wiTangent;
    float halfLength2 = dot(halfVector, halfVector);
    if (halfLength2 > 1e-12) {
        vec3 H = halfVector * inversesqrt(halfLength2);
        if (H.z > 0.0 && dot(wiTangent, H) > 0.0) {
            float a2 = pow2(material.roughness);
            float NdotH2 = pow2(H.z);
            float dDenominator = dot(H.xy, H.xy) + a2 * NdotH2;
            float d = a2 / (PI * pow2(dDenominator));
            float smithDenominator = wiTangent.z
                + sqrt(a2 + (1.0 - a2) * pow2(wiTangent.z));
            vndfPdf = d / (2.0 * smithDenominator);
        }
    }

    float pdf = pSpec * vndfPdf + (1.0 - pSpec) * cosinePdf;
    return pdf > 0.0 && !isnan(pdf) && !isinf(pdf) ? pdf : 0.0;
}

// Slot 0 (RANDOM_FRAME/64u)   -> branch random (stbnVec1)
// Slot 1 (RANDOM_FRAME/64u+1) -> direction random (stbnVec2 or stbnUnitVec3Cosine)
vec3 restir_initialSample_generateRayDir(
    ivec2 texelPos,
    vec3 geomNormal,
    vec3 V,
    Material material,
    out float pdf
) {
    vec3 wiTangent;
    bool useGeomFrame = restir_initialSample_useGeomSamplingFrame(V, material, wiTangent);
    mat3 samplingTbn = useGeomFrame ? material.geomTbn : material.tbn;
    pdf = 0.0;
    if (wiTangent.z <= 0.0 || !restir_initialSample_isFinite(wiTangent)) {
        return normalize(geomNormal);
    }

    ivec2 directionRandKey = rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 1u);
    vec2 xi = rand_stbnVec2(directionRandKey, RANDOM_FRAME);
    xi = (xi * 255.0 + 0.5) * (1.0 / 256.0);
    float choiceRand = rand_stbnVec1(
        rand_newStbnPos(texelPos, RANDOM_FRAME / 64u),
        RANDOM_FRAME
    );
    choiceRand = (choiceRand * 255.0 + 0.5) * (1.0 / 256.0);

    float pSpec = restir_initialSample_specularProbability(wiTangent, material);
    vec3 sampleDirTangent;
    if (choiceRand < pSpec) {
        vec3 halfTangent = bsdf_VNDFSphericalCap(
            wiTangent,
            vec2(material.roughness),
            xi
        );
        sampleDirTangent = reflect(-wiTangent, halfTangent);
    } else {
        sampleDirTangent = rand_stbnUnitVec3Cosine(directionRandKey, RANDOM_FRAME);
    }

    vec3 rayDirView = normalize(samplingTbn * sampleDirTangent);
    rayDirView = nzpacking_unpackNormalOct32(nzpacking_packNormalOct32(rayDirView));
    pdf = restir_initialSample_evaluateRayPdf(rayDirView, geomNormal, V, material);
    return rayDirView;
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
