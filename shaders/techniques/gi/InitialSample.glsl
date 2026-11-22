#ifndef INCLUDE_techniques_restir_InitialSample_glsl
#define INCLUDE_techniques_restir_InitialSample_glsl a

#include "Irradiance.glsl"
#include "/util/BSDF.glsl"
#include "/techniques/gi/ReprojectInfo.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/HitDirectLighting.glsl"
#include "/techniques/gi/PathGuiding.glsl"
#include "/techniques/gi/RadianceCacheSample.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"
#include "/techniques/voxel/SurfaceData.glsl"
#include "/techniques/voxel/VoxelHit.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"
#include "/util/NZPacking.glsl"

const float RESTIR_INITIAL_CANDIDATE_ENVIRONMENT = -1.0;
const float RESTIR_INITIAL_CANDIDATE_NEEDS_VOXEL = -2.0;
const float RESTIR_INITIAL_CANDIDATE_INVALID = -3.0;
// The voxel trace ran out of steps: a legal sample with zero radiance, not a sky miss.
const float RESTIR_INITIAL_CANDIDATE_TRACE_EXHAUSTED = -4.0;

struct restir_InitialCandidate {
    vec3 radiance;
    float hitDistance;
    vec3 rayDirView;
    float pdf;
    vec3 hitNormalView;
    bool screenOnly;
};

restir_InitialCandidate restir_initialCandidate_init() {
    restir_InitialCandidate candidate;
    candidate.radiance = vec3(0.0);
    candidate.hitDistance = RESTIR_INITIAL_CANDIDATE_INVALID;
    candidate.rayDirView = vec3(0.0, 0.0, 1.0);
    candidate.pdf = 0.0;
    candidate.screenOnly = false;
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

#ifdef RESTIR_INITIAL_RESULT_WRITE
void restir_initialCandidate_storeResult(ivec2 texelPos, restir_InitialCandidate candidate) {
    float state = candidate.hitDistance;
    if (state > 0.0) {
        // Temporal reuse consumes this pixel before replacing it with its spatial sample.
        transient_restir_spatialInput_store(texelPos, uvec4(
            floatBitsToUint(state),
            (nzpacking_packNormalOct32(candidate.hitNormalView) & ~1u) | uint(candidate.screenOnly), 0u, 0u
        ));
        state = 1.0;
    }
    transient_restir_initialCandidate_store(texelPos, vec4(candidate.radiance, state));
}

#endif

#ifdef RESTIR_INITIAL_DIRECTION_WRITE
void restir_initialCandidate_store(ivec2 texelPos, restir_InitialCandidate candidate) {
    transient_restir_initialCandidate_store(texelPos, vec4(candidate.radiance, candidate.hitDistance));
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
    uvec2 geometry = radianceAndDistance.w > 0.0 ? transient_restir_spatialInput_fetch(texelPos).xy : uvec2(0u);
    uint packedHitNormal = geometry.y;
    candidate.screenOnly = (packedHitNormal & 1u) != 0u;
    candidate.radiance = radianceAndDistance.rgb;
    candidate.hitDistance = radianceAndDistance.w > 0.0 ? uintBitsToFloat(geometry.x) : radianceAndDistance.w;
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

vec3 gi_secondaryIndirect(vec3 V, vec3 worldPos, vec3 N, vec3 geomN, uint voxelMaterialID,
                          ivec2 screenPixel, vec3 albedo) {
    #if SETTING_GI_USE_REFERENCE != 2
    #ifdef SETTING_RC_ENABLE
    if (voxelMaterialID != 0u) {
        uint level = rc_selectLevel(worldPos);
        vec3 face = rc_faceNormal(rc_dominantFaceId(geomN));
        vec3 ownerPos = worldPos - face * ldexp(1e-3, int(level));
        if (voxel_getMaterialID(ivec3(floor(ownerPos))) == voxelMaterialID) {
            RCLookupResult rc = rc_lookupDiffuseGI(V, worldPos, N, geomN);
            if (rc.weight > 0.0 && restir_initialSample_isFinite(rc.radiance)) return rc.radiance;
        }
    }
    #endif
    if (screenPixel.x >= 0) {
        ReprojectInfo info = reprojectInfo_unpack(transient_gi_diffuse_reprojInfo_fetch(screenPixel));
        if (info.historyResetFactor * global_historyResetFactor > 0.0
            && any(greaterThan(info.bilateralWeights, vec4(0.0)))
            && transient_gi5Reprojected_fetch(screenPixel).x > 0.0) {
            return transient_gi2Reprojected_fetch(screenPixel).rgb * min(albedo, 0.95);
        }
    }
    #endif
    return vec3(0.0);
}

void gi_shadeScreen(inout restir_InitialCandidate candidate, float t, ivec2 pixel,
                    vec3 worldPos, vec3 V, uint voxelMaterialID, vec3 cachePos) {
    GBufferData data = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, pixel, 0), data);
    gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, pixel, 0), data);
    Material material = material_decode(data);
    vec3 N = coords_dir_viewToWorld(data.normal);
    vec3 geomN = coords_dir_viewToWorld(data.geomNormal);
    candidate.hitDistance = t;
    candidate.hitNormalView = data.geomNormal;
    candidate.screenOnly = voxelMaterialID == 0u;
    if (dot(geomN, V) <= 0.0) {
        candidate.radiance = restir_initialSample_sanitizeRadiance(material.emissive);
        return;
    }
    candidate.radiance = restir_initialSample_sanitizeRadiance(material.emissive
        + gi_hitDirectLighting(material, worldPos, V, N, geomN)
        + gi_secondaryIndirect(V, cachePos, N, geomN, voxelMaterialID, pixel, material.albedo));
}

void gi_shadeVoxel(inout restir_InitialCandidate candidate, VoxelHit hit, float t,
                   vec3 V, ivec2 screenPixel) {
    candidate.hitDistance = t;
    candidate.hitNormalView = coords_dir_worldToView(hit.normal);
    voxel_SurfaceData surface = voxel_sampleVoxelSurface(hit, 0.0);
    if (!surface.valid) return;
    candidate.radiance = restir_initialSample_sanitizeRadiance(surface.material.emissive
        + gi_hitDirectLighting(surface.material, hit.hitPos, V, hit.normal, hit.normal)
        + gi_secondaryIndirect(V, hit.hitPos, hit.normal, hit.normal, hit.materialID, screenPixel, surface.material.albedo));
}

float restir_initialSample_specularProbability(vec3 wiTangent, Material material) {
    if (material.dielectric <= 0.0) {
        return 1.0;
    }

    vec3 fresnelV = saturate(fresnel_evalMaterial(material, wiTangent.z));
    vec3 totalEnergy = material.albedo * (vec3(1.0) - fresnelV) + fresnelV;
    float pSpec = colors_colorspaces_luma(
        COLORS_WORKING_COLORSPACE,
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

// Slot 0 (RANDOM_FRAME/64u)   -> branch random (stbnVec1)
// Slot 1 (RANDOM_FRAME/64u+1) -> direction random (stbnVec2 or stbnUnitVec3Cosine)
vec3 restir_initialSample_generateRayDir(
    ivec2 texelPos,
    vec3 geomNormal,
    vec3 V,
    Material material,
    PathGuide guide,
    out float pdf
) {
    vec3 wiTangent;
    bool useGeomFrame = restir_initialSample_useGeomSamplingFrame(V, material, wiTangent);
    mat3 samplingTbn = useGeomFrame ? material.geomTbn : material.tbn;
    mat3 samplingTbnInv = useGeomFrame ? material.geomTbnInv : material.tbnInv;
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
    uvec2 guideBins = pathGuide_bins(guide, pSpec);
    vec3 sampleDirTangent;
    if (choiceRand < pSpec) {
        vec3 halfTangent = bsdf_VNDFSphericalCap(
            wiTangent,
            vec2(material.roughness),
            xi
        );
        sampleDirTangent = reflect(-wiTangent, halfTangent);
    } else if (choiceRand < pSpec + float(guideBins.x + guideBins.y) * (1.0 / 256.0)) {
        PathGuideLobe lobe = choiceRand < pSpec + float(guideBins.x) * (1.0 / 256.0) ? guide.lobe0 : guide.lobe1;
        sampleDirTangent = pathGuide_sampleFolded(normalize(samplingTbnInv * lobe.axisView), lobe.kappa, xi);
    } else {
        sampleDirTangent = rand_stbnUnitVec3Cosine(directionRandKey, RANDOM_FRAME);
    }

    vec3 rayDirView = normalize(samplingTbn * sampleDirTangent);
    rayDirView = nzpacking_unpackNormalOct32(nzpacking_packNormalOct32(rayDirView));
    vec3 lightTangent = normalize(samplingTbnInv * rayDirView);
    if (
        lightTangent.z <= 0.0
        || dot(rayDirView, geomNormal) <= 0.0
        || !restir_initialSample_isFinite(lightTangent)
    ) {
        return rayDirView;
    }

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

    // Diffuse bins mix cosine sampling with the two learned guide lobes.
    float diffusePdf = cosinePdf;
    if (guideBins.x + guideBins.y > 0u) {
        vec2 guideFractions = vec2(guideBins) / (256.0 - pSpec * 256.0);
        diffusePdf *= 1.0 - guideFractions.x - guideFractions.y;
        if (guideBins.x > 0u) {
            diffusePdf += guideFractions.x * pathGuide_lobePdf(guide.lobe0, samplingTbnInv, lightTangent);
        }
        if (guideBins.y > 0u) {
            diffusePdf += guideFractions.y * pathGuide_lobePdf(guide.lobe1, samplingTbnInv, lightTangent);
        }
    }

    pdf = pSpec * vndfPdf + (1.0 - pSpec) * diffusePdf;
    pdf = pdf > 0.0 && pdf <= FLT_MAX ? pdf : 0.0;
    return rayDirView;
}

#endif
