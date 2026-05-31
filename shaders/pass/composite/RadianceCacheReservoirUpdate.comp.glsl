#define RC_DATA_MODIFIER restrict buffer

layout(local_size_x = 256) in;

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"
#include "/util/Colors2.glsl"
#include "/util/Fresnel.glsl"
#include "/util/HardcodedPBR.glsl"
#include "/util/MaterialIDConst.glsl"
#include "/util/Rand.glsl"

const ivec3 workGroups = ivec3(5120, 1, 1);

vec3 rcHemisphereDirection(vec3 normal, vec3 localDir) {
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, normal));
    vec3 B = cross(normal, T);
    return normalize(T * localDir.x + B * localDir.y + normal * localDir.z);
}

void rcTouchHit(VoxelHit hit) {
    uint faceId = rcFaceIdFromNormal(hit.normal);
    vec3 faceNormal = rcFaceNormal(faceId);
    vec3 surfacePos = hit.hitPos - faceNormal * 0.02;
    for (uint level = 0u; level < RC_CLIP_LEVELS; level++) {
        ivec3 worldCellCoord = rcWorldCellCoord(surfacePos, level);
        rcTouchFace(level, worldCellCoord, faceId);
    }
}

bool rcLoadPreviousHitReservoir(
    VoxelHit hit,
    out RCReservoir reservoir,
    out vec3 faceNormal
) {
    reservoir = rcReservoirInit();
    uint faceId = rcFaceIdFromNormal(hit.normal);
    faceNormal = rcFaceNormal(faceId);
    vec3 surfacePos = hit.hitPos - faceNormal * 0.02;
    uint level = rcSelectLevel(surfacePos);
    ivec3 worldCellCoord = rcWorldCellCoord(surfacePos, level);
    uint entryIndex = rcEntryIndex(level, worldCellCoord);
    uint prevBufferIndex = rcBufferEntryIndex(rcPreviousSide(), entryIndex);
    uvec4 prevEntry = rc_indirection[prevBufferIndex];
    uint worldKeyHash = rcWorldKeyHash(level, worldCellCoord);
    if (
        prevEntry.x == RC_INVALID
        || prevEntry.z != worldKeyHash
        || !rcEntryMetaValid(prevEntry.w)
        || rcEntryMetaLevel(prevEntry.w) != level
        || !rcHasFace(prevEntry.y, faceId)
    ) {
        return false;
    }

    uint reservoirIndex = rcFaceReservoirIndex(prevEntry.x, prevEntry.y, faceId);
    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        return false;
    }

    reservoir = rcReservoirLoad(rcPreviousSide(), reservoirIndex);
    return rcReservoirValid(reservoir);
}

struct RCHitSurface {
    vec3 albedo;
    vec3 emissive;
    ResampleMaterial material;
    bool valid;
};

RCHitSurface rcHitSurfaceInit() {
    RCHitSurface surface;
    surface.albedo = vec3(0.0);
    surface.emissive = vec3(0.0);
    surface.material = resampleMaterial_init();
    surface.valid = false;
    return surface;
}

RCHitSurface rcSampleHitSurface(VoxelHit hit) {
    RCHitSurface surface = rcHitSurfaceInit();
    if (!hit.hit || hit.materialID == 0u || hit.materialID == MATERIAL_ID_WATER) {
        return surface;
    }

    HardcodedPBR hardcoded = hardcodedpbr_decode(hit.materialID);
    uint faceId = voxel_faceIndexFromNormal(hit.normal);
    uvec2 tcData = voxel_faceTexcoords[voxel_faceTexcoordIndex(hit.materialID, faceId)];
    vec4 tc = unpackUnorm4x16(tcData);
    if (all(equal(tc, vec4(0.0)))) {
        return surface;
    }

    vec2 localUV = voxel_faceLocalUV(faceId, hit.hitPos);
    vec2 atlasUV = mix(tc.xw, tc.zy, localUV);
    vec3 baseColor = colors2_material_toWorkSpace(texture(usam_blockAtlasColor, atlasUV).rgb);
    if (any(isnan(baseColor))) {
        return surface;
    }

    surface.albedo = baseColor;
    surface.material.f0 = fresnel_iorToF0(max(hardcoded.ior, AIR_IOR));
    surface.material.dielectric = 1.0;
    surface.material.roughness = max(pow2(hardcoded.roughness), 0.001);

    float emissiveScale = hardcoded.emissive * exp2(float(hardcoded.emissiveMultiplier));
    surface.emissive = baseColor * emissiveScale;
    surface.valid = true;
    return surface;
}

vec3 rcSampleMissRadiance(vec3 rayDir) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, rayDir);
    return atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
}

vec3 rcSampleHitRadiance(VoxelHit hit, vec3 outgoingDir, out bool valid) {
    valid = false;
    if (!hit.hit) {
        vec3 missRadiance = rcSampleMissRadiance(normalize(-outgoingDir));
        valid = rcLuminance(missRadiance) > 0.0 && !any(isnan(missRadiance));
        return valid ? missRadiance : vec3(0.0);
    }

    RCHitSurface surface = rcSampleHitSurface(hit);
    if (!surface.valid) {
        return vec3(0.0);
    }

    vec3 radiance = surface.emissive;
    valid = rcLuminance(radiance) > 0.0 && !any(isnan(radiance));

    RCReservoir prevReservoir;
    vec3 faceNormal;
    if (!rcLoadPreviousHitReservoir(hit, prevReservoir, faceNormal)) {
        return radiance;
    }

    vec3 incomingDir = normalize(prevReservoir.sampleDir);
    vec3 viewDir = normalize(outgoingDir);
    float NDotL = dot(faceNormal, incomingDir);
    float NDotV = dot(faceNormal, viewDir);
    if (NDotL <= 0.0 || NDotV <= 0.0) {
        return radiance;
    }

    if (rcLuminance(prevReservoir.radiance) <= 0.0 || any(isnan(prevReservoir.radiance)) || any(isnan(incomingDir))) {
        return radiance;
    }

    vec3 H = incomingDir + viewDir;
    float invHLen = inversesqrt(max(dot(H, H), 1e-6));
    float NDotH = saturate(dot(faceNormal, H * invHLen));
    float LDotH = saturate(dot(incomingDir, H * invHLen));
    ResampleBRDF brdf = resampleMaterial_evalBRDF(surface.material, NDotL, NDotV, NDotH, LDotH);
    if (brdf.full <= 0.0) {
        return radiance;
    }

    vec3 bounceFactor = surface.albedo * brdf.diffuse + vec3(brdf.specular);
    vec3 bounceRadiance = prevReservoir.radiance * bounceFactor;
    if (rcLuminance(bounceRadiance) <= 0.0 || any(isnan(bounceRadiance))) {
        return radiance;
    }

    radiance += bounceRadiance;
    valid = true;
    return radiance;
}

RCCandidate rcGenerateCandidate(uint entryIndex, ivec3 worldCellCoord, uint level, uint faceId) {
    RCCandidate candidate;
    candidate.radiance = vec3(0.0);
    candidate.dir = rcFaceNormal(faceId);
    candidate.hitPos = rcFaceCenter(worldCellCoord, level, faceId);
    candidate.targetWeight = 0.0;
    candidate.valid = false;

    vec3 faceNormal = rcFaceNormal(faceId);
    uvec4 randHash = hash_44_q3(uvec4(entryIndex, faceId, frameCounter, 0x9E3779B9u));
    vec2 randValue = hash_uintToFloat(randHash.xy);
    vec4 localSample = rand_sampleInCosineWeightedHemisphere(randValue);
    vec3 worldDir = rcHemisphereDirection(faceNormal, localSample.xyz);
    float cosTheta = max(dot(faceNormal, worldDir), 0.0);
    if (cosTheta <= 0.0 || localSample.w <= 0.0) {
        return candidate;
    }

    vec3 rayOrigin = rcFaceCenter(worldCellCoord, level, faceId) + faceNormal * 0.05;
    VoxelRay voxelRay = voxelray_setup(rayOrigin, worldDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 128);
    if (hit.hit) {
        rcTouchHit(hit);
    }

    bool radianceValid = false;
    vec3 radiance = rcSampleHitRadiance(hit, -worldDir, radianceValid);
    float targetWeight = rcLuminance(radiance) * PI;
    bool candidateValid = radianceValid
        && targetWeight > 0.0
        && !any(isnan(radiance))
        && !isnan(targetWeight);
    if (!candidateValid) {
        return candidate;
    }

    candidate.radiance = radiance;
    candidate.dir = worldDir;
    if (hit.hit) {
        candidate.hitPos = hit.hitPos;
    }
    candidate.targetWeight = targetWeight;
    candidate.valid = true;
    return candidate;
}

void rcUpdateFace(uint entryIndex, uvec4 entry, ivec3 worldCellCoord, uint level, uint faceId) {
    uint reservoirIndex = rcFaceReservoirIndex(entry.x, entry.y, faceId);
    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        return;
    }

    RCCandidate candidate = rcGenerateCandidate(entryIndex, worldCellCoord, level, faceId);
    RCReservoir reservoir = rcReservoirInit();

    uint prevBufferIndex = rcBufferEntryIndex(rcPreviousSide(), entryIndex);
    uvec4 prevEntry = rc_indirection[prevBufferIndex];
    bool historyValid = prevEntry.x != RC_INVALID
        && prevEntry.z == entry.z
        && rcEntryMetaValid(prevEntry.w)
        && rcEntryMetaLevel(prevEntry.w) == level
        && rcHasFace(prevEntry.y, faceId);

    uint historyAge = 0u;
    if (historyValid) {
        uint prevReservoirIndex = rcFaceReservoirIndex(prevEntry.x, prevEntry.y, faceId);
        if (prevReservoirIndex < uint(SETTING_RC_POOL_SIZE)) {
            reservoir = rcReservoirLoad(rcPreviousSide(), prevReservoirIndex);
            historyValid = rcReservoirValid(reservoir);
            historyAge = rcReservoirMetaAge(reservoir.meta);
        } else {
            historyValid = false;
        }
    }

    if (historyValid) {
        float randValue = hash_uintToFloat(hash_41_q3(uvec4(entryIndex, faceId, frameCounter, 0x85EBCA6Bu)));
        float reservoirTargetWeight = rcReservoirTargetWeight(reservoir);
        float wSum = max(0.0, reservoir.avgWY) * reservoir.m * reservoirTargetWeight;
        bool selectedCandidate = rcReservoirUpdate(reservoir, wSum, candidate, randValue);
        float selectedTargetWeight = selectedCandidate ? candidate.targetWeight : reservoirTargetWeight;
        bool reservoirValid = rcReservoirValid(reservoir) && selectedTargetWeight > 0.0 && wSum > 0.0;
        reservoir.avgWY = reservoirValid ? wSum * safeRcp(reservoir.m) * safeRcp(selectedTargetWeight) : 0.0;
        reservoir.meta = rcPackReservoirMeta(min(historyAge + 1u, 255u), reservoirValid, 0u);
    } else {
        rcReservoirInitFromCandidate(reservoir, candidate);
    }

    rcReservoirStore(rcCurrentSide(), reservoirIndex, reservoir);
}

void main() {
    voxel_initShared();

    uint entryIndex = gl_GlobalInvocationID.x;
    if (entryIndex >= RC_ENTRY_COUNT) {
        return;
    }

    uint level = rcEntryLevel(entryIndex);
    ivec3 worldCellCoord = rcWorldCellCoordFromEntryIndex(entryIndex);
    uint bufferIndex = rcBufferEntryIndex(rcCurrentSide(), entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];
    if (entry.x == RC_INVALID || entry.z != rcWorldKeyHash(level, worldCellCoord) || !rcEntryMetaValid(entry.w) || rcEntryMetaLevel(entry.w) != level) {
        return;
    }

    uint faceMask = entry.y & 0x3fu;
    for (uint faceId = 0u; faceId < 6u; faceId++) {
        if (rcHasFace(faceMask, faceId)) {
            rcUpdateFace(entryIndex, entry, worldCellCoord, level, faceId);
        }
    }
}
