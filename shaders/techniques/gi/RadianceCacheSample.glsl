#ifndef INCLUDE_techniques_gi_RadianceCacheSample_glsl
#define INCLUDE_techniques_gi_RadianceCacheSample_glsl a

#include "RadianceCache.glsl"
#include "/techniques/voxel/SurfaceData.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"

void rc_lookupSampleFaceWeighted(
    inout RCLookupResult result,
    vec3 V,
    vec3 P,
    vec3 N,
    uint level,
    ivec3 worldCellCoord,
    uint faceId,
    float interpWeight
) {
    if (interpWeight <= 0.0) {
        return;
    }

    uint entryIndex = rc_entryIndex(level, worldCellCoord);
    uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];

    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);

    if (
    entry.x == RC_INVALID ||
    entry.z != worldKeyHash ||
    !rc_entryMetaValid(entry.w) ||
    rc_entryMetaLevel(entry.w) != level ||
    !rc_hasFace(entry.y, faceId)
    ) {
        result.misses++;
        return;
    }

    uint reservoirIndex = rc_faceReservoirIndex(entry.x, entry.y, faceId);

    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        result.misses++;
        return;
    }

    RCReservoir reservoir = rc_reservoirLoad(rc_currentSide(), reservoirIndex);

    if (!rc_reservoirValid(reservoir)) {
        result.misses++;
        return;
    }

    vec3 faceNormal = rc_faceNormal(faceId);

    float normalWeight = saturate(dot(N, faceNormal));
    if (normalWeight <= 0.0) {
        result.misses++;
        return;
    }

    vec3 faceCenter = rc_faceCenter(worldCellCoord, level, faceId);

    float thickness = ldexp(1.0, int(level));

    float side = dot(P - faceCenter, faceNormal);
    if (side < -thickness) {
        result.misses++;
        return;
    }

    uint age = rc_reservoirMetaAge(reservoir.meta);

    float w = interpWeight * normalWeight;

    if (w <= 0.0) {
        result.misses++;
        return;
    }

    // Move into the owner voxel so the face owner is stable.
    // For +Y face, this moves slightly below the surface.
    // For -Y face, this moves slightly above the surface.
    float surfaceEpsilon = max(ldexp(1e-3, int(level)), 1e-3);
    vec3 ownerP = P - faceNormal * surfaceEpsilon;

    VoxelHit hit;
    hit.hit = true;
    hit.hitPos = P;
    hit.normal = faceNormal;
    hit.materialID = voxel_getMaterialID(ivec3(floor(ownerP)));
    voxel_SurfaceData surface = voxel_sampleVoxelSurface(hit, 0.0);
    surface.material.roughness = max(surface.material.roughness, RC_MAX_ROUGHNESS);
    vec3 wi = normalize(reservoir.sampleDir);

    float faceNoL = max(dot(faceNormal, wi), 0.0);
    float NoL     = max(dot(N, wi), 0.0);
    float NoV     = max(dot(N, V), 0.0);

    if (faceNoL <= 1e-4 || NoL <= 0.0 || NoV <= 0.0) {
        result.misses++;
        return;
    }

    float pCache = faceNoL * RCP_PI;

    vec3 H = normalize(wi + V);
    float NoH = saturate(dot(N, H));
    float LoH = saturate(dot(wi, H));

    ResampleMaterial material = resampleMaterial_fromMaterial(surface.material);

    ResampleBRDF brdf = resampleMaterial_evalBRDF(
        material,
        NoL,
        NoV,
        NoH,
        LoH
    );

    vec3 f = surface.material.albedo * brdf.diffuse + vec3(brdf.specular);

    vec3 estimatedRadiance =
    reservoir.radiance *
    reservoir.avgWY *
    f *
    NoL *
    safeRcp(max(pCache, 1e-4));
    if (rc_luminance(estimatedRadiance) <= 0.0 || any(isnan(estimatedRadiance))) {
        result.misses++;
        return;
    }

    result.radiance += estimatedRadiance * w;
    result.weight += w;

    result.hits++;
    result.levelMask |= 1u << level;
    result.faceMask |= rc_faceBit(faceId);
    result.m = max(result.m, reservoir.m);
    result.age = max(result.age, age);
    result.debug = reservoir.meta & 0xFF;
}

RCLookupResult rc_lookupDiffuseGISmooth(vec3 V, vec3 P, vec3 N, vec3 geomN) {
    RCLookupResult result = rc_lookupInit();

    uint level = rc_selectLevel(P);

    uint faceId = rc_dominantFaceId(geomN);
    vec3 faceNormal = rc_faceNormal(faceId);

    uint axis0;
    uint axis1;
    rc_faceTangentAxes(faceId, axis0, axis1);

    uint normalAxis = rc_faceAxis(faceId);

    // Move slightly behind the queried surface so the owner cell is stable.
    // For +Y face, this moves into the solid cell below the face.
    // For -Y face, this moves into the solid cell above the face.
    float surfaceEpsilon = max(ldexp(1e-3, int(level)), 1e-3);
    vec3 ownerP = P - faceNormal * surfaceEpsilon;

    ivec3 ownerCell = rc_worldCellCoord(ownerP, level);

    // Face centers lie at cell + 0.5 along tangent axes.
    // Therefore the bilinear coordinate over face centers is:
    //
    //     P_t / voxelSize - 0.5
    //
    // The integer part selects the lower face-center cell,
    // and the fractional part is the interpolation weight.
    vec3 cellSpace = ldexp(P, ivec3(-int(level))) - vec3(0.5);

    float u = cellSpace[int(axis0)];
    float v = cellSpace[int(axis1)];

    int u0 = int(floor(u));
    int v0 = int(floor(v));

    float fu = fract(u);
    float fv = fract(v);

    // Cubic smoothstep interpolation weights:
    //
    //   smoothstep01(t) = t * t * (3 - 2 * t)
    //
    // This keeps the same 2x2 footprint as bilinear interpolation, but makes
    // the transition C1-continuous inside each cell interval.
    float su = fu * fu * (3.0 - 2.0 * fu);
    float sv = fv * fv * (3.0 - 2.0 * fv);

    float w00 = (1.0 - su) * (1.0 - sv);
    float w10 = su * (1.0 - sv);
    float w01 = (1.0 - su) * sv;
    float w11 = su * sv;

    ivec3 cell00 = ownerCell;
    ivec3 cell10 = ownerCell;
    ivec3 cell01 = ownerCell;
    ivec3 cell11 = ownerCell;

    cell00[int(axis0)] = u0;
    cell00[int(axis1)] = v0;

    cell10[int(axis0)] = u0 + 1;
    cell10[int(axis1)] = v0;

    cell01[int(axis0)] = u0;
    cell01[int(axis1)] = v0 + 1;

    cell11[int(axis0)] = u0 + 1;
    cell11[int(axis1)] = v0 + 1;

    // The normal-axis coordinate remains fixed from ownerCell.
    // This is what turns the lookup from 2x2x2 into 2x2x1.
    cell00[int(normalAxis)] = ownerCell[int(normalAxis)];
    cell10[int(normalAxis)] = ownerCell[int(normalAxis)];
    cell01[int(normalAxis)] = ownerCell[int(normalAxis)];
    cell11[int(normalAxis)] = ownerCell[int(normalAxis)];

    rc_lookupSampleFaceWeighted(result, V, P, N, level, cell00, faceId, w00);
    rc_lookupSampleFaceWeighted(result, V, P, N, level, cell10, faceId, w10);
    rc_lookupSampleFaceWeighted(result, V, P, N, level, cell01, faceId, w01);
    rc_lookupSampleFaceWeighted(result, V, P, N, level, cell11, faceId, w11);

    if (result.weight > 0.0) {
        result.radiance /= result.weight;
    }

    return result;
}
void rc_lookupSampleFace1x1(
    inout RCLookupResult result,
    vec3 V,
    vec3 P,
    vec3 N,
    uint level,
    ivec3 worldCellCoord,
    uint faceId
) {
    uint entryIndex = rc_entryIndex(level, worldCellCoord);
    uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];

    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);

    if (
    entry.x == RC_INVALID ||
    entry.z != worldKeyHash ||
    !rc_entryMetaValid(entry.w) ||
    rc_entryMetaLevel(entry.w) != level ||
    !rc_hasFace(entry.y, faceId)
    ) {
        result.misses++;
        return;
    }

    uint reservoirIndex = rc_faceReservoirIndex(entry.x, entry.y, faceId);

    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        result.misses++;
        return;
    }

    RCReservoir reservoir = rc_reservoirLoad(rc_currentSide(), reservoirIndex);

    if (!rc_reservoirValid(reservoir)) {
        result.misses++;
        return;
    }

    vec3 faceNormal = rc_faceNormal(faceId);

    float normalWeight = max(dot(N, faceNormal), 0.0);
    if (normalWeight <= 0.0) {
        result.misses++;
        return;
    }

    vec3 faceCenter = rc_faceCenter(worldCellCoord, level, faceId);

    float thickness = ldexp(2.0, int(level));

    float side = dot(P - faceCenter, faceNormal);
    if (side < -thickness) {
        result.misses++;
        return;
    }

    uint age = rc_reservoirMetaAge(reservoir.meta);

    // 1x1 lookup: no bilinear and no tangent distance filter.
    // Weight only by normal compatibility and history freshness.
    float w = normalWeight;

    if (w <= 0.0) {
        result.misses++;
        return;
    }

    // Move into the owner voxel so the face owner is stable.
    // For +Y face, this moves slightly below the surface.
    // For -Y face, this moves slightly above the surface.
    float surfaceEpsilon = max(ldexp(1e-3, int(level)), 1e-3);
    vec3 ownerP = P - faceNormal * surfaceEpsilon;

    VoxelHit hit;
    hit.hit = true;
    hit.hitPos = P;
    hit.normal = faceNormal;
    hit.materialID = voxel_getMaterialID(ivec3(floor(ownerP)));
    voxel_SurfaceData surface = voxel_sampleVoxelSurface(hit, 0.0);
    surface.material.roughness = max(surface.material.roughness, RC_MAX_ROUGHNESS);
    vec3 wi = normalize(reservoir.sampleDir);

    float faceNoL = max(dot(faceNormal, wi), 0.0);
    float NoL     = max(dot(N, wi), 0.0);
    float NoV     = max(dot(N, V), 0.0);

    if (faceNoL <= 1e-4 || NoL <= 0.0 || NoV <= 0.0) {
        result.misses++;
        return;
    }

    float pCache = faceNoL * RCP_PI;

    vec3 H = normalize(wi + V);
    float NoH = saturate(dot(N, H));
    float LoH = saturate(dot(wi, H));

    ResampleMaterial material = resampleMaterial_fromMaterial(surface.material);

    ResampleBRDF brdf = resampleMaterial_evalBRDF(
        material,
        NoL,
        NoV,
        NoH,
        LoH
    );

    vec3 f = surface.material.albedo * brdf.diffuse + vec3(brdf.specular);

    vec3 estimatedRadiance =
    reservoir.radiance *
    reservoir.avgWY *
    f *
    NoL *
    safeRcp(max(pCache, 1e-4));
    if (rc_luminance(estimatedRadiance) <= 0.0 || any(isnan(estimatedRadiance))) {
        result.misses++;
        return;
    }

    result.radiance += estimatedRadiance * w;
    result.weight += w;

    result.hits++;
    result.levelMask |= 1u << level;
    result.faceMask |= rc_faceBit(faceId);
    result.m = max(result.m, reservoir.m);
    result.age = max(result.age, age);
    result.debug = reservoir.meta & 0xFF;
}

RCLookupResult rc_lookupDiffuseGI(vec3 V, vec3 P, vec3 N, vec3 geomN) {
    RCLookupResult result = rc_lookupInit();

    uint level = rc_selectLevel(P);

    uint faceId = rc_dominantFaceId(geomN);
    vec3 faceNormal = rc_faceNormal(faceId);

    // Move into the owner voxel so the face owner is stable.
    // For +Y face, this moves slightly below the surface.
    // For -Y face, this moves slightly above the surface.
    float surfaceEpsilon = max(ldexp(1e-3, int(level)), 1e-3);
    vec3 ownerP = P - faceNormal * surfaceEpsilon;

    ivec3 ownerCell = rc_worldCellCoord(ownerP, level);

    rc_lookupSampleFace1x1(
        result,
        V,
        P,
        N,
        level,
        ownerCell,
        faceId
    );

    if (result.weight > 0.0) {
        result.radiance /= result.weight;
    }

    return result;
}

#endif
