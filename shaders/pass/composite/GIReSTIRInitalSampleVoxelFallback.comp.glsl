#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#pragma optionNV(unroll none)

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(r32ui) uniform restrict uimage2D uimg_r32ui;
layout(rg32ui) uniform restrict writeonly uimage2D uimg_rg32ui;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
layout(rgba8) uniform restrict writeonly image2D uimg_rgba8;

#define RESTIR_INITIAL_CANDIDATE_WRITE
#define RESTIR_INITIAL_RESULT_WRITE
#include "/techniques/gi/InitialSample.glsl"
#include "/techniques/gi/TemporalSplat.glsl"
#define VOXEL_TRACE_TRUST_MATERIAL_ID
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/techniques/gi/HybridTrace.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Morton.glsl"
#include "/util/ThreadGroupTiling.glsl"

bool gi_sameVoxelSurface(VoxelHit hit, vec3 screenWorld, ivec2 pixel) {
    GBufferData data = gbufferData_init();
    gbufferData1_unpack_world(texelFetch(usam_gbufferSolidData1, pixel, 0), data);
    vec3 ownerOffset = hit.normal * 0.001;
    return data.materialID == hit.materialID && dot(data.geomNormal, hit.normal) > 0.98
        && distance(screenWorld, hit.hitPos) <= 0.015
        && all(equal(floor(screenWorld - ownerOffset), floor(hit.hitPos - ownerOffset)));
}

ivec2 gi_voxelScreenPixel(VoxelHit hit, vec3 hitView) {
    if (hitView.z >= -nearPlane) return ivec2(-1);
    ivec2 pixel = ivec2(floor(gi_rasterPos(hitView).xy));
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, uval_mainImageSizeI))) return ivec2(-1);
    float z = texelFetch(usam_gbufferSolidViewZ, pixel, 0).r;
    if (z <= -65536.0) return ivec2(-1);
    vec3 center = gi_screenSurface(pixel, z);
    vec3 N = coords_dir_worldToView(hit.normal);
    // Lift the depth sample's plane to the projected hit's exact subpixel.
    float denom = dot(hitView, N);
    if (abs(denom) < 1e-5) return ivec2(-1);
    vec3 delta = hitView * (dot(center - hitView, N) / denom);
    return gi_sameVoxelSurface(hit, hit.hitPos + mat3(gbufferModelViewInverse) * delta, pixel) ? pixel : ivec2(-1);
}

void gi_resolveRay(inout restir_InitialCandidate candidate, ivec2 pixel, vec3 p, vec3 worldPos, vec3 dir) {
    vec3 d = candidate.rayDirView;
    vec2 grid = vec2(0.0, GI_RAY_END);
    bool hasGrid = gi_gridInterval(worldPos, dir, grid);
    float t;
    ivec2 hitPixel;
    bool retained;
    int status = gi_traceTerrain(p, d, vec2(0.0, hasGrid ? grid.x : GI_RAY_END), false, t, hitPixel, retained);
    if (status == GI_SCREEN_CLEAR && hasGrid) {
        status = gi_traceScreen(p, d, vec2(grid.x, min(GI_SCREEN_NEAR_DISTANCE, grid.y)), false, t, hitPixel);
        const float selfOffset = 0.005;
        float bias = grid.x > 0.0 ? 0.0 : min(selfOffset, grid.y);
        VoxelRay ray = voxelray_setup(fma(dir, vec3(bias), worldPos), dir, 0u);
        VoxelHit hit = voxel_traceRay(ray, 256, true);
        float voxelT = hit.hit ? ray.lastT + bias : GI_RAY_END;
        if (status == GI_SCREEN_HIT) {
            vec3 screenWorld = fma(dir, vec3(t), worldPos);
            bool same = hit.hit && gi_sameVoxelSurface(hit, screenWorld, hitPixel);
            if (same || (t < voxelT && (!voxel_traceExhausted(hit, ray) || ray.lastT + bias > t))) {
                gi_shadeScreen(candidate, t, hitPixel, screenWorld, -dir, same ? hit.materialID : 0u, hit.hitPos);
                return;
            }
        }
        if (hit.hit) {
            if (dot(hit.normal, hit.normal) == 0.0) {
                // An opaque occupied start cell has no entering face.
                candidate.hitDistance = RESTIR_INITIAL_CANDIDATE_TRACE_EXHAUSTED;
                return;
            }
            gi_shadeVoxel(candidate, hit, voxelT, -dir, gi_voxelScreenPixel(hit, fma(d, vec3(voxelT), p)));
            return;
        }
        status = voxel_traceExhausted(hit, ray) ? GI_SCREEN_EXHAUSTED :
            gi_traceTerrain(p, d, vec2(grid.y, GI_RAY_END), false, t, hitPixel, retained);
    }
    if (status == GI_SCREEN_HIT) {
        #if defined(VOXY) && defined(GI_RETAINED_PROJECTION)
        if (retained) {
            vec4 surface = texelFetch(usam_hiz, hitPixel + ivec2(0, 2 * uval_mainImageSizeI.y), 0);
            uvec2 outgoing = floatBitsToUint(surface.zw) - 0x00800000u;
            vec3 normal, tangent;
            nzpacking_unpackNormalOct16(uint(surface.y), normal, tangent);
            candidate.hitDistance = t;
            candidate.hitNormalView = coords_dir_worldToView(normal);
            candidate.screenOnly = true;
            candidate.radiance = colors_FP16LuvToWorkingColor(outgoing.y);
            if (dot(normal, dir) < 0.0) candidate.radiance += colors_FP16LuvToWorkingColor(outgoing.x) * global_giWideDiffuseValidity;
            candidate.radiance = restir_initialSample_sanitizeRadiance(candidate.radiance);
            return;
        }
        #endif
        gi_shadeScreen(candidate, t, hitPixel, fma(dir, vec3(t), worldPos), -dir, 0u, vec3(0.0));
    } else if (status == GI_SCREEN_CLEAR) {
        candidate.hitDistance = RESTIR_INITIAL_CANDIDATE_ENVIRONMENT;
        candidate.radiance = gi_environmentRadiance(pixel, coords_pos_viewToWorld(p, gbufferModelViewInverse), dir);
        candidate.radiance = restir_initialSample_sanitizeRadiance(candidate.radiance);
    } else {
        candidate.hitDistance = RESTIR_INITIAL_CANDIDATE_TRACE_EXHAUSTED;
    }
}

void main() {
    #ifdef VOXY
        #ifdef GI_RETAINED_PROJECTION
        if (global_giWideState != 3) return;
        #else
        if (global_giWideState == 3) return;
        #endif
    #endif
    restir_temporalSplatInitShared();
    voxel_initShared();

    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    ivec2 texelPos = ivec2(workGroupOrigin + mortonPos);

    if (!all(lessThan(texelPos, uval_mainImageSizeI))) {
        return;
    }

    restir_temporalSplat(texelPos);

    if (transient_restir_initialCandidate_fetch(texelPos).w != RESTIR_INITIAL_CANDIDATE_NEEDS_VOXEL) {
        return;
    }
    uvec4 directionAndPdf = transient_restir_initialCandidateDirection_fetch(texelPos);
    restir_InitialCandidate candidate = restir_initialCandidate_makeVoxelFallback(
        nzpacking_unpackNormalOct32(directionAndPdf.x),
        uintBitsToFloat(directionAndPdf.y)
    );

    float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).x;
    if (viewZ <= -65536.0 || candidate.pdf <= 0.0) {
        restir_initialCandidate_storeResult(texelPos, restir_initialCandidate_makeInvalid(candidate.rayDirView));
        return;
    }

    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
    vec3 rayOriginView = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec3 rayOriginWorld = coords_pos_viewToWorld(rayOriginView, gbufferModelViewInverse) + cameraPosition;
    vec3 rayWorldDir = coords_dir_viewToWorld(candidate.rayDirView);

    gi_resolveRay(candidate, texelPos, rayOriginView, rayOriginWorld, rayWorldDir);
    restir_initialCandidate_storeResult(texelPos, candidate);
}
