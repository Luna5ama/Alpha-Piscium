// VoxelDebug.comp.glsl
// Debug pass: traces a ray per pixel through the sparse-64-tree voxel
// representation, maps the hit material ID to a per-ID random colour, and
// writes the result (with alpha = 1 on hit, 0 on miss) into uimg_overlays.
// The OverlayComposite pass later alpha-blends this over the main image.
//
// Enabled via SETTING_DEBUG_VOXEL_TRACE (see options.main.kts / Options.glsl).
// Entry point: deferred69.csh
//
// Three ray modes, selected by SETTING_DEBUG_VOXEL_MODE:
//   0 – Primary camera ray (default)
//   1 – Uniform hemisphere ray from the gbuffer solid surface normal
//   2 – Mirror reflection ray from the gbuffer solid surface normal

#include "/Base.glsl"
layout(local_size_x = 16, local_size_y = 16) in;

// All voxel SSBOs are read-only in this pass.
#define VOXEL_BRICK_DATA_MODIFIER    restrict readonly buffer
#define VOXEL_MATERIAL_DATA_MODIFIER restrict readonly buffer
#define VOXEL_TREE_DATA_MODIFIER     restrict readonly buffer
#define VOXEL_FACE_TEXCOORD_MODIFIER restrict readonly buffer
#define VOXEL_TRACE_DEBUG_COUNTERS   0
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"

#include "/util/GBufferData.glsl"
#include "/util/Hash.glsl"
#include "/util/Rand.glsl"

layout(rgba8) restrict uniform image2D uimg_overlays;
layout(rgba16f) restrict writeonly uniform image2D uimg_temp1;
layout(rgba16f) restrict uniform image2D uimg_temp3;

const vec2 workGroupsRender = vec2(1.0, 1.0);

// ---------------------------------------------------------------------------
// Pseudo-random color from a material ID using a fast integer hash.
// Used as fallback when no texcoord data is available.
// ---------------------------------------------------------------------------
vec4 materialIdToColor(uint id) {
    if (id == 0u) return vec4(0.0, 0.0, 0.0, 1.0);

    // 3-round xorshift-multiply (PCG-like) to spread the bits
    uint h = id;
    h ^= h >> 17u;
    h *= 0xBF324C81u;
    h ^= h >> 13u;
    h *= 0x9C8A2F35u;
    h ^= h >> 16u;

    float r = float(h & 0xFFu) * (1.0 / 255.0);
    float g = float((h >> 8) & 0xFFu) * (1.0 / 255.0);
    float b = float((h >> 16) & 0xFFu) * (1.0 / 255.0);
    // Keep colours reasonably bright so they are distinguishable on-screen.
    return vec4(r * 0.6 + 0.4, g * 0.6 + 0.4, b * 0.6 + 0.4, 1.0);
}

// ---------------------------------------------------------------------------
// Sample the block atlas colour for a voxel hit.
// Falls back to hash colour if no texcoord data has been stored yet.
// ---------------------------------------------------------------------------
vec4 hitToColor(VoxelHit hit) {
    if (!hit.hit) return vec4(0.0);

    uint faceIdx = voxel_faceIndexFromNormal(hit.normal);
    vec4 tc = voxel_faceTexcoords[voxel_faceTexcoordIndex(hit.materialID, faceIdx)];

    // tc == vec4(0) means uninitialised — fall back to hash colour.
    if (tc == vec4(0.0)) return materialIdToColor(hit.materialID);

    // Compute local face UV from the sub-block hit position.
    // Assumes 1×1×1 textured cubes; fract gives position within the block.
    vec3 f = fract(hit.hitPos);
    vec2 localUV;
    vec3 absNormal = abs(hit.normal);
    if (absNormal.x >= absNormal.y && absNormal.x >= absNormal.z) {
        localUV = vec2(f.z, f.y);
    } else if (absNormal.y >= absNormal.z) {
        localUV = vec2(f.x, f.z);
    } else {
        localUV = vec2(f.x, f.y);
    }

    vec2 atlasUV = mix(tc.xw, tc.zy, localUV);
    return vec4(texture(usam_blockAtlasColor, atlasUV).rgb, 1.0);
}

void main() {
    voxel_initShared();

    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;

    #if SETTING_DEBUG_VOXEL_MODE == 0
    // ------------------------------------------------------------------
    // Mode 0: Primary camera ray
    // NDC → view direction → world direction.
    // ------------------------------------------------------------------
    vec2 uv = (vec2(texelPos) + 0.5) / vec2(uval_mainImageSizeI);
    vec2 ndc = uv * 2.0 - 1.0;

    vec4 viewFar = global_camProjInverse * vec4(ndc, 1.0, 1.0);
    viewFar.xyz /= viewFar.w;

    vec3 worldDir = normalize((gbufferModelViewInverse * vec4(viewFar.xyz, 0.0)).xyz);
    vec3 worldOrigin = vec3(cameraPositionInt) + cameraPositionFract;

    VoxelRay voxelRay = voxelray_setup(worldOrigin, worldDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 256);
    imageStore(uimg_overlays, texelPos, hit.hit ? hitToColor(hit) : materialIdToColor(0u));
    #if VOXEL_TRACE_DEBUG_COUNTERS
    imageStore(uimg_temp1, texelPos, vec4(hit.debugCounters));
    #endif

    #else
    // ------------------------------------------------------------------
    // Modes 1 & 2: derive ray origin and direction from the gbuffer solid.
    // ------------------------------------------------------------------

    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    // Sky / no-geometry pixels: let the main image show through.
    if (viewZ <= -65000.0) {
        imageStore(uimg_overlays, texelPos, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Reconstruct world-space surface position.
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec3 scenePos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 worldPos = scenePos + vec3(cameraPositionInt) + cameraPositionFract;

    // Unpack world-space geometry normal.
    vec3 viewNormal = transient_viewNormal_fetch(texelPos).xyz * 2.0 - 1.0;
    vec3 worldNormal = coords_dir_viewToWorld(viewNormal);

    vec3 worldDir;

    #if SETTING_DEBUG_VOXEL_MODE == 1
    // ------------------------------------------------------------------
    // Mode 1: Perfect mirror reflection of the primary view ray.
    // ------------------------------------------------------------------
    vec2 uv2 = (vec2(texelPos) + 0.5) / vec2(uval_mainImageSizeI);
    vec2 ndc2 = uv2 * 2.0 - 1.0;
    vec4 vf2 = global_camProjInverse * vec4(ndc2, 1.0, 1.0);
    vf2.xyz /= vf2.w;
    vec3 incidentWorldDir = normalize((gbufferModelViewInverse * vec4(vf2.xyz, 0.0)).xyz);
    worldDir = reflect(incidentWorldDir, worldNormal);
    #else
    uvec4 rh = hash_44_q3(uvec4(uvec2(texelPos), uint(1144u), 0x9E3779B9u));
    vec2  r2 = hash_uintToFloat(rh.xy);

    #if SETTING_DEBUG_VOXEL_MODE == 2
    // ------------------------------------------------------------------
    // Mode 2: Uniform hemisphere sample above the surface normal.
    // ------------------------------------------------------------------
    vec4  s = rand_sampleInHemisphere(r2);
    #else // SETTING_DEBUG_VOXEL_MODE == 3
    // ------------------------------------------------------------------
    // Mode 3: Cosine-weighted hemisphere sample above the surface normal.
    // ------------------------------------------------------------------
    vec4  s = rand_sampleInCosineWeightedHemisphere(r2);
    #endif

    // Build orthonormal basis (Gram-Schmidt from world normal).
    vec3 up = abs(worldNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, worldNormal));
    vec3 B = cross(worldNormal, T);
    worldDir = normalize(T * s.x + B * s.y + worldNormal * s.z);
    #endif

    // Offset origin slightly along normal to avoid self-intersection.
    vec3 worldOrigin = worldPos + worldNormal * 1.0;

    VoxelRay voxelRay = voxelray_setup(worldOrigin, worldDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 256);
    imageStore(uimg_overlays, texelPos, hit.hit ? hitToColor(hit) : vec4(0.0, 0.0, 0.0, 1.0));
//    #if VOXEL_TRACE_DEBUG_COUNTERS
//    imageStore(uimg_temp1, texelPos, vec4(hit.debugCounters));
//    vec3 ao = hit.materialID == 0u ? vec3(1.0) : vec3(0.0);
//    vec4 prev = imageLoad(uimg_temp3, texelPos);
//    float newF = clamp(prev.a * global_taaResetFactor.z + 1.0, 1.0, 64.0);
//    float alpha = 1.0 / newF;
//    ao = mix(prev.rgb, ao, alpha);
//    imageStore(uimg_temp3, texelPos, vec4(ao, newF));
//    #endif
    #endif
}

