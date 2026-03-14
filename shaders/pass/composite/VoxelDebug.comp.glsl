// VoxelDebug.comp.glsl
// Debug pass: traces a ray per pixel through the sparse-64-tree voxel
// representation, maps the hit material ID to a per-ID random colour, and
// writes the result (with alpha = 1 on hit, 0 on miss) into uimg_overlays.
// The OverlayComposite pass later alpha-blends this over the main image.
//
// Enabled via SETTING_DEBUG_VOXEL_TRACE (see options.main.kts / Options.glsl).
// Entry point: deferred26.csh
//
// Three ray modes, selected by SETTING_DEBUG_VOXEL_MODE:
//   0 – Primary camera ray (default)
//   1 – Uniform hemisphere ray from the gbuffer solid surface normal
//   2 – Mirror reflection ray from the gbuffer solid surface normal

#include "/Base.glsl"

// All voxel SSBOs are read-only in this pass.
#define VOXEL_BRICK_DATA_MODIFIER    restrict readonly buffer
#define VOXEL_MATERIAL_DATA_MODIFIER restrict readonly buffer
#define VOXEL_TREE_DATA_MODIFIER     restrict readonly buffer
#define VOXEL_TRACE_DEBUG_COUNTERS   1
#include "/techniques/VoxelTrace.glsl"

#include "/util/GBufferData.glsl"
#include "/util/Hash.glsl"
#include "/util/Rand.glsl"

layout(rgba8) restrict uniform image2D uimg_overlays;
layout(rgba16f) restrict writeonly uniform image2D uimg_temp1;

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

// ---------------------------------------------------------------------------
// Pseudo-random color from a material ID using a fast integer hash.
// ID 0 → opaque black (covers main image; miss shown as black).
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

void main() {
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

    VoxelHit hit = voxel_traceRay(worldOrigin, worldDir, 256);
    imageStore(uimg_overlays, texelPos, materialIdToColor(hit.materialID));
    imageStore(uimg_temp1, texelPos, vec4(hit.debugCounters));

    #else
    // ------------------------------------------------------------------
    // Modes 1 & 2: derive ray origin and direction from the gbuffer solid.
    // ------------------------------------------------------------------

    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    // Sky / no-geometry pixels: let the main image show through.
    if (viewZ <= -65000.0) {
        imageStore(uimg_overlays, texelPos, vec4(0.0));
        return;
    }

    // Reconstruct world-space surface position.
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec3 scenePos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 worldPos = scenePos + vec3(cameraPositionInt) + cameraPositionFract;

    // Unpack world-space geometry normal.
    GBufferData gData = gbufferData_init();
    gbufferData1_unpack_world(texelFetch(usam_gbufferData1, texelPos, 0), gData);
    vec3 worldNormal = gData.geomNormal;

    vec3 worldDir;

    #if SETTING_DEBUG_VOXEL_MODE == 1
    // ------------------------------------------------------------------
    // Mode 1: Uniform hemisphere sample above the surface normal.
    // ------------------------------------------------------------------
    uvec4 rh = hash_44_q3(uvec4(uvec2(texelPos), uint(114u), 0x9E3779B9u));
    vec2  r2 = hash_uintToFloat(rh.xy);
    vec4  s = rand_sampleInHemisphere(r2);

    // Build orthonormal basis (Gram-Schmidt from world normal).
    vec3 up = abs(worldNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, worldNormal));
    vec3 B = cross(worldNormal, T);
    worldDir = normalize(T * s.x + B * s.y + worldNormal * s.z);

    #else // SETTING_DEBUG_VOXEL_MODE == 2
    // ------------------------------------------------------------------
    // Mode 2: Perfect mirror reflection of the primary view ray.
    // ------------------------------------------------------------------
    vec2 uv2 = (vec2(texelPos) + 0.5) / vec2(uval_mainImageSizeI);
    vec2 ndc2 = uv2 * 2.0 - 1.0;
    vec4 vf2 = global_camProjInverse * vec4(ndc2, 1.0, 1.0);
    vf2.xyz /= vf2.w;
    vec3 incidentWorldDir = normalize((gbufferModelViewInverse * vec4(vf2.xyz, 0.0)).xyz);
    worldDir = reflect(incidentWorldDir, worldNormal);
    #endif

    // Offset origin slightly along normal to avoid self-intersection.
    vec3 worldOrigin = worldPos + worldNormal * 1.0;

    VoxelHit hit = voxel_traceRay(worldOrigin, worldDir, 256);
    imageStore(uimg_overlays, texelPos, materialIdToColor(hit.materialID));
    imageStore(uimg_temp1, texelPos, vec4(hit.debugCounters));
    #endif
}

