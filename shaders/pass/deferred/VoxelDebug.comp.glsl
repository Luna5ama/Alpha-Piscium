// VoxelDebug.comp.glsl
// Debug pass: traces a primary ray per pixel through the sparse-64-tree voxel
// representation, maps the hit material ID to a per-ID random colour, and
// writes the result (with alpha = 1 on hit, 0 on miss) into uimg_overlays.
// The OverlayComposite pass later alpha-blends this over the main image.
//
// Enabled via SETTING_DEBUG_VOXEL_TRACE (see options.main.kts / Options.glsl).
// Entry point: deferred26.csh

#include "/Base.glsl"

// All voxel SSBOs are read-only in this pass.
#define VOXEL_BRICK_DATA_MODIFIER    restrict readonly buffer
#define VOXEL_MATERIAL_DATA_MODIFIER restrict readonly buffer
#define VOXEL_TREE_DATA_MODIFIER     restrict readonly buffer
#include "/techniques/VoxelTrace.glsl"

layout(rgba8) restrict uniform image2D uimg_overlays;

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

// ---------------------------------------------------------------------------
// Pseudo-random color from a material ID using a fast integer hash.
// ID 0 → transparent black (no block / background shows through).
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

    float r = float( h        & 0xFFu) * (1.0 / 255.0);
    float g = float((h >>  8) & 0xFFu) * (1.0 / 255.0);
    float b = float((h >> 16) & 0xFFu) * (1.0 / 255.0);
    // Keep colours reasonably bright so they are distinguishable on-screen.
    return vec4(r * 0.6 + 0.4, g * 0.6 + 0.4, b * 0.6 + 0.4, 1.0);
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;

    // ------------------------------------------------------------------
    // Build primary ray from camera matrices.
    // NDC → view direction → world direction.
    // ------------------------------------------------------------------
    vec2 uv  = (vec2(texelPos) + 0.5) / vec2(uval_mainImageSizeI);
    vec2 ndc = uv * 2.0 - 1.0;

    // Unproject NDC to view-space direction (use global unjittered proj).
    vec4 viewFar = global_camProjInverse * vec4(ndc, 1.0, 1.0);
    viewFar.xyz /= viewFar.w;

    // Rotate to world space (w=0 → direction, no translation).
    vec3 worldDir    = normalize((gbufferModelViewInverse * vec4(viewFar.xyz, 0.0)).xyz);
    vec3 worldOrigin = vec3(cameraPositionInt) + cameraPositionFract;

    // ------------------------------------------------------------------
    // Trace and output.
    // ------------------------------------------------------------------
    VoxelHit hit = voxel_traceRay(worldOrigin, worldDir, 256);

    imageStore(uimg_overlays, texelPos, materialIdToColor(hit.materialID));
}

