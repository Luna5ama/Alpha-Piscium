// Adds one training record per 4x4 pixel tile to the path guide statistics of the pixel's RC face.
// The pixel within the tile rotates every frame independently of its content.
#define GLOBAL_DATA_MODIFIER restrict buffer
#define PG_DATA_MODIFIER restrict buffer

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.25, 0.25);

#include "/techniques/gi/PathGuiding.glsl"
#include "/util/Morton.glsl"
#include "/util/NZPacking.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy) << 2;
    texelPos += ivec2(morton_8bDecode(uint(frameCounter + morton_32bEncode(gl_GlobalInvocationID.xy)) & 15u));
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) {
        return;
    }
    float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).x;
    if (viewZ <= -65536.0) {
        return;
    }
    atomicAdd(pg_selectedCounter, 1u);

    uvec2 record = transient_pathGuide_trainRecord_fetch(texelPos).xy;
    if ((record.y & PG_RECORD_VALID) == 0u) {
        return;
    }

    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec3 geomNormal = normalize(transient_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
    uint slot = pathGuide_faceSlot(viewPos, geomNormal);
    if (slot == RC_INVALID) {
        atomicAdd(pg_noSlotCounter, 1u);
        return;
    }

    vec3 directionWorld = nzpacking_unpackNormalOct32(record.x);
    ivec3 direction = ivec3(round(directionWorld * PG_FIXED_POINT_SCALE));
    uint statsIndex = pathGuide_statsIndex(rc_currentSide(), slot);
    statsIndex += pathGuide_trainingRecord(directionWorld, pg_stats[statsIndex], pg_stats[statsIndex + 1u]);
    atomicAdd(pg_stats[statsIndex].x, uint(direction.x));
    atomicAdd(pg_stats[statsIndex].y, uint(direction.y));
    atomicAdd(pg_stats[statsIndex].z, uint(direction.z));
    atomicAdd(pg_stats[statsIndex].w, uint(PG_FIXED_POINT_SCALE));

    atomicAdd(pg_splatCounter, 1u);
    if ((record.y & PG_RECORD_NEIGHBOR) != 0u) {
        atomicAdd(pg_neighborCounter, 1u);
    }
    if ((record.y & PG_RECORD_SKY) != 0u) {
        atomicAdd(pg_skyCounter, 1u);
    }
}
