#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/SST2.glsl"
#include "/techniques/restir/InitialSample.glsl"
#include "/util/ThreadGroupTiling.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(std430, binding = 4) readonly buffer RayData {
    uvec4 ssbo_rayData[];
};

layout(std430, binding = 5) readonly buffer RayDataIndices {
    uint ssbo_rayDataIndices[];
};

layout(rgba16f) uniform restrict writeonly image2D uimg_temp1;
layout(rgba32ui) uniform restrict writeonly uimage2D uimg_rgba32ui;

void main() {
    sst_init(SETTING_GI_SST_THICKNESS);

    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;

    uvec2 binId = uvec2(swizzledWGPos >> 1); // 32x32 bin = 4x4 work groups
    uint numBinX = (uval_mainImageSizeI.x + 31) >> 5; // 32x32 bin
    uint binIdx = binId.y * numBinX + binId.x;

    // Calculate bin origin in pixels
    uvec2 binOrigin = binId << 5u; // binId * 32

    // Local position within bin (relative to bin origin)
    uvec2 binLocalPos = mortonGlobalPosU - binOrigin;

    // For edge bins, we need to use linear indexing since the bin is not full 32x32
    // Calculate the actual bin size (clamped to screen bounds)
    uvec2 binEnd = min(binOrigin + 32u, uvec2(uval_mainImageSizeI));
    uvec2 binSize = binEnd - binOrigin;

    uint binLocalIndex;
    if (binSize == uvec2(32u)) {
        // Full bin: use Morton encoding
        binLocalIndex = morton_16bEncode(binLocalPos);
    } else {
        // Edge bin: use linear indexing
        binLocalIndex = binLocalPos.y * binSize.x + binLocalPos.x;
    }

    uint binReadBaseIndex = binIdx * 1024;
    uint binIndicesReadBaseIndex = binIdx * 1024;
    uint indexReadIndex = binIndicesReadBaseIndex + binLocalIndex;
    uint rayIndex = ssbo_rayDataIndices[indexReadIndex];

    if (rayIndex < 0xFFFFFFFFu){
        uint actualRayIndex = sst2_decodeBinLocalIndex(rayIndex);
        uvec4 packedData = ssbo_rayData[binReadBaseIndex + actualRayIndex];
        SSTRay sstRay = sstray_unpack(packedData);
        ivec2 texelPos = sstRay.pRayOriginTexelPos;
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        sstray_recoverOrigin(sstRay, viewZ);
        sst_trace(sstRay, 104);

        // If ray still didn't finish, force it to be a miss
        // This ensures all deferred rays produce output (matching RayGenTrace immediate finish behavior)
        if (sstRay.currT >= 0.0) {
            sstRay.currT = -1.0; // Sky miss
        }

        restir_InitialSampleData sampleData = restir_initialSample_handleRayResult(sstRay);
        transient_restir_initialSample_store(texelPos, restir_initialSampleData_pack(sampleData));
        #if SETTING_DEBUG_OUTPUT
//        imageStore(uimg_temp1, texelPos, vec4(sampleData.hitRadiance, 0.0));
        #endif
    }
}
