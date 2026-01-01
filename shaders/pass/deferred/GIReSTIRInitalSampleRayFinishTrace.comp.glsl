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

    uvec2 clusterId = uvec2(swizzledWGPos >> 1); // 32x32 cluster = 4x4 work groups
    uint numClusterX = (uval_mainImageSizeI.x + 31) >> 5; // 32x32 cluster
    uint clusterIdx = clusterId.y * numClusterX + clusterId.x;

    // Calculate cluster origin in pixels
    uvec2 clusterOrigin = clusterId << 5u; // clusterId * 32

    // Local position within cluster (relative to cluster origin)
    uvec2 clusterLocalPos = mortonGlobalPosU - clusterOrigin;

    // For edge clusters, we need to use linear indexing since the cluster is not full 32x32
    // Calculate the actual cluster size (clamped to screen bounds)
    uvec2 clusterEnd = min(clusterOrigin + 32u, uvec2(uval_mainImageSizeI));
    uvec2 clusterSize = clusterEnd - clusterOrigin;

    uint clusterLocalIndex;
    if (clusterSize == uvec2(32u)) {
        // Full cluster: use Morton encoding
        clusterLocalIndex = morton_16bEncode(clusterLocalPos);
    } else {
        // Edge cluster: use linear indexing
        clusterLocalIndex = clusterLocalPos.y * clusterSize.x + clusterLocalPos.x;
    }


    uint clusterReadBaseIndex = clusterIdx * 1024;
    uint clusterIndicesReadBaseIndex = clusterIdx * 1024;
    uint indexReadIndex = clusterIndicesReadBaseIndex + clusterLocalIndex;
    uint rayIndex = ssbo_rayDataIndices[indexReadIndex];

    if (rayIndex < 0xFFFFFFFFu){
        uint actualRayIndex = bitfieldExtract(rayIndex, 0, 12);
        uvec4 packedData = ssbo_rayData[clusterReadBaseIndex + actualRayIndex];
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
        imageStore(uimg_temp1, texelPos, vec4(sampleData.hitRadiance, 0.0));
        #endif
    }
}
