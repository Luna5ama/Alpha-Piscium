#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/util/Math.glsl"

// Supa cool sorting from Bob
// Optimized by Claude Opus 4.5 xD
layout(std430, binding = 5) buffer RayDataIndices {
    uint ssbo_rayDataIndices[];
};

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

shared uint temp[2][1024];

void main() {
    uint clusterIdx = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    uvec2 numClusters = (uvec2(uval_mainImageSizeI) + 31u) >> 5u; // 32x32 cluster
    uint totalClusters = numClusters.x * numClusters.y;

    if (clusterIdx < totalClusters) {
        uint clusterIndexBase = clusterIdx * 1024u + gl_LocalInvocationIndex;
        uvec4 blockIdx = uvec4(0, 256u, 512u, 768u) + gl_LocalInvocationIndex;

        // Load all 4 values from global memory
        uvec4 val;
        val.x = ssbo_rayDataIndices[clusterIndexBase];
        val.y = ssbo_rayDataIndices[clusterIndexBase + 256u];
        val.z = ssbo_rayDataIndices[clusterIndexBase + 512u];
        val.w = ssbo_rayDataIndices[clusterIndexBase + 768u];

        // Subgroup-level bitonic sort for each value
        for (uint size = 2u; size <= gl_SubgroupSize; size <<= 1u) {
            for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
                bool isLower = gl_SubgroupInvocationID < (gl_SubgroupInvocationID ^ stride);

                // Block 0 & 2 ascending within subgroup, block 1 & 3 descending (for merge)
                uvec4 pair = subgroupShuffleXor(val, stride);

                bool asc = !bool(gl_SubgroupInvocationID & size);
                bool cond = (asc != bool(gl_SubgroupID & 1u)) == isLower;

                val = mix(max(val, pair), min(val, pair), bvec4(cond));
            }
        }

        // Store to shared memory
        temp[0][blockIdx.x] = val.x;
        temp[0][blockIdx.y] = val.y;
        temp[0][blockIdx.z] = val.z;
        temp[0][blockIdx.w] = val.w;

        barrier();

        uint readBuf = 0u;
        for (uint k = gl_SubgroupSize * 2u; k <= 1024u; k <<= 1u) {
            // Large strides: use shared memory with double buffering
            for (uint j = k >> 1u; j >= gl_SubgroupSize; j >>= 1u) {
                uint writeBuf = readBuf ^ 1u;

                uvec4 pair = blockIdx ^ j;

                uvec4 a;
                a.x = temp[readBuf][blockIdx.x];
                a.y = temp[readBuf][blockIdx.y];
                a.z = temp[readBuf][blockIdx.z];
                a.w = temp[readBuf][blockIdx.w];

                uvec4 b;
                b.x = temp[readBuf][pair.x];
                b.y = temp[readBuf][pair.y];
                b.z = temp[readBuf][pair.z];
                b.w = temp[readBuf][pair.w];

                bvec4 cond = equal(not(bvec4(blockIdx & k)), lessThan(blockIdx, pair));
                uvec4 result = mix(max(a, b), min(a, b), cond);

                temp[writeBuf][blockIdx.x] = result.x;
                temp[writeBuf][blockIdx.y] = result.y;
                temp[writeBuf][blockIdx.z] = result.z;
                temp[writeBuf][blockIdx.w] = result.w;

                readBuf = writeBuf;
                barrier();
            }

            // Small strides: use subgroup shuffles (faster, no shared memory access)
            // Load all 4 values into registers and precompute ascending flags
            uvec4 val;
            val.x = temp[readBuf][blockIdx.x];
            val.y = temp[readBuf][blockIdx.y];
            val.z = temp[readBuf][blockIdx.z];
            val.w = temp[readBuf][blockIdx.w];

            bvec4 ascs = not(bvec4(blockIdx & k));

            for (uint j = gl_SubgroupSize >> 1u; j > 0u; j >>= 1u) {
                bool isLower = gl_SubgroupInvocationID < (gl_SubgroupInvocationID ^ j);

                uvec4 pair = subgroupShuffleXor(val, j);
                bvec4 cond = equal(ascs, bvec4(isLower));

                val = mix(max(val, pair), min(val, pair), cond);
            }

            // Write all 4 values back
            temp[readBuf][blockIdx.x] = val.x;
            temp[readBuf][blockIdx.y] = val.y;
            temp[readBuf][blockIdx.z] = val.z;
            temp[readBuf][blockIdx.w] = val.w;
            barrier();
        }

        // Write back to global memory
        ssbo_rayDataIndices[clusterIndexBase] = temp[readBuf][blockIdx.x];
        ssbo_rayDataIndices[clusterIndexBase + 256u] = temp[readBuf][blockIdx.y];
        ssbo_rayDataIndices[clusterIndexBase + 512u] = temp[readBuf][blockIdx.z];
        ssbo_rayDataIndices[clusterIndexBase + 768u] = temp[readBuf][blockIdx.w];
    }
}