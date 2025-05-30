#include "RTWSM.glsl"

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = IMAP_SIZE_D2, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(SETTING_RTWSM_IMAP_SIZE, 2, 1);

layout(r32f) uniform restrict image2D uimg_rtwsm_imap;

shared float shared_reduceBuffer[IMAP_SIZE_D32];

const vec2 TEXEL_SIZE = 1.0 / vec2(SETTING_RTWSM_IMAP_SIZE);

void main() {
    if (gl_LocalInvocationID.x < IMAP_SIZE_D32) {
        shared_reduceBuffer[gl_LocalInvocationID.x] = 0.0;
    }
    barrier();
    uint xtIdx = gl_LocalInvocationID.x << 1;
    uint ytIdx = gl_WorkGroupID.x;

    int f = int(gl_WorkGroupID.y);

    ivec2 coordI = ivec2(xtIdx, ytIdx) * f + ivec2(ytIdx, xtIdx) * (f ^ 1);
    vec2 coordDelta = vec2(1.0, 0.0) * f + vec2(0.0, 1.0) * (f ^ 1);

    {
        float tValue = imageLoad(uimg_rtwsm_imap, coordI).r;
        float subgroupMax = subgroupMax(tValue);
        if (subgroupElect()) {
            shared_reduceBuffer[gl_SubgroupID] = subgroupMax;
        }
    }
    barrier();

    uint subgroupShift = findMSB(gl_SubgroupSize);
    uint remainCount = gl_NumSubgroups;
    while (remainCount > 1) {
        uint subgroupBound = (remainCount + gl_SubgroupSize - 1) >> subgroupShift;
        uint subgroupEnd = min(remainCount - (gl_SubgroupID << subgroupShift), gl_SubgroupSize);
        if (all(bvec2(gl_SubgroupID < subgroupBound, gl_SubgroupInvocationID < subgroupEnd))) {
            float tValue = shared_reduceBuffer[gl_SubgroupInvocationID];
            float subgroupMax = subgroupMax(tValue);
            if (subgroupElect()) {
                shared_reduceBuffer[gl_SubgroupID] = subgroupMax;
            }
        }
        remainCount = remainCount >> subgroupShift;
        barrier();
    }

    if (gl_LocalInvocationID.x == 0) {
        ivec2 outputPos = ivec2(gl_WorkGroupID.xy);
        outputPos.y += IMAP1D_X_Y;
        imageStore(uimg_rtwsm_imap, outputPos, vec4(shared_reduceBuffer[0], 0.0, 0.0, 0.0));
    }
}
