#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable

#include "/rtwsm/Backward.glsl"
#include "/util/NZPacking.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp1;
uniform sampler2D usam_gbufferViewZ;

layout(rg32ui) uniform writeonly uimage2D uimg_packedNZ;

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;

    ivec2 texelPos1x1 = ivec2(mortonGlobalPosU);
    if (all(lessThan(texelPos1x1, global_mainImageSizeI))) {
        ivec2 texelPos2x2 = ivec2(mortonGlobalPosU) >> 1;

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos1x1, 0).r;
        vec3 viewNormal = texelFetch(usam_temp1, texelPos2x2 + ivec2(global_mipmapSizesI[1].x, 0), 0).rgb;

        if ((threadIdx & 3u) == 0u) {
            vec3 worldNormal = mat3(gbufferModelViewInverse) * viewNormal;
            uvec4 prevNZOutput = uvec4(0u);
            nzpacking_pack(prevNZOutput.xy, worldNormal, viewZ);
            imageStore(uimg_packedNZ, texelPos2x2, prevNZOutput);
        }

        #ifdef SETTING_RTWSM_B
        rtwsm_backward(texelPos1x1, viewZ, viewNormal);
        #endif
    }
}