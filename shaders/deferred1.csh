#version 460 compatibility

#extension GL_KHR_shader_subgroup_clustered : enable

#include "/util/Colors.glsl"
#include "/util/GBuffers.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

shared uint shared_normalData[18][18];
shared uint shared_albedoData[18][18];
shared float shared_viewZData[18][18];

uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_temp5;

layout(rgba32ui) uniform restrict uimage2D uimg_gbufferData;
layout(rgba8) uniform writeonly image2D uimg_temp7;

void loadShared(uint idx) {
    if (idx < 324){
        uvec2 localPos = uvec2(idx % 18u, idx / 18u);
        ivec2 texelPos = ivec2(gl_WorkGroupID.xy << 4);
        texelPos += ivec2(localPos) - 1;
        texelPos = clamp(texelPos, ivec2(0), ivec2(global_mainImageSizeI) - 1);
        uvec4 packedData = imageLoad(uimg_gbufferData, texelPos);
        vec3 albedo = texelFetch(usam_temp5, texelPos, 0).rgb;

        shared_albedoData[localPos.y][localPos.x] = packUnorm4x8(vec4(albedo, 1.0));
        shared_normalData[localPos.y][localPos.x] = packedData.b;
        shared_viewZData[localPos.y][localPos.x] = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (all(lessThan(localPos, uvec2(16))) && all(greaterThanEqual(localPos, uvec2(0)))) {
            GBufferData gData;
            gbuffer_unpack(packedData, gData);
            gData.albedo = albedo;
            gbuffer_pack(packedData, gData);
            imageStore(uimg_gbufferData, texelPos, packedData);
        }
    }
}

vec3 readSharedNormal(ivec2 sharedPos) {
    uint packedNormal = shared_normalData[sharedPos.y][sharedPos.x];
    return coords_octDecode11(unpackSnorm2x16(packedNormal));
}

vec3 readSharedAlbedo(ivec2 sharedPos) {
    return unpackUnorm4x8(shared_albedoData[sharedPos.y][sharedPos.x]).rgb;
}

void main() {
    loadShared(gl_LocalInvocationIndex);
    loadShared(gl_LocalInvocationIndex + 256u);
    barrier();

    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        ivec2 centerShared = ivec2(mortonPos) + 1;
        vec3 centerNormal = readSharedNormal(centerShared);
        float centerViewZ = shared_viewZData[centerShared.y][centerShared.x];
        vec3 centerAlbedo = readSharedAlbedo(centerShared);

        float normalWeight = 1.0;
        normalWeight *= dot(centerNormal, readSharedNormal(centerShared + ivec2(-1, 0)));
        normalWeight *= dot(centerNormal, readSharedNormal(centerShared + ivec2(1, 0)));
        normalWeight *= dot(centerNormal, readSharedNormal(centerShared + ivec2(0, -1)));
        normalWeight *= dot(centerNormal, readSharedNormal(centerShared + ivec2(0, 1)));
        normalWeight = pow4(subgroupClusteredMin(normalWeight, 4u));

        float viewZWeight = 1.0;
        float a = max(0.01 * pow2(centerViewZ), 0.5);
        viewZWeight *= a / (a + pow2(centerViewZ - shared_viewZData[centerShared.y - 1][centerShared.x]));
        viewZWeight *= a / (a + pow2(centerViewZ - shared_viewZData[centerShared.y + 1][centerShared.x]));
        viewZWeight *= a / (a + pow2(centerViewZ - shared_viewZData[centerShared.y][centerShared.x - 1]));
        viewZWeight *= a / (a + pow2(centerViewZ - shared_viewZData[centerShared.y][centerShared.x + 1]));

        const float albedoA = 0.2;
        float albedoWeight = 1.0;
        albedoWeight *= albedoA / (albedoA + colors_srgbLuma(abs(centerAlbedo - readSharedAlbedo(centerShared + ivec2(-1, 0)))));
        albedoWeight *= albedoA / (albedoA + colors_srgbLuma(abs(centerAlbedo - readSharedAlbedo(centerShared + ivec2(1, 0)))));
        albedoWeight *= albedoA / (albedoA + colors_srgbLuma(abs(centerAlbedo - readSharedAlbedo(centerShared + ivec2(0, -1)))));
        albedoWeight *= albedoA / (albedoA + colors_srgbLuma(abs(centerAlbedo - readSharedAlbedo(centerShared + ivec2(0, 1)))));

        float noPixelWeight = float(subgroupClusteredMin(centerViewZ, 4u) != -65536.0);
        vec4 vrsWeight2x2 = vec4(normalWeight, viewZWeight, albedoWeight, 1.0) * noPixelWeight;

        if ((threadIdx & 3u) == 0u) {
            imageStore(uimg_temp7, texelPos >> 1, vec4(vrsWeight2x2));
            vec4 vrsWeighr4x4 = subgroupClusteredMin(vrsWeight2x2, 16u);
            if ((threadIdx & 15u) == 0u) {
                ivec2 texelPos4x4 = texelPos >> 2;
                texelPos4x4.x += global_mipmapSizesI[1].x;
                imageStore(uimg_temp7, texelPos4x4, vrsWeighr4x4);
            }
        }
    }
}