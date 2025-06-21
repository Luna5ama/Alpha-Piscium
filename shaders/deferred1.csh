#version 460 compatibility

#extension GL_KHR_shader_subgroup_clustered : enable

#include "/util/Colors.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

shared uint shared_geomNormalData[18][18];
shared uint shared_normalData[18][18];
shared uint shared_albedoData[18][18];
shared float shared_viewZData[18][18];

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_gbufferViewZ;

layout(rgba8) uniform writeonly image2D uimg_temp7;

void loadShared(uint idx) {
    if (idx < 324){
        uvec2 localPos = uvec2(idx % 18u, idx / 18u);
        ivec2 localTexelOffset = ivec2(localPos) - 1;
        ivec2 texelPos = ivec2(gl_WorkGroupID.xy << 4);
        texelPos += localTexelOffset;
        texelPos = clamp(texelPos, ivec2(0), ivec2(global_mainImageSizeI) - 1);
        uvec4 packedData = texelFetch(usam_gbufferData32UI, texelPos, 0);
        vec3 albedo = texelFetch(usam_gbufferData8UN, texelPos, 0).rgb;

        shared_albedoData[localPos.y][localPos.x] = packUnorm4x8(vec4(albedo, 1.0));
        shared_geomNormalData[localPos.y][localPos.x] = packedData.r;
        shared_normalData[localPos.y][localPos.x] = packedData.b;
        shared_viewZData[localPos.y][localPos.x] = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    }
}

vec3 readSharedGeomNormal(ivec2 sharedPos) {
    uint packedNormal = shared_geomNormalData[sharedPos.y][sharedPos.x];
    return coords_octDecode11(unpackSnorm2x16(packedNormal));
}

vec3 readSharedNormal(ivec2 sharedPos) {
    uint packedNormal = shared_normalData[sharedPos.y][sharedPos.x];
    return coords_octDecode11(unpackSnorm2x16(packedNormal));
}

vec3 readSharedAlbedo(ivec2 sharedPos) {
    return unpackUnorm4x8(shared_albedoData[sharedPos.y][sharedPos.x]).rgb;
}

float readSharedViewZ(ivec2 sharedPos) {
    return shared_viewZData[sharedPos.y][sharedPos.x];
}

vec3 readSharedViewPos(ivec2 sharedPos) {
    ivec2 localTexelOffset = ivec2(sharedPos) - 1;
    ivec2 texelPos = ivec2(gl_WorkGroupID.xy << 4);
    texelPos += localTexelOffset;
    texelPos = clamp(texelPos, ivec2(0), ivec2(global_mainImageSizeI) - 1);
    vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
    return coords_toViewCoord(screenPos, readSharedViewZ(sharedPos), global_camProjInverse);
}

float computeGeometryWeight(vec3 centerViewPos, vec3 centerViewGeomNormal, ivec2 sharedPos, float a) {
    vec3 sampleViewPos = readSharedViewPos(sharedPos);
    vec3 sampleViewGeomNormal = readSharedGeomNormal(sharedPos);

    float normalWeight = pow4(dot(centerViewGeomNormal, sampleViewGeomNormal));

    vec3 posDiff = centerViewPos - sampleViewPos;
    float planeDist1 = pow2(dot(posDiff, centerViewGeomNormal));
    float planeDist2 = pow2(dot(posDiff, sampleViewGeomNormal));
    float maxPlaneDist = max(planeDist1, planeDist2);
    float posWeight = a / (a + maxPlaneDist);

    return normalWeight * posWeight;
}

void main() {
    loadShared(gl_LocalInvocationIndex);
    loadShared(gl_LocalInvocationIndex + 256u);
    barrier();

    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos1x1 = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos1x1, global_mainImageSizeI))) {
        ivec2 centerShared = ivec2(mortonPos) + 1;
        vec3 centerGeomNormal = readSharedGeomNormal(centerShared);
        vec3 centerNormal = readSharedNormal(centerShared);
        vec3 centerViewPos = readSharedViewPos(centerShared);
        float centerViewZ = centerViewPos.z;
        vec3 centerAlbedo = readSharedAlbedo(centerShared);

        float geometryWeight = 1.0;
        float a = 0.0001 * pow2(centerViewZ);
        geometryWeight *= computeGeometryWeight(centerViewPos, centerGeomNormal, centerShared + ivec2(-1, 0), a);
        geometryWeight *= computeGeometryWeight(centerViewPos, centerGeomNormal, centerShared + ivec2(1, 0), a);
        geometryWeight *= computeGeometryWeight(centerViewPos, centerGeomNormal, centerShared + ivec2(0, -1), a);
        geometryWeight *= computeGeometryWeight(centerViewPos, centerGeomNormal, centerShared + ivec2(0, 1), a);
        geometryWeight = subgroupClusteredMin(geometryWeight, 4u);

        float normalWeight = 1.0;
        normalWeight *= dot(centerNormal, readSharedNormal(centerShared + ivec2(-1, 0)));
        normalWeight *= dot(centerNormal, readSharedNormal(centerShared + ivec2(1, 0)));
        normalWeight *= dot(centerNormal, readSharedNormal(centerShared + ivec2(0, -1)));
        normalWeight *= dot(centerNormal, readSharedNormal(centerShared + ivec2(0, 1)));
        normalWeight = pow4(subgroupClusteredMin(normalWeight, 4u));

        const float albedoA = 0.2;
        float albedoWeight = 1.0;
        albedoWeight *= albedoA / (albedoA + colors_sRGB_luma(abs(centerAlbedo - readSharedAlbedo(centerShared + ivec2(-1, 0)))));
        albedoWeight *= albedoA / (albedoA + colors_sRGB_luma(abs(centerAlbedo - readSharedAlbedo(centerShared + ivec2(1, 0)))));
        albedoWeight *= albedoA / (albedoA + colors_sRGB_luma(abs(centerAlbedo - readSharedAlbedo(centerShared + ivec2(0, -1)))));
        albedoWeight *= albedoA / (albedoA + colors_sRGB_luma(abs(centerAlbedo - readSharedAlbedo(centerShared + ivec2(0, 1)))));

        vec4 vrsWeight2x2 = vec4(geometryWeight);
        vrsWeight2x2 = subgroupClusteredMin(centerViewZ, 4u) == -65536.0 ? vec4(1.0) : vrsWeight2x2;

        if ((threadIdx & 3u) == 0u) {
            ivec2 texelPos2x2 = texelPos1x1 >> 1;
            imageStore(uimg_temp7, texelPos2x2, vrsWeight2x2);
            vec4 vrsWeight4x4 = subgroupClusteredMin(vrsWeight2x2, 16u);
            if ((threadIdx & 15u) == 0u) {
                ivec2 texelPos4x4 = texelPos1x1 >> 2;
                texelPos4x4.x += global_mipmapSizesI[1].x;
                imageStore(uimg_temp7, texelPos4x4, vrsWeight4x4);
            }
        }
    }
}