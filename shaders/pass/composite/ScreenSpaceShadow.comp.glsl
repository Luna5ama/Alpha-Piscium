#extension GL_KHR_shader_subgroup_ballot : enable

#define HIZ_SUBGROUP_CHECK a

#include "/techniques/HiZ.glsl"
#include "/techniques/SST.glsl"
#include "/util/Celestial.glsl"
#include "/util/Material.glsl"
#include "/util/Morton.glsl"
#include "/util/Hash.glsl"
#include "/util/Rand.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba8) uniform restrict image2D uimg_temp5;

ivec2 texelPos = ivec2(0);

vec3 sampleInCone(vec3 center, float coneHalfAngle, vec2 rand) {
    // Random azimuth angle
    float phi = 2.0 * 3.141592653589793 * rand.x;

    // Uniform sampling on spherical cap
    float cosTheta = cos(coneHalfAngle);
    float cosAlpha = mix(1.0, cosTheta, rand.y);
    float sinAlpha = sqrt(1.0 - cosAlpha * cosAlpha);

    // Build orthonormal basis (u, v, center)
    vec3 other = abs(center.x) < 0.9 ? vec3(1,0,0) : vec3(0,1,0);
    vec3 u = normalize(cross(center, other));
    vec3 v = cross(center, u);

    // Final direction
    return normalize(cosAlpha * center +
    sinAlpha * (cos(phi) * u + sin(phi) * v));
}

float sss() {
    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
    Material material = material_decode(gData);
    if (material.sss > 0.0) {
        return 1.0;
    }

    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    vec2 screenPos = coords_texelToUV(texelPos, global_mainImageSizeRcp);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec3 rayDir = sampleInCone(uval_shadowLightDirView, SUN_ANGULAR_RADIUS, rand_stbnVec2(texelPos, frameCounter));
    SSTResult result = sst_trace(viewPos, rayDir, 128u);

    return float(!result.hit);
}

void main() {
    sst_init();

    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        if (hiz_groupGroundCheckSubgroup(gl_WorkGroupID.xy, 4)) {
            float sssV = sss();
            vec4 result = imageLoad(uimg_temp5, texelPos);
            result.rgb *= sssV;
            imageStore(uimg_temp5, texelPos, result);
        }
    }
}
