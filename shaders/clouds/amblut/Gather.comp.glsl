#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "Common.glsl"
#include "../Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"

#define ATMOSPHERE_RAYMARCHING_SKY a
#include "/atmosphere/Raymarching.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(AMBIENT_IRRADIANCE_LUT_SIZE, AMBIENT_IRRADIANCE_LUT_SIZE, 1);

shared vec3 shared_inSctrSum[8];

layout(rgba16f) uniform restrict image3D uimg_cloudsAmbLUT;

void main() {
    vec2 jitter = rand_r2Seq2(gl_LocalInvocationID.x + gl_WorkGroupSize.x * frameCounter);
    ivec2 texelPos = ivec2(gl_WorkGroupID.xy);
    vec2 viewDirUV = (vec2(texelPos) + jitter) / vec2(AMBIENT_IRRADIANCE_LUT_SIZE);
    vec3 viewDir = coords_octDecode01(viewDirUV);

    vec2 rayDirSpherical = ssbo_ambLUTWorkingBuffer.rayDir[gl_LocalInvocationIndex];
    float phi = rayDirSpherical.x;
    float theta = rayDirSpherical.y;
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    vec3 rayDir = vec3(cosTheta * sinPhi, sinTheta * sinPhi, cosPhi);
    vec3 inSctr = ssbo_ambLUTWorkingBuffer.inSctr[gl_LocalInvocationIndex];

    float cosLightTheta = dot(viewDir, rayDir);
    CloudParticpatingMedium cloudMedium = clouds_cirrus_medium(cosLightTheta);
    vec3 phase = mix(cornetteShanksPhase(cosLightTheta, CLOUDS_CIRRUS_ASYM), cloudMedium.phase, 0.5);
    phase = mix(phase, vec3(UNIFORM_PHASE), 0.25);

    vec3 phasedInSctr = inSctr * phase * SPHERE_SOLID_ANGLE / float(SAMPLE_COUNT);
    vec3 subgroupSum1 = subgroupAdd(phasedInSctr);
    if (subgroupElect()) {
        shared_inSctrSum[gl_SubgroupID] = subgroupSum1;
    }
    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        vec3 subgroupSum2 = shared_inSctrSum[gl_SubgroupInvocationID];
        subgroupSum2 = subgroupAdd(subgroupSum2);
        if (subgroupElect()) {
            vec3 currResult = subgroupSum2;
            ivec3 texelPos3D = ivec3(texelPos, 0);
            vec4 prevResult = imageLoad(uimg_cloudsAmbLUT, texelPos3D);
            vec4 newResult;
            newResult.a = min(prevResult.a + 1.0, 1024.0);
            newResult.rgb = mix(prevResult.rgb, currResult, 1.0 / newResult.a);

            imageStore(uimg_cloudsAmbLUT, texelPos3D, newResult);
        }
    }
}
