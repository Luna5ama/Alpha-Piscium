#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"

layout(local_size_x = 64) in;
const ivec3 workGroups = ivec3(AMBIENT_IRRADIANCE_LUT_SIZE, AMBIENT_IRRADIANCE_LUT_SIZE, 1);

layout(rgba16f) uniform restrict image2D uimg_frgba16f;

shared vec3 shared_skySum[64];
shared vec3 shared_groundSum[64];
shared vec3 shared_directSum[64];

void clouds_amblut_accumulate(ivec2 texelPos, vec3 current) {
    vec4 previous = persistent_cloudsAmbLUT_load(texelPos);
    previous.a *= global_historyResetFactor;
    #ifndef SETTING_VIDEO_RENDER_MODE
    float historyLength = min(previous.a + 1.0, 2.0);
    #else
    float historyLength = min(previous.a + 1.0, 16.0);
    #endif
    persistent_cloudsAmbLUT_store(texelPos, vec4(mix(previous.rgb, current, 1.0 / historyLength), historyLength));
}

void main() {
    bool cumulus = clouds_amblut_isCumulus();
    ivec2 texelPos = ivec2(gl_WorkGroupID.xy);
    bool firstGroup = all(equal(texelPos, ivec2(0)));
    if (global_historyResetFactor < 1.0) {
        if (cumulus && gl_LocalInvocationIndex == 0u) {
            vec4 previous = persistent_cloudsAmbLUT_load(texelPos);
            previous.a *= global_historyResetFactor;
            persistent_cloudsAmbLUT_store(texelPos, previous);
        }
        if (!cumulus && firstGroup && gl_LocalInvocationIndex < 3u) {
            ivec2 resetPos = ivec2(gl_LocalInvocationIndex, 32);
            vec4 previous = persistent_cloudsAmbLUT_load(resetPos);
            previous.a *= global_historyResetFactor;
            persistent_cloudsAmbLUT_store(resetPos, previous);
        }
    }
    if (cumulus && !firstGroup) return;

    vec3 skySum = vec3(0.0);
    vec3 groundSum = vec3(0.0);
    vec3 directSum = vec3(0.0);
    for (uint i = gl_LocalInvocationIndex; i < SAMPLE_COUNT; i += 64u) {
        vec2 spherical = ssbo_ambLUTWorkingBuffer.rayDir[i];
        float cosPhi = cos(spherical.x);
        vec3 inScattering = ssbo_ambLUTWorkingBuffer.inSctr[i];
        vec3 groundDirect = ssbo_ambLUTWorkingBuffer.groundDirect[i];
        if (cumulus) {
            // Uniform sphere PDF and isotropic phase cancel; keep the hemispheres separate.
            if (cosPhi >= 0.0) skySum += inScattering / float(SAMPLE_COUNT);
            else groundSum += inScattering / float(SAMPLE_COUNT);
            directSum += groundDirect / float(SAMPLE_COUNT);
        } else {
            vec2 jitter = rand_r2Seq2(i + 32u * frameCounter);
            vec3 viewDir = coords_equirectanglarBackward((vec2(texelPos) + jitter) / 32.0);
            float sinPhi = sin(spherical.x);
            vec3 rayDir = vec3(cos(spherical.y) * sinPhi, cosPhi, sin(spherical.y) * sinPhi);
            float cosTheta = dot(viewDir, rayDir);
            float phase = mix(phasefunc_CornetteShanks(cosTheta, 0.9), phasefunc_CornetteShanks(cosTheta, -0.6), SETTING_CLOUDS_AMB_BACKSCATTER_FACTOR);
            skySum += (inScattering + groundDirect) * phase * SPHERE_SOLID_ANGLE / float(SAMPLE_COUNT);
        }
    }
    skySum = subgroupAdd(skySum);
    groundSum = subgroupAdd(groundSum);
    directSum = subgroupAdd(directSum);
    if (subgroupElect()) {
        shared_skySum[gl_SubgroupID] = skySum;
        shared_groundSum[gl_SubgroupID] = groundSum;
        shared_directSum[gl_SubgroupID] = directSum;
    }
    barrier();

    if (gl_SubgroupID == 0u) {
        skySum = vec3(0.0);
        groundSum = vec3(0.0);
        directSum = vec3(0.0);
        for (uint i = gl_SubgroupInvocationID; i < gl_NumSubgroups; i += gl_SubgroupSize) {
            skySum += shared_skySum[i];
            groundSum += shared_groundSum[i];
            directSum += shared_directSum[i];
        }
        skySum = subgroupAdd(skySum);
        groundSum = subgroupAdd(groundSum);
        directSum = subgroupAdd(directSum);
        if (subgroupElect()) {
            if (cumulus) {
                clouds_amblut_accumulate(ivec2(0, 32), skySum);
                clouds_amblut_accumulate(ivec2(1, 32), groundSum);
                clouds_amblut_accumulate(ivec2(2, 32), directSum);
            } else {
                clouds_amblut_accumulate(texelPos, skySum);
            }
        }
    }
}
