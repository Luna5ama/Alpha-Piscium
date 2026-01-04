/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere

        You can find full license texts in /licenses
*/
#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/NoiseTex.glsl"

#define SAMPLE_COUNT 64

layout(local_size_x = SAMPLE_COUNT) in;
const ivec3 workGroups = ivec3(32, 32, 1);

/*const*/
#define ATMOSPHERE_RAYMARCHING_MULTI_SCTR a
/*const*/
#include "../Raymarching.glsl"

layout(rgba16f) restrict uniform image2D uimg_frgba16f;

shared vec3 shared_inSctrSum[4];
shared vec3 shared_multiSctrAs1Sum[4];

void main() {
    ivec2 texelPos = ivec2(gl_WorkGroupID.xy);

    const float isotopicPhase = UNIFORM_PHASE;

    vec2 texCoord = (texelPos + 0.5) / vec2(MULTI_SCTR_LUT_SIZE);
    texCoord = _atmospherics_air_lut_fromSubUvsToUnit(texCoord, vec2(MULTI_SCTR_LUT_SIZE));

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    float cosLightZenith = texCoord.x * 2.0 - 1.0;
    vec3 lightDir = vec3(0.0, sqrt(saturate(1.0 - cosLightZenith * cosLightZenith)), cosLightZenith);
    // We adjust again viewHeight according to PLANET_RADIUS_OFFSET to be in a valid range.
    float viewHeight = atmosphere.bottom + saturate(texCoord.y + PLANET_RADIUS_OFFSET) * (atmosphere.top - atmosphere.bottom - PLANET_RADIUS_OFFSET);

    {
        vec3 randV = rand_r2Seq3(frameCounter);
        randV = fract(randV + noisetex_blueNoise3D(ivec3(texelPos, int(gl_LocalInvocationIndex))).xyz);
        float randA = randV.x;
        float randB = randV.y;
        float theta = 2.0 * PI * randA;
        float phi = acos(1.0 - 2.0 * randB);
        float cosPhi = cos(phi);
        float sinPhi = sin(phi);
        float cosTheta = cos(theta);
        float sinTheta = sin(theta);

        vec3 rayDir = vec3(cosTheta * sinPhi, sinTheta * sinPhi, cosPhi);

        RaymarchParameters params = raymarchParameters_init();
        params.rayStart = vec3(0.0, 0.0, viewHeight);
        setupRayEnd(atmosphere, params, rayDir);
        params.steps = 16u;

        LightParameters lightParams = lightParameters_init(atmosphere, vec3(0.0), lightDir, rayDir);
        lightParams.rayleighPhase = isotopicPhase;
        lightParams.miePhase = isotopicPhase;

        MultiScatteringResult result = raymarchMultiScattering(atmosphere, params, lightParams, randV.z);

        vec3 inSctr = result.inScattering * SPHERE_SOLID_ANGLE / float(SAMPLE_COUNT);
        vec3 multiSctrAs1 = result.multiSctrAs1 * SPHERE_SOLID_ANGLE / float(SAMPLE_COUNT);

        vec3 inSctrSum = subgroupAdd(inSctr);
        vec3 multiSctrAs1Sum = subgroupAdd(multiSctrAs1);

        if (subgroupElect()) {
            shared_inSctrSum[gl_SubgroupID] = inSctrSum;
            shared_multiSctrAs1Sum[gl_SubgroupID] = multiSctrAs1Sum;
        }
    }
    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        vec3 inSctr = shared_inSctrSum[gl_SubgroupInvocationID];
        vec3 multiSctrAs1 = shared_multiSctrAs1Sum[gl_SubgroupInvocationID];
        vec3 inSctrSum = subgroupAdd(inSctr) * isotopicPhase;
        vec3 multiSctrAs1Sum = subgroupAdd(multiSctrAs1) * isotopicPhase;
        if (subgroupElect()) {
            vec3 r = multiSctrAs1Sum;
            vec3 sumOfAllMultiSctrEventsContribution = 1.0 / (1.0 - r);
            vec3 currResult = inSctrSum * sumOfAllMultiSctrEventsContribution * MULTI_SCTR_LUT_QUANTIZATION_MUL;

            vec4 prevData = persistent_multiSctrLUT_load(texelPos);
            persistent_multiSctrLUT_store(texelPos, temporalUpdate(prevData, currResult, 64.0, texelPos));
        }
    }
}