/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere

        You can find full license texts in /licenses
*/
#include "/util/Rand.glsl"
#include "Common.glsl"

#define SAMPLE_COUNT 64

layout(local_size_x = SAMPLE_COUNT) in;
const ivec3 workGroups = ivec3(32, 32, 1);

#define ATMOSPHERE_RAYMARCHING_MULTI_SCTR a
#include "Raymarching.glsl"

layout(rgba16f) restrict uniform image2D uimg_multiSctrLUT;

shared vec3 shared_inSctrSum[SAMPLE_COUNT];
shared vec3 shared_multiSctrAs1Sum[SAMPLE_COUNT];


void main() {
    ivec2 texelPos = ivec2(gl_WorkGroupID.xy);

    if (all(lessThan(texelPos, ivec2(MULTI_SCTR_LUT_SIZE)))) {
        const float sphereSolidAngle = 4.0 * PI;
        const float isotopicPhase = 1.0 / sphereSolidAngle;

        vec2 texCoord = (texelPos + 0.5) / vec2(MULTI_SCTR_LUT_SIZE);
        texCoord = fromSubUvsToUnit(texCoord, vec2(MULTI_SCTR_LUT_SIZE));

        AtmosphereParameters atmosphere = getAtmosphereParameters();

        float cosLightZenith = texCoord.x * 2.0 - 1.0;
        vec3 lightDir = vec3(0.0, sqrt(saturate(1.0 - cosLightZenith * cosLightZenith)), cosLightZenith);
        // We adjust again viewHeight according to PLANET_RADIUS_OFFSET to be in a valid range.
        float viewHeight = atmosphere.bottom + saturate(texCoord.y + PLANET_RADIUS_OFFSET) * (atmosphere.top - atmosphere.bottom - PLANET_RADIUS_OFFSET);

        {
            vec3 randV = rand_r2Seq3(gl_LocalInvocationIndex + SAMPLE_COUNT * frameCounter);
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
            params.steps = 64u;

            LightParameters lightParams = lightParameters_init(atmosphere, vec3(0.0), lightDir, rayDir);
            lightParams.rayleighPhase = isotopicPhase;
            lightParams.miePhase = isotopicPhase;

            MultiScatteringResult result = raymarchMultiScattering(atmosphere, params, lightParams);
            shared_inSctrSum[gl_LocalInvocationIndex] = result.inScattering * sphereSolidAngle / float(SAMPLE_COUNT);
            shared_multiSctrAs1Sum[gl_LocalInvocationIndex] = result.multiSctrAs1 * sphereSolidAngle / float(SAMPLE_COUNT);
        }

        barrier();

        if (gl_LocalInvocationIndex < 32) {
            shared_inSctrSum[gl_LocalInvocationIndex] += shared_inSctrSum[gl_LocalInvocationIndex + 32];
            shared_multiSctrAs1Sum[gl_LocalInvocationIndex] += shared_multiSctrAs1Sum[gl_LocalInvocationIndex + 32];
        }

        barrier();

        if (gl_LocalInvocationIndex < 16) {
            shared_inSctrSum[gl_LocalInvocationIndex] += shared_inSctrSum[gl_LocalInvocationIndex + 16];
            shared_multiSctrAs1Sum[gl_LocalInvocationIndex] += shared_multiSctrAs1Sum[gl_LocalInvocationIndex + 16];
        }

        barrier();

        if (gl_LocalInvocationIndex < 8) {
            shared_inSctrSum[gl_LocalInvocationIndex] += shared_inSctrSum[gl_LocalInvocationIndex + 8];
            shared_multiSctrAs1Sum[gl_LocalInvocationIndex] += shared_multiSctrAs1Sum[gl_LocalInvocationIndex + 8];
        }

        barrier();

        if (gl_LocalInvocationIndex < 4) {
            shared_inSctrSum[gl_LocalInvocationIndex] += shared_inSctrSum[gl_LocalInvocationIndex + 4];
            shared_multiSctrAs1Sum[gl_LocalInvocationIndex] += shared_multiSctrAs1Sum[gl_LocalInvocationIndex + 4];
        }

        barrier();

        if (gl_LocalInvocationIndex < 2) {
            shared_inSctrSum[gl_LocalInvocationIndex] += shared_inSctrSum[gl_LocalInvocationIndex + 2];
            shared_multiSctrAs1Sum[gl_LocalInvocationIndex] += shared_multiSctrAs1Sum[gl_LocalInvocationIndex + 2];
        }

        barrier();

        if (gl_LocalInvocationIndex == 0) {
            shared_inSctrSum[gl_LocalInvocationIndex] += shared_inSctrSum[gl_LocalInvocationIndex + 1];
            shared_multiSctrAs1Sum[gl_LocalInvocationIndex] += shared_multiSctrAs1Sum[gl_LocalInvocationIndex + 1];

            vec3 inSctrLuminance = shared_inSctrSum[0] * isotopicPhase;
            vec3 multiSctrAs1 = shared_multiSctrAs1Sum[0] * isotopicPhase;

            vec3 r = multiSctrAs1;
            vec3 sumOfAllMultiSctrEventsContribution = 1.0 / (1.0 - r);
            vec3 currResult = inSctrLuminance * sumOfAllMultiSctrEventsContribution;

            vec4 prevResult = imageLoad(uimg_multiSctrLUT, texelPos);
            vec4 newResult;
            prevResult.a *= global_historyResetFactor;
            newResult.a = min(prevResult.a + 1.0, 1024.0);
            newResult.rgb = mix(prevResult.rgb, currResult, 1.0 / newResult.a);

            imageStore(uimg_multiSctrLUT, texelPos, newResult);
        }
    }
}