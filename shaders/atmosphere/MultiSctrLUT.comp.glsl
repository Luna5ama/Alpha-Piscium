// Contains code adopted from:
// https://github.com/sebh/UnrealEngineSkyAtmosphere
// MIT License
// Copyright (c) 2020 Epic Games, Inc.
//
// You can find full license texts in /licenses
#include "/util/Rand.glsl"
#include "Common.glsl"

#define SAMPLE_COUNT 64
#define SAMPLE_COUNT_SQRT 8

layout(local_size_x = SAMPLE_COUNT_SQRT, local_size_y = SAMPLE_COUNT_SQRT) in;
const ivec3 workGroups = ivec3(32, 32, 1);

layout(rgba16f) restrict uniform image2D uimg_multiSctrLUT;

shared vec3 shared_inSctrSum[64];
shared vec3 shared_multiSctrAs1Sum[64];

struct MultiScatteringResult {
    vec3 inScattering;
    vec3 multiSctrAs1;
};

MultiScatteringResult raymarchMultiScattering(
AtmosphereParameters atmosphere, RaymarchParameters params, LightParameters lightParams, float stepJitter
) {
    MultiScatteringResult result = MultiScatteringResult(vec3(0.0), vec3(0.0));

    float rcpSteps = 1.0 / float(params.steps);
    vec3 rayStepDelta = (params.rayEnd - params.rayStart) * rcpSteps;
    float rayStepLength = length(params.rayEnd - params.rayStart) * rcpSteps;

    vec3 tSampleToOrigin = vec3(1.0);

    for (uint stepIndex = 0u; stepIndex < params.steps; stepIndex++) {
        float stepIndexF = float(stepIndex) + stepJitter;
        vec3 samplePos = params.rayStart + stepIndexF * rayStepDelta;
        float sampleHeight = length(samplePos);

        vec3 sampleDensity = sampleParticleDensity(atmosphere, sampleHeight);
        vec3 sampleExtinction = computeOpticalDepth(atmosphere, sampleDensity);
        vec3 sampleOpticalDepth = sampleExtinction * rayStepLength;
        vec3 sampleTransmittance = exp(-sampleOpticalDepth);

        vec3 tSunToSample = sampleTransmittanceLUT(atmosphere, lightParams.cosZenith, sampleHeight);

        vec3 sampleInSctr = tSunToSample * computeTotalInSctr(atmosphere, lightParams, sampleDensity);
        // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
        vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
        result.inScattering += tSampleToOrigin * sampleInSctrInt;

        vec3 rayleighInSctr = sampleDensity.x * atmosphere.rayleighSctrCoeff;
        vec3 mieInSctr = sampleDensity.y * atmosphere.mieSctrCoeff;
        vec3 sampleMultiSctr = rayleighInSctr + mieInSctr;
        vec3 sampleMultiSctrInt = (sampleMultiSctr - sampleMultiSctr * sampleTransmittance) / sampleExtinction;
        result.multiSctrAs1 += tSampleToOrigin * sampleMultiSctrInt;

        tSampleToOrigin *= sampleTransmittance;
    }

    return result;
}

void main() {
    ivec2 pixelPos = ivec2(gl_WorkGroupID.xy);

    if (all(lessThan(pixelPos, ivec2(MULTI_SCTR_LUT_SIZE)))) {
        const float sphereSolidAngle = 4.0 * PI;
        const float isotopicPhase = 1.0 / sphereSolidAngle;

        vec2 texCoord = (pixelPos + 0.5) / vec2(MULTI_SCTR_LUT_SIZE);
        texCoord = fromSubUvsToUnit(texCoord, vec2(MULTI_SCTR_LUT_SIZE));

        AtmosphereParameters atmosphere = getAtmosphereParameters();

        float cosLightZenith = texCoord.x * 2.0 - 1.0;
        vec3 lightDir = vec3(0.0, sqrt(saturate(1.0 - cosLightZenith * cosLightZenith)), cosLightZenith);
        // We adjust again viewHeight according to PLANET_RADIUS_OFFSET to be in a valid range.
        float viewHeight = atmosphere.bottom + saturate(texCoord.y + PLANET_RADIUS_OFFSET) * (atmosphere.top - atmosphere.bottom - PLANET_RADIUS_OFFSET);

        float randV = rand_IGN(gl_GlobalInvocationID.xy, frameCounter);

        {
            float i = randV + gl_LocalInvocationID.x;
            float j = randV + gl_LocalInvocationID.y;
            float randA = i / float(SAMPLE_COUNT_SQRT);
            float randB = j / float(SAMPLE_COUNT_SQRT);
            float theta = 2.0 * PI * randA;
            float phi = acos(1.0 - 2.0 * randB);
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);
            float cosTheta = cos(theta);
            float sinTheta = sin(theta);

            vec3 rayDir = vec3(cosTheta * sinPhi, sinTheta * sinPhi, cosPhi);

            RaymarchParameters params;
            params.rayStart = vec3(0.0, 0.0, viewHeight);
            setupRayEnd(atmosphere, params, rayDir);
            LightParameters lightParams;
            lightParams.cosZenith = cosLightZenith;
            lightParams.rayleighPhase = isotopicPhase;
            lightParams.miePhase = isotopicPhase;
            params.steps = 8u;

            MultiScatteringResult result = raymarchMultiScattering(atmosphere, params, lightParams, randV);
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

            vec3 prevResult = imageLoad(uimg_multiSctrLUT, pixelPos).rgb;
            vec3 newResult = mix(currResult, prevResult, 0.9);

            imageStore(uimg_multiSctrLUT, pixelPos, vec4(newResult, 1.0));
        }
    }
}