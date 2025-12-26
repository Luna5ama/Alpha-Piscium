#ifndef INCLUDE_atmosphere_Common_glsl
#define INCLUDE_atmosphere_Common_glsl a
/*
    References:
        [BRU08] Bruneton, Eric. "Precomputed Atmospheric Scattering". EGSR 2008. 2008.
            https://hal.inria.fr/inria-00290084/document
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere
        [HIL20] Hillaire, Sébastien. "A Scalable and Production Ready Sky and Atmosphere Rendering Technique".
            EGSR 2020. 2020.
            https://sebh.github.io/publications/egsr2020.pdf
        [INT17] Intel Corporation. "Outdoor Light Scattering Sample". 2017.
            Apache License 2.0. Copyright (c) 2017 Intel Corporation.
            https://github.com/GameTechDev/OutdoorLightScattering
        [YUS13] Yusov, Egor. “Practical Implementation of Light Scattering Effects Using Epipolar Sampling and
            1D Min/Max Binary Trees”. GDC 2013. 2013.
            http://gdcvault.com/play/1018227/Practical-Implementation-of-Light-Scattering

        You can find full license texts in /licenses
*/

#include "../Common.glsl"
#include "Constants.glsl"
#include "/util/Math.glsl"
#include "/util/Rand.glsl"
#include "/util/Dither.glsl"
#include "/util/PhaseFunc.glsl"

// Calculate the air density ratio at a given height(km) relative to sea level
// Fitted to U.S. Standard Atmosphere 1976
// See https://www.desmos.com/calculator/8zep6vmnxa
float sampleRayleighDensity(AtmosphereParameters atmosphere, float altitude) {
    const float a0 = 0.00947927584794;
    const float a1 = -0.138528179963;
    const float a2 = -0.00235619411773;
    return exp2(a0 + a1 * altitude + a2 * altitude * altitude);
}

float sampleMieDensity(AtmosphereParameters atmosphere, float altitude) {
    return exp(-altitude / atmosphere.mieHeight);
}

// Calculate the ozone number density in 10^17 molecules/m^3
// See https://www.desmos.com/calculator/ykoihjoqdm
float sampleOzoneDensity(AtmosphereParameters atmosphere, float altitude) {
    float x = max(altitude, 0.0);
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;

    const float d10 = 3.14463183276;
    const float d11 = 0.0498300739786;
    const float d12 = -0.13053950591;
    const float d13 = 0.021937805502;
    const float d14 = -0.000931031499395;
    float d1 = exp2(d10 + d11 * x + d12 * x2 + d13 * x3 + d14 * x4);

    const float d20 = -15.9975955967;
    const float d21 = 2.79421136239;
    const float d22 = -0.128226752502;
    const float d23 = 0.00249280242662;
    const float d24 = -0.0000185558309121;
    float d2 = exp2(d20 + d21 * x + d22 * x2 + d23 * x3 + d24 * x4);

    return d1 + d2;
}

vec3 sampleParticleDensity(AtmosphereParameters atmosphere, float height) {
    float altitude = height - atmosphere.bottom;
    return vec3(
        sampleRayleighDensity(atmosphere, altitude),
        sampleMieDensity(atmosphere, altitude),
        sampleOzoneDensity(atmosphere, altitude)
    );
}

float atmosphere_height(AtmosphereParameters atmosphere, float worldPosY) {
    float worldHeight = max(worldPosY - 62.0, float(SETTING_ATM_ALT_SCALE) * 0.001);
    return worldHeight * (1.0 / float(SETTING_ATM_ALT_SCALE)) + atmosphere.bottom;
}

vec3 atmosphere_viewToAtm(AtmosphereParameters atmosphere, vec3 viewPos) {
    vec3 feetPlayer = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 world = feetPlayer + cameraPosition;
    float height = atmosphere_height(atmosphere, world.y);
    return vec3(feetPlayer.x, 0.0, feetPlayer.z) * (1.0 / float(SETTING_ATM_D_SCALE)) + vec3(0.0, height, 0.0);
}

float atmosphere_heightNoClamping(AtmosphereParameters atmosphere, vec3 worldPos) {
    float worldHeight = worldPos.y - 62.0;
    return worldHeight * (1.0 / float(SETTING_ATM_ALT_SCALE)) + atmosphere.bottom;
}

vec3 atmosphere_viewToAtmNoClamping(AtmosphereParameters atmosphere, vec3 viewPos) {
    // TODO: higher precision
    vec3 feetPlayer = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 world = feetPlayer + cameraPosition;
    float height = atmosphere_heightNoClamping(atmosphere, world);
    return vec3(world.x, 0.0, world.z) * (1.0 / float(SETTING_ATM_D_SCALE)) + vec3(0.0, height, 0.0);
}

void unpackEpipolarData(uvec4 epipolarData, out ScatteringResult sctrResult, out float viewZ) {
    vec2 unpacked1 = unpackHalf2x16(epipolarData.x);
    vec2 unpacked2 = unpackHalf2x16(epipolarData.y);
    vec2 unpacked3 = unpackHalf2x16(epipolarData.z);
    sctrResult.inScattering = vec3(unpacked1.xy, unpacked2.x);
    sctrResult.transmittance = vec3(unpacked2.y, unpacked3.xy);
    viewZ = uintBitsToFloat(epipolarData.w);
}

void packEpipolarData(out uvec4 epipolarData, ScatteringResult sctrResult, ivec2 texelPos) {
    epipolarData.x = packHalf2x16(sctrResult.inScattering.xy);
    epipolarData.y = packHalf2x16(vec2(sctrResult.inScattering.z, sctrResult.transmittance.x));
    epipolarData.z = packHalf2x16(sctrResult.transmittance.yz);
    epipolarData.w = bitfieldInsert(texelPos.x, texelPos.y, 16, 16);
}

#define INVALID_EPIPOLAR_LINE vec4(-1000.0, -1000.0, -100.0, -100.0)

bool isValidScreenLocation(vec2 f2XY) {
    const float SAFETY_EPSILON = 0.2f;
    return all(lessThanEqual(abs(f2XY), 1.0 - (1.0 - SAFETY_EPSILON) / vec2(uval_mainImageSizeI)));
}

vec4 getOutermostScreenPixelCoords() {
    // The outermost visible screen pixels centers do not lie exactly on the boundary (+1 or -1), but are biased by
    // 0.5 screen pixel size inwards
    //
    //                                        2.0
    //    |<---------------------------------------------------------------------->|
    //
    //       2.0/Res
    //    |<--------->|
    //    |     X     |      X     |     X     |    ...    |     X     |     X     |
    //   -1     |                                                            |    +1
    //          |                                                            |
    //          |                                                            |
    //      -1 + 1.0/Res                                                  +1 - 1.0/Res
    //
    // Using shader macro is much more efficient than using constant buffer variable
    // because the compiler is able to optimize the code more aggressively
    return vec4(-1.0, -1.0, 1.0, 1.0) + vec4(1.0, 1.0, -1.0, -1.0) / uval_mainImageSizeI.xyxy;
}

vec4 temporalUpdate(vec4 prevData, vec3 currData, float maxFrames, ivec2 texelPos) {
    vec4 newResult = vec4(0.0);
    newResult.a = min(prevData.a + 1.0, maxFrames);
    newResult.rgb = mix(prevData.rgb, currData, 1.0 / newResult.a);
    newResult.rgb = dither_fp16(newResult.rgb, rand_stbnVec1(texelPos, frameCounter));
    return newResult;
}
const vec3 GROUND_ALBEDO_BASE = vec3(ivec3(SETTING_ATM_GROUND_ALBEDO_R, SETTING_ATM_GROUND_ALBEDO_G, SETTING_ATM_GROUND_ALBEDO_B)) / 255.0;

#endif
