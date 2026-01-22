#ifndef INCLUDE_techniques_WaterWave_glsl
#define INCLUDE_techniques_WaterWave_glsl a

#include "/util/Mat2.glsl"
#include "/util/noise/GradientNoise.glsl"
#include "/util/Sampling.glsl"

float oscillate(float x) {
    return cos(1.1 * x + 0.5) * 0.95 + sin(1.6 * x + 1.5) * 0.43 + cos(3.7 * x + 2.2) * 0.15;
}

float sampleNoise(vec2 coord) {
    return textureLod(usam_waveNoise, coord * 0.175, 0.0).r;
}

const float WAVE_POS_BASE = 0.023;

float waveHeight(vec3 wavePos, bool base) {
    const vec2 WAVE_DIR = vec2(0.777, -0.555);
    const vec2 CURL_DIR = vec2(-0.21, 0.15);

    #ifndef SETTING_SCREENSHOT_MODE
    float timeV = frameTimeCounter;
    #else
    float timeV = 13.37;
    #endif

    vec2 waveTime = vec2(0.0);
    waveTime += timeV * WAVE_DIR * 0.1;

    vec2 waveTexCoordOg = wavePos.xz + wavePos.y * 0.07;
    vec2 waveTexCoordOgCurl = waveTexCoordOg;
    vec2 waveTexCoord = waveTexCoordOg;
    vec2 curl = vec2(0.0);

    {
        vec2 curlCoord = waveTexCoordOgCurl * 0.11;
        float ff = oscillate(timeV * 4.5 + 6.9);
        curlCoord += ff * CURL_DIR * 0.0018;
        curlCoord += waveTime * 0.11;
        curl = textureLod(usam_waveHFCurl, curlCoord, 0.0).rg * 0.061;
    }

    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord + waveTime * -0.025);

    float wave = 0.0;

    if (base) {
        #ifdef DISTANT_HORIZONS
        wave += 0.4 * sampleNoise((waveTexCoord + curl + waveTime * 0.31) * vec2(0.1, 0.06)) * 2.0; // 0.8
        #endif
        wave += 0.4 * sampleNoise((waveTexCoord + curl + waveTime * 0.29) * vec2(0.5, 0.8)); // 1.2
    }

    // base [0.0, 1.2]

    // detail [-0.83, 0.0]
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 2.23);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.85);
    {
        waveTexCoordOgCurl = MAT2_GOLDEN_ANGLE * waveTexCoordOgCurl;
        vec2 curlCoord2 = waveTexCoordOgCurl * 0.45;
        curlCoord2 -= waveTime * 0.78;
        curlCoord2 -= (MAT2_GOLDEN_ANGLE * waveTime) * 1.57;
        curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.026;
    }

    wave += -0.16 * sampleNoise((waveTexCoord + curl - waveTime * 0.96) * vec2(-0.85, 0.52)); // -0.16

    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 1.33 - waveTime * 1.87);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.35);

    wave += -0.59 * (sampleNoise((waveTexCoord + curl) * vec2(-0.13, -0.36))); // -0.75

    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 1.88 - waveTime * 1.58);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.35);

    {
        waveTexCoordOgCurl = MAT2_GOLDEN_ANGLE * waveTexCoordOgCurl;
        vec2 curlCoord2 = waveTexCoordOgCurl;
        curlCoord2 = curlCoord2 * 0.98;
        curlCoord2 += sin(timeV * 6.1 + 2.1) * 0.042 * CURL_DIR;
        curlCoord2 -= waveTime * 2.17;
        curlCoord2 -= (MAT2_GOLDEN_ANGLE * waveTime) * 1.63;
        curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.068;
    }
    wave += -0.07 * sampleNoise((waveTexCoord + curl) * vec2(-0.57, -0.69) + waveTime * vec2(3.2, -5.1)); // -0.82

    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 1.87 - waveTime * 1.68);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.73);

    wave += -0.05 * sampleNoise((waveTexCoord + curl) * vec2(0.4, 0.6) - waveTime * vec2(1.1, 1.1)); // -0.86

    return wave;
}

#endif