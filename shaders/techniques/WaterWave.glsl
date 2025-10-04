#ifndef INCLUDE_techniques_WaterWave_glsl
#define INCLUDE_techniques_WaterWave_glsl a

#include "/util/Mat2.glsl"
#include "/util/noise/GradientNoise.glsl"
#include "/util/Sampling.glsl"

float fuck(float x) {
    return cos(1.1 * x + 0.5) * 0.95 + sin(1.6 * x + 1.5) * 0.43 + cos(3.7 * x + 2.2) * 0.15;
}

float sampleNoise(vec2 coord) {
    return textureLod(usam_waveNoise, coord * 0.175, 0.0).r * 3.0;
}

float sampleNoiseR(vec2 coord) {
    return sampling_textureRepeat(usam_waveNoise, coord * 0.175, 0.01, 0.5).r * 3.0;
}
vec4 textureNice(sampler2D sam, vec2 uv)
{
    float textureResolution = float(textureSize(sam, 0).x);
    uv = uv*textureResolution + 0.5;
    vec2 iuv = floor(uv);
    vec2 fuv = fract(uv);
    uv = iuv + fuv*fuv*(3.0-2.0*fuv);
    uv = (uv - 0.5)/textureResolution;
    return texture(sam, uv);
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
        float ff = fuck(timeV * 4.5 + 6.9);
        curlCoord += ff * CURL_DIR * 0.0018;
        curlCoord += waveTime * 0.11;
        curl = textureLod(usam_waveHFCurl, curlCoord, 0.0).rg * 0.047;
    }

    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord + waveTime * -0.025);

    float amp = 0.1;
    float wave = 0.0;

    if (base) {
        wave += amp * sampleNoise((waveTexCoord + curl + waveTime * 0.27) * vec2(0.7, 1.1) + curl * 1.8);
    }

    amp *= -0.64;
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 2.23 + waveTime * 1.21);
    curl = MAT2_GOLDEN_ANGLE * (curl * 2.67);
    {
        waveTexCoordOgCurl = MAT2_GOLDEN_ANGLE * waveTexCoordOgCurl;
        vec2 curlCoord2 = waveTexCoordOgCurl * 1.12;
        curlCoord2 += waveTime * 0.62;
        curlCoord2 += (MAT2_GOLDEN_ANGLE * waveTime) * 0.37;
        curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.037;
    }

    wave += amp * sampleNoise((waveTexCoord + curl - waveTime * 1.23) * vec2(-0.85, 0.52));

    amp *= 0.31;
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 1.23 + waveTime * 2.52);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.51);

    wave += amp * pow4(sampleNoise((waveTexCoord + curl) * vec2(-0.45, -0.17) + waveTime * vec2(-0.14, 0.23)));

    amp *= -1.26;
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 3.73 + waveTime * 2.13);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.29);

    {
        waveTexCoordOgCurl = MAT2_GOLDEN_ANGLE * waveTexCoordOgCurl;
        vec2 curlCoord2 = waveTexCoordOgCurl * 2.43;

        curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.045;

        curlCoord2 = MAT2_GOLDEN_ANGLE * (curlCoord2 * 0.72);
        curlCoord2 += sin(timeV * 2.1 + 2.1) * 0.067 * CURL_DIR;
        curlCoord2 -= waveTime * 1.55;
        curlCoord2 -= (MAT2_GOLDEN_ANGLE * waveTime) * 0.63;
        curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.048;
    }

    wave += amp * sampleNoise((waveTexCoord + curl) * vec2(-0.57, -0.89) - waveTime * vec2(1.22, -1.69));

    return wave;
}

#endif