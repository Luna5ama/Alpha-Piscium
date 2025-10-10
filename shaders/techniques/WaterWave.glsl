#ifndef INCLUDE_techniques_WaterWave_glsl
#define INCLUDE_techniques_WaterWave_glsl a

#include "/util/Mat2.glsl"
#include "/util/noise/GradientNoise.glsl"
#include "/util/Sampling.glsl"

float fuck(float x) {
    return cos(1.1 * x + 0.5) * 0.95 + sin(1.6 * x + 1.5) * 0.43 + cos(3.7 * x + 2.2) * 0.15;
}

float sampleNoise(vec2 coord) {
    return textureLod(usam_waveNoise, coord * 0.175, 0.0).r;
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

float waveHeight(vec3 wavePos, bool base, bool detail) {
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
        curlCoord += waveTime * 0.21;
        curl = textureLod(usam_waveHFCurl, curlCoord, 0.0).rg * 0.072;
    }

    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord + waveTime * -0.025);

    float amp = 0.4;
    float wave = 0.0;

    if (base) {
        #ifdef DISTANT_HORIZONS
        wave += amp * sampleNoise((waveTexCoord + curl + waveTime * 0.31) * vec2(0.1, 0.06)) * 2.0;
        #endif
        wave += amp * sampleNoise((waveTexCoord + curl + waveTime * 0.29) * vec2(0.5, 0.8));
    }

    amp *= 0.25;
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 2.23);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.85);
    {
        waveTexCoordOgCurl = MAT2_GOLDEN_ANGLE * waveTexCoordOgCurl;
        vec2 curlCoord2 = waveTexCoordOgCurl * 0.72;
        curlCoord2 -= waveTime * 1.28;
        curlCoord2 -= (MAT2_GOLDEN_ANGLE * waveTime) * 1.57;
        curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.028;
    }

    wave += amp * sampleNoise((waveTexCoord + curl - waveTime * 0.86) * vec2(-0.85, 0.52));

    amp *= -2.28;
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 1.03 - waveTime * 2.17);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.75);

    wave += amp * pow2(sampleNoise((waveTexCoord + curl) * vec2(-0.13, -0.36))) * 2.0;

    if (detail){
        amp *= 0.17;
        waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 2.24 - waveTime * 2.78);
        curl = MAT2_GOLDEN_ANGLE * (curl * 1.15);

        {
            waveTexCoordOgCurl = MAT2_GOLDEN_ANGLE * waveTexCoordOgCurl;
            vec2 curlCoord2 = waveTexCoordOgCurl;
            curlCoord2 = curlCoord2 * 1.61;
            curlCoord2 += sin(timeV * 6.1 + 2.1) * 0.032 * CURL_DIR;
            curlCoord2 -= waveTime * 2.17;
            curlCoord2 -= (MAT2_GOLDEN_ANGLE * waveTime) * 1.43;
            curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.048;
        }

        wave += amp * sampleNoise((waveTexCoord + curl) * vec2(-0.57, -0.89) + waveTime * vec2(1.6, -5.8));
    }

    return wave;
}

#endif