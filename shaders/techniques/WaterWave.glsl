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
vec4 textureNice( sampler2D sam, vec2 uv )
{
    float textureResolution = float(textureSize(sam,0).x);
    uv = uv*textureResolution + 0.5;
    vec2 iuv = floor( uv );
    vec2 fuv = fract( uv );
    uv = iuv + fuv*fuv*(3.0-2.0*fuv);
    uv = (uv - 0.5)/textureResolution;
    return texture( sam, uv );
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
    vec2 waveTexCoord = waveTexCoordOg;
    vec2 curl = vec2(0.0);

//    {
//        FBMParameters curlParams;
//        curlParams.frequency = 1.0;
//        curlParams.persistence = 0.6;
//        curlParams.lacunarity = 2.5;
//        curlParams.octaveCount = 2u;
//        vec2 curlCoord = waveTexCoordOg * 3.478260869565217391304347826087;
//        float ff = fuck(timeV * 4.5 + 6.9);
//        curlCoord += ff * CURL_DIR * 0.048;
//        curlCoord -= waveTime * 3.36;
//        curl = GradientNoise_2D_grad_fbm(curlParams, MAT2_GOLDEN_ANGLE, curlCoord) * 0.003;
//    }

    waveTexCoord += waveTime * 0.06;

    float amp = 0.6;
    float wave = 0.0;
    if (base) {
        wave += amp * sampleNoise((waveTexCoord + curl - waveTime * 0.12) * vec2(0.7, 1.1) + curl * 0.84);
    }
    amp *= 0.8;
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 1.53 - waveTime * 0.085);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.63);
    if (base) {
        wave += amp * sampleNoise((waveTexCoord + curl + waveTime * 0.27) * vec2(0.7, 1.1) + curl * 1.8);
    }

    amp *= -0.37;
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 1.75 + waveTime * 1.21);
    curl = MAT2_GOLDEN_ANGLE * (curl * 2.67);
    {
        vec2 curlCoord2 = waveTexCoordOg;
        curlCoord2 += waveTime * 0.3082;
        curlCoord2 += (MAT2_GOLDEN_ANGLE * waveTime) * 0.4163;
        curlCoord2 *= 1.12;
        curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.04806;
    }

    if (base) {
        wave += amp * sampleNoise((waveTexCoord + curl + waveTime * 0.84) * vec2(-1.65, 0.93) + waveTime * 0.2);
    }
    amp *= -0.9;
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 1.14 + waveTime * 3.23);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.27);

    wave += amp * pow3(sampleNoise((waveTexCoord + curl) * vec2(-0.48, -0.19) + waveTime * vec2(0.08, -0.8))) * 0.4;

    amp *= 0.65;
    waveTexCoord = MAT2_GOLDEN_ANGLE * (waveTexCoord * 2.97 + waveTime * 2.17);
    curl = MAT2_GOLDEN_ANGLE * (curl * 1.19);

    {
        vec2 curlCoord2 = waveTexCoordOg * 0.61;

//        curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.135;

        curlCoord2 = MAT2_GOLDEN_ANGLE * curlCoord2;
        curlCoord2 += sin(timeV * 2.7 + 2.1) * 0.017 * CURL_DIR;
        curlCoord2 -= waveTime * 0.96;
        curlCoord2 -= (MAT2_GOLDEN_ANGLE * waveTime) * 0.53;
        curl += textureLod(usam_waveHFCurl, curlCoord2, 0.0).rg * 0.028;
    }

    wave += amp * sampleNoise((waveTexCoord + curl) * vec2(-0.85, -0.75) - waveTime * vec2(4.2, -7.12));

    return wave;
}

#endif