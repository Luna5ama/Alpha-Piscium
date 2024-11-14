#version 460 compatibility

#define GLOBAL_DATA_MODIFIER
#include "_Util.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

layout(rgba16f) restrict uniform image2D uimg_main;

mat4 shadowDeRotateMatrix(mat4 shadowMatrix) {
    vec2 p1 = (shadowMatrix * vec4(0.0, -1000.0, 0.0, 1.0)).xy;
    vec2 p2 = (shadowMatrix * vec4(0.0, 1000.0, 0.0, 1.0)).xy;

    float angle1 = -atan(p1.y, p1.x);

    float cos1 = cos(angle1 - PI_HALF_CONST) * 0.9;
    float sin1 = sin(angle1 - PI_HALF_CONST) * 0.9;

    return mat4(
            cos1, sin1, 0.0, 0.0,
            -sin1, cos1, 0.0, 0.0,
            0.0, 0.0, 0.1, 0.0,
            0.0, 0.0, 0.0, 1.0
    );
}

vec2 taaJitter() {
    return r2Seq2(frameCounter) - 0.5;
}

mat4 taaJitterMat(vec2 baseJitter) {
    vec2 jitter = baseJitter * 2.0 * viewResolution.zw;
    return mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            jitter.x, jitter.y, 0.0, 1.0
    );
}

void main() {
    vec2 jitter = taaJitter();
    global_shadowRotationMatrix = shadowDeRotateMatrix(shadowModelView);
    global_taaJitter = jitter;
    global_taaJitterMat = taaJitterMat(jitter);

    #ifdef SETTING_REAL_SUN_TEMPERATURE
    vec4 sunRadiance = colors_blackBodyRadiation(5772, OMEGA_SUN);
    #else
    vec4 sunRadiance = colors_blackBodyRadiation(SETTING_SUN_TEMPERATURE, OMEGA_SUN);
    #endif
    sunRadiance.a *= 683.002; // Radiance to luminance conversion factor
    global_sunRadiance = sunRadiance;

    ivec2 mainImgSize = imageSize(uimg_main);

    float top5Percent = float(mainImgSize.x * mainImgSize.y / 20);
    float topBin = float(max(global_topBinCount, 1));
    global_topBinCount = 0u;

    float expLast = global_exposure;
    float expNew = (top5Percent / topBin) * expLast;
    const float timeCoeff = 0.01;
    expNew = clamp(expNew, 0.00001, 1.0);
    expNew = mix(expLast, expNew, timeCoeff);
    global_exposure = expNew;
}