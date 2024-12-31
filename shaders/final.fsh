#version 460 compatibility

#include "_Util.glsl"

uniform sampler2D colortex0;

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform sampler2D usam_rtwsm_imap2D;
uniform sampler2D usam_rtwsm_imap1D;
uniform sampler2D usam_rtwsm_warpingMap;
uniform sampler2D usam_transmittanceLUT;
uniform sampler2D usam_skyLUT;
uniform sampler2D usam_epipolarSliceEnd;
uniform sampler2D usam_epipolarInSctr;
uniform sampler2D usam_epipolarTransmittance;
uniform sampler2D usam_epipolarViewZ;

uniform sampler2D usam_temp3;

varying vec2 texcoord;

const float IMPORTANCE_MUL = uintBitsToFloat(0x4F800000u);

bool inViewPort(vec4 originSize, out vec2 texCoord) {
    vec2 min = originSize.xy;
    vec2 max = originSize.xy + originSize.zw;
    texCoord = (gl_FragCoord.xy - min) / originSize.zw;
    if (all(greaterThanEqual(gl_FragCoord.xy, min)) && all(lessThan(gl_FragCoord.xy, max))) {
        return true;
    }
    return false;
}

layout(location = 0) out vec4 rt_out;

vec3 gammaCorrect(vec3 color) {
    return pow(color, vec3(1.0 / SETTING_TONE_MAPPING_OUTPUT_GAMMA));
}

float gammaCorrect(float color) {
    return pow(color, float(1.0 / SETTING_TONE_MAPPING_OUTPUT_GAMMA));
}

void main() {
    ivec2 intTexCoord = ivec2(gl_FragCoord.xy);

    vec4 color = texelFetch(colortex0, intTexCoord, 0);

    vec2 debugTexCoord;
    #ifdef SETTING_DEBUG_RTWSM
    if (inViewPort(vec4(0, 0, 512, 512), debugTexCoord)) {
        color.rgb = pow(texture(shadowtex0, debugTexCoord).r, 2.0).rrr;
    }
//    if (inViewPort(vec4(512, 0, 512, 512), debugTexCoord)) {
//        color.rgb = pow(texture(shadowtex1, debugTexCoord).r, 2.0).rrr;
//    }
//    if (inViewPort(vec4(1024, 0, 512, 512), debugTexCoord)) {
//        color.rgb = texture(shadowcolor0, debugTexCoord).rgb;
//    }

    if (inViewPort(vec4(0, 512, 512, 512), debugTexCoord)) {
        color.rgb = gammaCorrect(texture(usam_rtwsm_imap2D, debugTexCoord).r * 2.0).rrr;
    }

    if (inViewPort(vec4(0, 1024, 512, 16), debugTexCoord)) {
        color.rgb = texture(usam_rtwsm_imap1D, vec2(debugTexCoord.x, 0.25)).rrr;
    }

    if (inViewPort(vec4(512, 512, 16, 512), debugTexCoord)) {
        color.rgb = texture(usam_rtwsm_imap1D, vec2(debugTexCoord.y, 0.75)).rrr;
    }

    if (inViewPort(vec4(0, 1024 + 16, 512, 16), debugTexCoord)) {
        vec2 v = texture(usam_rtwsm_warpingMap, vec2(debugTexCoord.x, 0.5)).rg;
        color.rgb = vec3(max(v.x, 0.0), max(-v.x, 0.0), 0.0);
    }

    if (inViewPort(vec4(512 + 16, 512, 16, 512), debugTexCoord)) {
        vec2 v = texture(usam_rtwsm_warpingMap, vec2(debugTexCoord.y, 0.75)).rg;
        color.rgb = vec3(max(v.x, 0.0), max(-v.x, 0.0), 0.0);
    }
    #endif

    #ifdef SETTING_DEBUG_ATMOSPHERE
    if (inViewPort(vec4(0, 0, 1024, 16), debugTexCoord)) {
        color.rgb = vec3(texture(usam_epipolarSliceEnd, vec2(debugTexCoord.x, 0.5)).rg, 0.0);
    }
    if (inViewPort(vec4(0, 16, 1024, 16), debugTexCoord)) {
        color.rgb = vec3(texture(usam_epipolarSliceEnd, vec2(debugTexCoord.x, 0.5)).ba, 0.0);
    }
    if (inViewPort(vec4(0, 32, 256, 64), debugTexCoord)) {
        color.rgb = gammaCorrect(texture(usam_transmittanceLUT, debugTexCoord).rgb);
    }
    if (inViewPort(vec4(0, 32 + 64, 256, 256), debugTexCoord)) {
        color.rgb = gammaCorrect(texture(usam_skyLUT, debugTexCoord).rgb * 0.1);
    }
    float whRatio = float(SETTING_EPIPOLAR_SLICES) / float(SETTING_SLICE_SAMPLES);
    if (inViewPort(vec4(256, 32, whRatio * 256, 256), debugTexCoord)) {
        debugTexCoord.y = 1.0 - debugTexCoord.y;
        color.rgb = gammaCorrect(texture(usam_epipolarInSctr, debugTexCoord).rgb);
    }
    if (inViewPort(vec4(256, 32 + 256, whRatio * 256, 256), debugTexCoord)) {
        debugTexCoord.y = 1.0 - debugTexCoord.y;
        color.rgb = gammaCorrect(texture(usam_epipolarTransmittance, debugTexCoord).rgb);
    }
    if (inViewPort(vec4(256, 32 + 512, whRatio * 256, 256), debugTexCoord)) {
        debugTexCoord.y = 1.0 - debugTexCoord.y;
        float depthV = texture(usam_epipolarViewZ, debugTexCoord).r;
        depthV = -depthV / far;
        color.rgb = gammaCorrect(depthV).rrr;
    }
    #endif

    #ifdef SETTING_DEBUG_TEMP3
    color.rgb = pow(texelFetch(usam_temp3, intTexCoord, 0).rgb * .1, vec3(1.0 / SETTING_TONE_MAPPING_OUTPUT_GAMMA));
    #endif

    rt_out = color;
}