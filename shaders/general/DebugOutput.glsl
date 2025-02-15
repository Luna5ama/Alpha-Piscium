#include "../_Util.glsl"
#include "../util/FullScreenComp.glsl"
#include "../rtwsm/RTWSM.glsl"

#ifdef SETTING_DEBUG_RTWSM
uniform sampler2D shadowtex0;
uniform sampler2D usam_rtwsm_imap;
#endif

#ifdef SETTING_DEBUG_ATMOSPHERE
uniform sampler2D usam_transmittanceLUT;
uniform sampler2D usam_multiSctrLUT;
uniform sampler2D usam_skyLUT;
uniform sampler2D usam_epipolarSliceEnd;
uniform sampler2D usam_epipolarInSctr;
uniform sampler2D usam_epipolarTransmittance;
uniform sampler2D usam_epipolarViewZ;
#endif

uniform usampler2D usam_gbufferData;

#if SETTING_DEBUG_TEX_NAME == 1
#define DEBUG_TEX_NAME usam_temp1
#elif SETTING_DEBUG_TEX_NAME == 2
#define DEBUG_TEX_NAME usam_temp2
#elif SETTING_DEBUG_TEX_NAME == 3
#define DEBUG_TEX_NAME usam_temp3
#elif SETTING_DEBUG_TEX_NAME == 4
#define DEBUG_TEX_NAME usam_temp4
#elif SETTING_DEBUG_SSVBIL == 1
#define DEBUG_TEX_NAME usam_ssvbil
#ifdef SETTING_DEBUG_ALPHA
#undef SETTING_DEBUG_ALPHA a
#endif
#elif SETTING_DEBUG_SSVBIL == 2
#define DEBUG_TEX_NAME usam_ssvbil
#ifndef SETTING_DEBUG_ALPHA
#define SETTING_DEBUG_ALPHA a
#endif
#endif

#ifdef DEBUG_TEX_NAME
uniform sampler2D DEBUG_TEX_NAME;
#endif

bool inViewPort(ivec4 originSize, out vec2 texCoord) {
    ivec2 min = originSize.xy;
    ivec2 max = originSize.xy + originSize.zw;
    texCoord = saturate((vec2(texelPos - min) + 0.5) / vec2(originSize.zw));
    if (all(greaterThanEqual(texelPos.xy, min)) && all(lessThan(texelPos.xy, max))) {
        return true;
    }
    return false;
}

float gammaCorrect(float color) {
    return pow(color, float(1.0 / SETTING_TONE_MAPPING_OUTPUT_GAMMA));
}

vec3 gammaCorrect(vec3 color) {
    return pow(color, vec3(1.0 / SETTING_TONE_MAPPING_OUTPUT_GAMMA));
}

vec4 gammaCorrect(vec4 color) {
    return vec4(gammaCorrect(color.rgb), color.a);
}

void debugOutput(inout vec4 outputColor) {
    #ifdef DEBUG_TEX_NAME
    if (all(lessThan(texelPos, textureSize(DEBUG_TEX_NAME, 0)))) {
        outputColor = texelFetch(DEBUG_TEX_NAME, texelPos, 0);
        outputColor *= exp2(SETTING_DEBUG_EXP);

        #ifdef SETTING_DEBUG_NEGATE
        outputColor = -outputColor;
        #endif

        #ifdef SETTING_DEBUG_ALPHA
        outputColor.rgb = outputColor.aaa;
        #else
        outputColor.a = 1.0;
        #endif

        #ifdef SETTING_DEBUG_GAMMA_CORRECT
        outputColor = gammaCorrect(outputColor);
        #endif
    }
    #endif

    GBufferData gData;
    gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);

    #if SETTING_DEBUG_NORMAL != 0
    outputColor.rgb = gData.normal;
    #if SETTING_DEBUG_NORMAL == 1
    outputColor.rgb = mat3(gbufferModelViewInverse) * outputColor.rgb;
    #endif
    outputColor.rgb = outputColor.rgb * 0.5 + 0.5;
    #endif

    vec2 debugTexCoord;

    #ifdef SETTING_DEBUG_RTWSM
    if (inViewPort(ivec4(0, 0, 512, 512), debugTexCoord)) {
        outputColor.rgb = pow(texture(shadowtex0, debugTexCoord).r, 2.0).rrr;
    }
    if (inViewPort(ivec4(0, 512, 512, 512), debugTexCoord)) {
        debugTexCoord.y = min(debugTexCoord.y * IMAP2D_V_RANGE, IMAP2D_V_CLAMP);
        outputColor.rgb = gammaCorrect(texture(usam_rtwsm_imap, debugTexCoord).r * 2.0).rrr;
    }
    if (inViewPort(ivec4(0, 1024 + 4, 512, 16), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_rtwsm_imap, vec2(debugTexCoord.x, IMAP1D_X_V)).r * 2.0).rrr;
    }
    if (inViewPort(ivec4(512 + 4, 512, 16, 512), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_rtwsm_imap, vec2(debugTexCoord.y, IMAP1D_Y_V)).r * 2.0).rrr;
    }
    if (inViewPort(ivec4(0, 1024 + 4 + 16 + 4, 512, 16), debugTexCoord)) {
        float v = texture(usam_rtwsm_imap, vec2(debugTexCoord.x, WARP_X_V)).r;
        outputColor.rgb = vec3(max(v, 0.0), max(-v, 0.0), 0.0);
    }
    if (inViewPort(ivec4(512 + 4 + 16 + 4, 512, 16, 512), debugTexCoord)) {
        float v = texture(usam_rtwsm_imap, vec2(debugTexCoord.y, WARP_Y_V)).r;
        outputColor.rgb = vec3(max(v, 0.0), max(-v, 0.0), 0.0);
    }
    #endif

    #ifdef SETTING_DEBUG_ATMOSPHERE
    if (inViewPort(ivec4(0, 0, 1024, 16), debugTexCoord)) {
        outputColor.rgb = vec3(texture(usam_epipolarSliceEnd, vec2(debugTexCoord.x, 0.5)).rg, 0.0);
    }
    if (inViewPort(ivec4(0, 16, 1024, 16), debugTexCoord)) {
        outputColor.rgb = vec3(texture(usam_epipolarSliceEnd, vec2(debugTexCoord.x, 0.5)).ba, 0.0);
    }
    if (inViewPort(ivec4(0, 32, 256, 64), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_transmittanceLUT, debugTexCoord).rgb);
    }
    if (inViewPort(ivec4(0, 32 + 64, 256, 256), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_skyLUT, debugTexCoord).rgb * 0.1);
    }
    if (inViewPort(ivec4(0, 32 + 64 + 256, 256, 256), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_multiSctrLUT, debugTexCoord).rgb * 10.0);
    }
    float whRatio = float(SETTING_EPIPOLAR_SLICES) / float(SETTING_SLICE_SAMPLES);
    if (inViewPort(ivec4(256, 32, whRatio * 256, 256), debugTexCoord)) {
        debugTexCoord.y = 1.0 - debugTexCoord.y;
        outputColor.rgb = gammaCorrect(texture(usam_epipolarInSctr, debugTexCoord).rgb);
    }
    if (inViewPort(ivec4(256, 32 + 256, whRatio * 256, 256), debugTexCoord)) {
        debugTexCoord.y = 1.0 - debugTexCoord.y;
        outputColor.rgb = gammaCorrect(texture(usam_epipolarTransmittance, debugTexCoord).rgb);
    }
    if (inViewPort(ivec4(256, 32 + 512, whRatio * 256, 256), debugTexCoord)) {
        debugTexCoord.y = 1.0 - debugTexCoord.y;
        float depthV = texture(usam_epipolarViewZ, debugTexCoord).r;
        depthV = -depthV / far;
        outputColor.rgb = gammaCorrect(depthV).rrr;
    }
    #endif
}