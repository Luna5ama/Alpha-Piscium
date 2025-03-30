#include "/util/FullScreenComp.glsl"
#include "/rtwsm/RTWSM.glsl"
#include "/atmosphere/Common.glsl"
#include "/general/EnvProbe.glsl"
#include "/util/NZPacking.glsl"
#include "/denoiser/Common.glsl"

#ifdef SETTING_DEBUG_RTWSM
uniform sampler2D shadowtex0;
uniform sampler2D usam_rtwsm_imap;
#endif

#ifdef SETTING_DEBUG_ATMOSPHERE
uniform sampler2D usam_skyLUT;
uniform sampler2D usam_epipolarSliceEnd;
uniform usampler2D usam_epipolarData;
#endif

#ifdef SETTING_DEBUG_ENV_PROBE
uniform usampler2D usam_envProbe;
#endif

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;

#if SETTING_DEBUG_TEMP_TEX == 1
#define DEBUG_TEX_NAME usam_temp1
#elif SETTING_DEBUG_TEMP_TEX == 2
#define DEBUG_TEX_NAME usam_temp2
#elif SETTING_DEBUG_TEMP_TEX == 3
#define DEBUG_TEX_NAME usam_temp3
#elif SETTING_DEBUG_TEMP_TEX == 4
#define DEBUG_TEX_NAME usam_temp4
#elif SETTING_DEBUG_TEMP_TEX == 5
#define DEBUG_TEX_NAME usam_temp5
#elif SETTING_DEBUG_TEMP_TEX == 6
#define DEBUG_TEX_NAME usam_temp6
#elif SETTING_DEBUG_TEMP_TEX == 7
#define DEBUG_TEX_NAME usam_temp7
#endif


#if SETTING_DEBUG_SVGF != 0
uniform usampler2D usam_svgfHistory;
#endif

#ifdef DEBUG_TEX_NAME
uniform sampler2D DEBUG_TEX_NAME;
#endif

#if SETTING_DEBUG_GI_INPUTS != 0
uniform usampler2D usam_tempRG32UI;
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

#ifdef SETTING_DEBUG_GAMMA_CORRECT
float gammaCorrect(float color) {
    return pow(color, float(1.0 / SETTING_TONE_MAPPING_OUTPUT_GAMMA));
}

vec3 gammaCorrect(vec3 color) {
    return pow(color, vec3(1.0 / SETTING_TONE_MAPPING_OUTPUT_GAMMA));
}

vec4 gammaCorrect(vec4 color) {
    return vec4(gammaCorrect(color.rgb), color.a);
}
#else
float gammaCorrect(float color) {
    return color;
}

vec3 gammaCorrect(vec3 color) {
    return color;
}

vec4 gammaCorrect(vec4 color) {
    return color;
}
#endif

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

        outputColor = gammaCorrect(outputColor);
    }
    #endif

    GBufferData gData;
    gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferData8UN, texelPos, 0), gData);
    Material material = material_decode(gData);

    #if SETTING_DEBUG_GBUFFER_DATA == 1
    outputColor.rgb = material.albedo;
    #elif SETTING_DEBUG_GBUFFER_DATA == 2 || SETTING_DEBUG_NORMAL_MODE == 3

    #if SETTING_DEBUG_GBUFFER_DATA == 2
    outputColor.rgb = gData.normal;
    #else
    outputColor.rgb = gData.geometryNormal;
    #endif
    #if SETTING_DEBUG_NORMAL_MODE == 0
    outputColor.rgb = mat3(gbufferModelViewInverse) * outputColor.rgb;
    #endif
    outputColor.rgb = outputColor.rgb * 0.5 + 0.5;

    #elif SETTING_DEBUG_GBUFFER_DATA == 4
    outputColor.rgb = vec3(material.roughness);
    #elif SETTING_DEBUG_GBUFFER_DATA == 5
    outputColor.rgb = vec3(material.f0);
    #elif SETTING_DEBUG_GBUFFER_DATA == 6
    outputColor.rgb = vec3(material.porosity);
    #elif SETTING_DEBUG_GBUFFER_DATA == 7
    outputColor.rgb = vec3(material.sss);
    #elif SETTING_DEBUG_GBUFFER_DATA == 8
    outputColor.rgb = vec3(gData.lmCoord.x);
    #elif SETTING_DEBUG_GBUFFER_DATA == 9
    outputColor.rgb = vec3(gData.lmCoord.y);
    #elif SETTING_DEBUG_GBUFFER_DATA == 10
    outputColor.rgb = vec3(float(gData.isHand));
    #endif

    #if SETTING_DEBUG_GI_INPUTS != 0
    if (all(lessThan(texelPos, global_mipmapSizesI[1]))) {
        uvec2 radianceData = texelFetch(usam_tempRG32UI, texelPos+ ivec2(0, global_mipmapSizesI[1].y), 0).xy;
        vec4 radiance = vec4(unpackHalf2x16(radianceData.x), unpackHalf2x16(radianceData.y));

        #if SETTING_DEBUG_GI_INPUTS == 1
        outputColor.rgb = radiance.rgb;
        #elif SETTING_DEBUG_GI_INPUTS == 2
        outputColor.rgb = vec3(abs(radiance.a));
        #elif SETTING_DEBUG_GI_INPUTS == 3
        outputColor.rgb = saturate(sign(radiance.a));
        #endif


        float prevZ;
        vec3 prevN;
        nzpacking_unpack(texelFetch(usam_tempRG32UI, texelPos, 0).xy, prevN, prevZ);
        #if SETTING_DEBUG_GI_INPUTS == 4
        outputColor.rgb = vec3(prevN * 0.5 + 0.5);
        #elif SETTING_DEBUG_GI_INPUTS == 5
        outputColor.rgb = linearStep(near, far, -prevZ).rrr;
        #endif
    }
    #endif

    #if SETTING_DEBUG_SVGF != 0
    uvec4 svgfData = texelFetch(usam_svgfHistory, texelPos >> 1, 0);
    vec4 svgfColorHLen;
    vec2 svgfMoments;
    svgf_unpack(svgfData, svgfColorHLen, svgfMoments);
    #if SETTING_DEBUG_SVGF == 1
    outputColor.rgb = svgfColorHLen.rgb;
    #elif SETTING_DEBUG_SVGF == 2
    outputColor.rgb = vec3(svgfColorHLen.a / SETTING_DENOISER_MAX_ACCUM);
    #elif SETTING_DEBUG_SVGF == 3
    outputColor.rgb = svgfMoments.xxx;
    #elif SETTING_DEBUG_SVGF == 4
    outputColor.rgb = svgfMoments.yyy;
    #endif
    #endif

    vec2 debugTexCoord;

    #ifdef SETTING_DEBUG_RTWSM
    if (inViewPort(ivec4(0, 0, 512, 512), debugTexCoord)) {
        outputColor.rgb = pow(texture(shadowtex0, debugTexCoord).r * 0.5, 2.0).rrr;
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
        ScatteringResult sampleResult;
        float viewZ;
        unpackEpipolarData(texture(usam_epipolarData, debugTexCoord), sampleResult, viewZ);
        outputColor.rgb = gammaCorrect(sampleResult.inScattering * exp2(SETTING_DEBUG_EXP));
    }
    if (inViewPort(ivec4(256, 32 + 256, whRatio * 256, 256), debugTexCoord)) {
        debugTexCoord.y = 1.0 - debugTexCoord.y;
        ScatteringResult sampleResult;
        float viewZ;
        unpackEpipolarData(texture(usam_epipolarData, debugTexCoord), sampleResult, viewZ);
        outputColor.rgb = gammaCorrect(sampleResult.transmittance);
    }
    if (inViewPort(ivec4(256, 32 + 512, whRatio * 256, 256), debugTexCoord)) {
        debugTexCoord.y = 1.0 - debugTexCoord.y;
        ScatteringResult sampleResult;
        float viewZ;
        unpackEpipolarData(texture(usam_epipolarData, debugTexCoord), sampleResult, viewZ);
        float depthV = -viewZ.r / far;
        outputColor.rgb = gammaCorrect(depthV).rrr;
    }
    #endif

    #ifdef SETTING_DEBUG_ENV_PROBE
    if (inViewPort(ivec4(0, 0, 512, 512), debugTexCoord)) {
        ivec2 texelPos = ivec2(debugTexCoord * ENV_PROBE_SIZE);
        EnvProbeData envProbeData = envProbe_decode(texelFetch(usam_envProbe, texelPos, 0));
        bool envProbeIsSky = envProbe_isSky(envProbeData);
        outputColor.rgb = envProbeIsSky ? vec3(0.0) : envProbeData.radiance;
        outputColor.rgb *= exp2(SETTING_DEBUG_EXP);
        outputColor.rgb = gammaCorrect(outputColor.rgb);
    }
    if (inViewPort(ivec4(0, 512, 512, 512), debugTexCoord)) {
        ivec2 texelPos = ivec2(debugTexCoord * ENV_PROBE_SIZE);
        EnvProbeData envProbeData = envProbe_decode(texelFetch(usam_envProbe, texelPos, 0));
        bool envProbeIsSky = envProbe_isSky(envProbeData);
        outputColor.rgb = envProbeIsSky ? vec3(0.0, 0.0, 1.0) : vec3(saturate(length(envProbeData.scenePos) / far));
    }
    if (inViewPort(ivec4(512, 0, 512, 512), debugTexCoord)) {
        ivec2 texelPos = ivec2(debugTexCoord * ENV_PROBE_SIZE);
        EnvProbeData envProbeData = envProbe_decode(texelFetch(usam_envProbe, texelPos, 0));
        outputColor.rgb = envProbeData.normal * 0.5 + 0.5;
    }
    if (inViewPort(ivec4(512, 512, 512, 512), debugTexCoord)) {
        ivec2 texelPos = ivec2(debugTexCoord * ENV_PROBE_SIZE);
        texelPos.x += 512;
        EnvProbeData envProbeData = envProbe_decode(texelFetch(usam_envProbe, texelPos, 0));
        bool envProbeIsSky = envProbe_isSky(envProbeData);
        outputColor.rgb = envProbeIsSky ? vec3(0.0) : envProbeData.radiance;
        outputColor.rgb *= exp2(SETTING_DEBUG_EXP);
        outputColor.rgb = gammaCorrect(outputColor.rgb);
    }
    #endif
}