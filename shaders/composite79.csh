#version 460 compatibility

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/util/FullScreenComp.glsl"
#include "/techniques/DebugOutput.glsl"

layout(rgba16f) restrict uniform writeonly image2D uimg_temp1;

#define FFXCAS_SHARPENESS SETTING_TAA_CAS_SHARPNESS
#include "/techniques/ffx/FFXCas.glsl"

vec3 ffxcas_load(ivec2 texelPos) {
    return texelFetch(usam_main, texelPos, 0).rgb;
}

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);
        if (FFXCAS_SHARPENESS > 0.0) {
            outputColor.rgb = ffxcas_pass(texelPos);
        }
        #if SETTING_DEBUG_OUTPUT == 3
        debugOutput(texelPos, outputColor);
        #endif
        #ifdef SETTING_DOF_SHOW_FOCUS_PLANE
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        float alpha = float(viewZ < -global_focusDistance);
        outputColor.rgb = mix(outputColor.rgb, vec3(1.0, 0.0, 1.0), alpha * 0.25);
        #endif

        beginText(texelPos >> ivec2(1), ivec2(4, (uval_mainImageSizeI.y >> 1) - 4));
        printLine();
        printLine();
        printLine();
        text.fpPrecision = 4;

        printString((_M, _e, _t, _h, _o, _d, _colon, _space));
        #if USE_REFERENCE == 0
        printString((_R, _e, _S, _T, _I, _R));
        printLine();

        printString((_S, _p, _a, _t, _i, _a, _l, _space));
        printString((_r, _e, _u, _s, _e, _space));
        printString((_v, _i, _s, _i, _b, _i, _l, _i, _t, _y, _space));
        printString((_t, _r, _a, _c, _e, _colon, _space));
        #if SPATIAL_REUSE_VISIBILITY_TRACE == 0
        printString((_n, _o, _n, _e));
        #elif SPATIAL_REUSE_VISIBILITY_TRACE == 1
        printString((_c, _o, _m, _b, _i, _n, _e, _d));
        #else
        printString((_f, _u, _l, _l));
        #endif
        printLine();

        printString((_S, _p, _a, _t, _i, _a, _l, _space));
        printString((_r, _e, _u, _s, _e, _space));
        printString((_s, _a, _m, _p, _l, _e, _s, _colon, _space));
        printInt(SPATIAL_REUSE_SAMPLES);
        printLine();

        printString((_S, _p, _a, _t, _i, _a, _l, _space));
        printString((_r, _e, _u, _s, _e, _space));
        printString((_r, _a, _d, _i, _u, _s, _colon, _space));
        printInt(SPATIAL_REUSE_RADIUS);
        printLine();

        printString((_S, _p, _a, _t, _i, _a, _l, _space));
        printString((_r, _e, _u, _s, _e, _space));
        printString((_f, _e, _e, _d, _b, _a, _c, _k, _colon, _space));
        printInt(SPATIAL_REUSE_FEEDBACK);
        printLine();


        #elif USE_REFERENCE == 1
        printString((_M, _o, _n, _t, _e, _space, _C, _a, _r, _l, _o));
        printLine();

        printString((_S, _P, _P, _colon, _space));
        printInt(MC_SPP);
        printLine();
        printLine();
        printLine();
        printLine();


        #elif USE_REFERENCE == 2
        printString((_V, _B, _G, _I));
        printLine();
        printString((_S, _t, _e, _p, _space, _c, _o, _u, _n, _t, _colon, _space));
        printInt(SSVBIL_SAMPLE_STEPS222);
        printLine();
        printLine();
        printLine();
        printLine();
        #endif
        printLine();

        printString((_M, _a, _x, _space, _F, _r, _a, _m, _e, _s, _colon, _space));
        printInt(MAX_FRAMES);
        printLine();

        int fCount = clamp(RANDOM_FRAME, 0, MAX_FRAMES);
        printString((_C, _u, _r, _r, _e, _n, _t, _space, _F, _r, _a, _m, _e, _colon, _space));
        printInt(fCount);
        printLine();

        endText(outputColor.rgb);

        imageStore(uimg_temp1, texelPos, outputColor);
    }
}