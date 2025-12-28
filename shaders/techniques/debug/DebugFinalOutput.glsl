#include "Common.glsl"
#include "/util/TextRender.glsl"

void debugFinalOutput(ivec2 texelPos, inout vec4 outputColor) {
    _debug_texelPos = texelPos;
    vec2 debugTexCoord;

    beginText(texelPos >> ivec2(1), ivec2(4, (uval_mainImageSizeI.y >> 1) - 4));
    printLine();
    printLine();
    printLine();
    const int DEFAULT_FP_PRECISION = 4;
    text.fpPrecision = DEFAULT_FP_PRECISION;

    #ifdef SETTING_DEBUG_GI_TEXT
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
    printString((_r, _e, _u, _s, _e, _colon, _space));
    printBool(SPATIAL_REUSE == 1);
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

    printLine();

    printString((_D, _e, _n, _o, _i, _s, _e, _r, _space, _colon, _space));
    printBool(ENABLE_DENOISER == 1);
    printLine();

    printString((_D, _e, _n, _o, _i, _s, _e, _r, _space, _A, _c, _c, _u, _m, _space, _colon, _space));
    printBool(ENABLE_DENOISER_ACCUM == 1);
    printLine();

    printString((_D, _e, _n, _o, _i, _s, _e, _r, _space, _F, _a, _s, _t, _space, _C, _l, _a, _m, _p, _space, _colon, _space));
    printBool(ENABLE_DENOISER_FAST_CLAMP == 1);
    printLine();

    printString((_D, _e, _n, _o, _i, _s, _e, _r, _space, _A, _n, _t, _i, _space, _F, _i, _r, _e, _f, _l, _y, _space, _colon, _space));
    printBool(ENABLE_DENOISER_ANTI_FIREFLY == 1);
    printLine();

    printLine();

    printString((_D, _e, _n, _o, _i, _s, _e, _r, _space, _H, _i, _s, _t, _o, _r, _y, _space, _F, _i, _x, _space, _colon, _space));
    printBool(DENOISER_HISTORY_FIX == 1);
    printLine();
    #endif

    #ifdef SETTING_DEBUG_AE
    printString((_A, _u, _t, _o, _space, _E, _x, _p, _o, _s, _u, _r, _e));
    printLine();
    printString((_A, _v, _g, _space, _C, _o, _l, _o, _r, _colon, _space));
    printVec3(global_aeData.screenAvgLum.xyz);
    printLine();
    printString((_A, _v, _g, _space, _L, _u, _m, _a, _colon, _space));
    printFloat(global_aeData.screenAvgLum.w);
    printLine();
    printString((_H, _i, _g, _h, _l, _i, _g, _h, _t, _space, _P, _e, _r, _c, _e, _n, _t, _colon, _space));
    {
        text.fpPrecision = 2;
        printFloat(global_aeData.hsPercents.x * 100.0);
        printLine();
        printString((_S, _h, _a, _d, _o, _w, _space, _P, _e, _r, _c, _e, _n, _t, _colon, _space));
        printFloat(global_aeData.hsPercents.y * 100.0);
        printLine();
        text.fpPrecision = DEFAULT_FP_PRECISION;
    }
    printLine();
    printString((_E, _V));
    printLine();
    printString((_A, _V, _G, _colon, _space));
    printFloat(global_aeData.expValues.x);
    printLine();
    printString((_H, _I, _S, _colon, _space));
    printFloat(global_aeData.expValues.y);
    printLine();
    printString((_M, _I, _X, _colon, _space));
    printFloat(global_aeData.expValues.z);
    printLine();
    printLine();
    if (inViewPort(ivec4(0, 0, 1024, 256), debugTexCoord)) {
        uint binIndex = min(uint(debugTexCoord.x * 256.0), 255u);
        float binCount = float(global_aeData.lumHistogram[binIndex]);
        float maxBinCount = float(global_aeData.lumHistogramMaxBinCount);
        float percentage = binCount / maxBinCount;
        if (debugTexCoord.y < percentage) {
            outputColor.rgb = interpolateTurbo(percentage);
        } else {
            outputColor.rgb = vec3(0.25);
        }
    }
    #endif

    endText(outputColor.rgb);
}
