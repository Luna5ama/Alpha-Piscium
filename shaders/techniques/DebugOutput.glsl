#include "/util/BitPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"
#include "/util/Celestial.glsl"
#include "/util/NZPacking.glsl"
#include "/util/TextRender.glsl"
#include "/techniques/EnvProbe.glsl"
#include "/techniques/atmospherics/air/Common.glsl"
#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/svgf/Common.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"
#include "/techniques/atmospherics/clouds/ss/Common.glsl"

uniform sampler2D usam_debug;

// By Belmu
const vec3 turboCurve[] = vec3[](vec3(0.18995, 0.07176, 0.23217), vec3(0.19483, 0.08339, 0.26149), vec3(0.19956, 0.09498, 0.29024), vec3(0.20415, 0.10652, 0.31844), vec3(0.20860, 0.11802, 0.34607), vec3(0.21291, 0.12947, 0.37314), vec3(0.21708, 0.14087, 0.39964), vec3(0.22111, 0.15223, 0.42558), vec3(0.22500, 0.16354, 0.45096), vec3(0.22875, 0.17481, 0.47578), vec3(0.23236, 0.18603, 0.50004), vec3(0.23582, 0.19720, 0.52373), vec3(0.23915, 0.20833, 0.54686), vec3(0.24234, 0.21941, 0.56942), vec3(0.24539, 0.23044, 0.59142), vec3(0.24830, 0.24143, 0.61286), vec3(0.25107, 0.25237, 0.63374), vec3(0.25369, 0.26327, 0.65406), vec3(0.25618, 0.27412, 0.67381), vec3(0.25853, 0.28492, 0.69300), vec3(0.26074, 0.29568, 0.71162), vec3(0.26280, 0.30639, 0.72968), vec3(0.26473, 0.31706, 0.74718), vec3(0.26652, 0.32768, 0.76412), vec3(0.26816, 0.33825, 0.78050), vec3(0.26967, 0.34878, 0.79631), vec3(0.27103, 0.35926, 0.81156), vec3(0.27226, 0.36970, 0.82624), vec3(0.27334, 0.38008, 0.84037), vec3(0.27429, 0.39043, 0.85393), vec3(0.27509, 0.40072, 0.86692), vec3(0.27576, 0.41097, 0.87936), vec3(0.27628, 0.42118, 0.89123), vec3(0.27667, 0.43134, 0.90254), vec3(0.27691, 0.44145, 0.91328), vec3(0.27701, 0.45152, 0.92347), vec3(0.27698, 0.46153, 0.93309), vec3(0.27680, 0.47151, 0.94214), vec3(0.27648, 0.48144, 0.95064), vec3(0.27603, 0.49132, 0.95857), vec3(0.27543, 0.50115, 0.96594), vec3(0.27469, 0.51094, 0.97275), vec3(0.27381, 0.52069, 0.97899), vec3(0.27273, 0.53040, 0.98461), vec3(0.27106, 0.54015, 0.98930), vec3(0.26878, 0.54995, 0.99303), vec3(0.26592, 0.55979, 0.99583), vec3(0.26252, 0.56967, 0.99773), vec3(0.25862, 0.57958, 0.99876), vec3(0.25425, 0.58950, 0.99896), vec3(0.24946, 0.59943, 0.99835), vec3(0.24427, 0.60937, 0.99697), vec3(0.23874, 0.61931, 0.99485), vec3(0.23288, 0.62923, 0.99202), vec3(0.22676, 0.63913, 0.98851), vec3(0.22039, 0.64901, 0.98436), vec3(0.21382, 0.65886, 0.97959), vec3(0.20708, 0.66866, 0.97423), vec3(0.20021, 0.67842, 0.96833), vec3(0.19326, 0.68812, 0.96190), vec3(0.18625, 0.69775, 0.95498), vec3(0.17923, 0.70732, 0.94761), vec3(0.17223, 0.71680, 0.93981), vec3(0.16529, 0.72620, 0.93161), vec3(0.15844, 0.73551, 0.92305), vec3(0.15173, 0.74472, 0.91416), vec3(0.14519, 0.75381, 0.90496), vec3(0.13886, 0.76279, 0.89550), vec3(0.13278, 0.77165, 0.88580), vec3(0.12698, 0.78037, 0.87590), vec3(0.12151, 0.78896, 0.86581), vec3(0.11639, 0.79740, 0.85559), vec3(0.11167, 0.80569, 0.84525), vec3(0.10738, 0.81381, 0.83484), vec3(0.10357, 0.82177, 0.82437), vec3(0.10026, 0.82955, 0.81389), vec3(0.09750, 0.83714, 0.80342), vec3(0.09532, 0.84455, 0.79299), vec3(0.09377, 0.85175, 0.78264), vec3(0.09287, 0.85875, 0.77240), vec3(0.09267, 0.86554, 0.76230), vec3(0.09320, 0.87211, 0.75237), vec3(0.09451, 0.87844, 0.74265), vec3(0.09662, 0.88454, 0.73316), vec3(0.09958, 0.89040, 0.72393), vec3(0.10342, 0.89600, 0.71500), vec3(0.10815, 0.90142, 0.70599), vec3(0.11374, 0.90673, 0.69651), vec3(0.12014, 0.91193, 0.68660), vec3(0.12733, 0.91701, 0.67627), vec3(0.13526, 0.92197, 0.66556), vec3(0.14391, 0.92680, 0.65448), vec3(0.15323, 0.93151, 0.64308), vec3(0.16319, 0.93609, 0.63137), vec3(0.17377, 0.94053, 0.61938), vec3(0.18491, 0.94484, 0.60713), vec3(0.19659, 0.94901, 0.59466), vec3(0.20877, 0.95304, 0.58199), vec3(0.22142, 0.95692, 0.56914), vec3(0.23449, 0.96065, 0.55614), vec3(0.24797, 0.96423, 0.54303), vec3(0.26180, 0.96765, 0.52981), vec3(0.27597, 0.97092, 0.51653), vec3(0.29042, 0.97403, 0.50321), vec3(0.30513, 0.97697, 0.48987), vec3(0.32006, 0.97974, 0.47654), vec3(0.33517, 0.98234, 0.46325), vec3(0.35043, 0.98477, 0.45002), vec3(0.36581, 0.98702, 0.43688), vec3(0.38127, 0.98909, 0.42386), vec3(0.39678, 0.99098, 0.41098), vec3(0.41229, 0.99268, 0.39826), vec3(0.42778, 0.99419, 0.38575), vec3(0.44321, 0.99551, 0.37345), vec3(0.45854, 0.99663, 0.36140), vec3(0.47375, 0.99755, 0.34963), vec3(0.48879, 0.99828, 0.33816), vec3(0.50362, 0.99879, 0.32701), vec3(0.51822, 0.99910, 0.31622), vec3(0.53255, 0.99919, 0.30581), vec3(0.54658, 0.99907, 0.29581), vec3(0.56026, 0.99873, 0.28623), vec3(0.57357, 0.99817, 0.27712), vec3(0.58646, 0.99739, 0.26849), vec3(0.59891, 0.99638, 0.26038), vec3(0.61088, 0.99514, 0.25280), vec3(0.62233, 0.99366, 0.24579), vec3(0.63323, 0.99195, 0.23937), vec3(0.64362, 0.98999, 0.23356), vec3(0.65394, 0.98775, 0.22835), vec3(0.66428, 0.98524, 0.22370), vec3(0.67462, 0.98246, 0.21960), vec3(0.68494, 0.97941, 0.21602), vec3(0.69525, 0.97610, 0.21294), vec3(0.70553, 0.97255, 0.21032), vec3(0.71577, 0.96875, 0.20815), vec3(0.72596, 0.96470, 0.20640), vec3(0.73610, 0.96043, 0.20504), vec3(0.74617, 0.95593, 0.20406), vec3(0.75617, 0.95121, 0.20343), vec3(0.76608, 0.94627, 0.20311), vec3(0.77591, 0.94113, 0.20310), vec3(0.78563, 0.93579, 0.20336), vec3(0.79524, 0.93025, 0.20386), vec3(0.80473, 0.92452, 0.20459), vec3(0.81410, 0.91861, 0.20552), vec3(0.82333, 0.91253, 0.20663), vec3(0.83241, 0.90627, 0.20788), vec3(0.84133, 0.89986, 0.20926), vec3(0.85010, 0.89328, 0.21074), vec3(0.85868, 0.88655, 0.21230), vec3(0.86709, 0.87968, 0.21391), vec3(0.87530, 0.87267, 0.21555), vec3(0.88331, 0.86553, 0.21719), vec3(0.89112, 0.85826, 0.21880), vec3(0.89870, 0.85087, 0.22038), vec3(0.90605, 0.84337, 0.22188), vec3(0.91317, 0.83576, 0.22328), vec3(0.92004, 0.82806, 0.22456), vec3(0.92666, 0.82025, 0.22570), vec3(0.93301, 0.81236, 0.22667), vec3(0.93909, 0.80439, 0.22744), vec3(0.94489, 0.79634, 0.22800), vec3(0.95039, 0.78823, 0.22831), vec3(0.95560, 0.78005, 0.22836), vec3(0.96049, 0.77181, 0.22811), vec3(0.96507, 0.76352, 0.22754), vec3(0.96931, 0.75519, 0.22663), vec3(0.97323, 0.74682, 0.22536), vec3(0.97679, 0.73842, 0.22369), vec3(0.98000, 0.73000, 0.22161), vec3(0.98289, 0.72140, 0.21918), vec3(0.98549, 0.71250, 0.21650), vec3(0.98781, 0.70330, 0.21358), vec3(0.98986, 0.69382, 0.21043), vec3(0.99163, 0.68408, 0.20706), vec3(0.99314, 0.67408, 0.20348), vec3(0.99438, 0.66386, 0.19971), vec3(0.99535, 0.65341, 0.19577), vec3(0.99607, 0.64277, 0.19165), vec3(0.99654, 0.63193, 0.18738), vec3(0.99675, 0.62093, 0.18297), vec3(0.99672, 0.60977, 0.17842), vec3(0.99644, 0.59846, 0.17376), vec3(0.99593, 0.58703, 0.16899), vec3(0.99517, 0.57549, 0.16412), vec3(0.99419, 0.56386, 0.15918), vec3(0.99297, 0.55214, 0.15417), vec3(0.99153, 0.54036, 0.14910), vec3(0.98987, 0.52854, 0.14398), vec3(0.98799, 0.51667, 0.13883), vec3(0.98590, 0.50479, 0.13367), vec3(0.98360, 0.49291, 0.12849), vec3(0.98108, 0.48104, 0.12332), vec3(0.97837, 0.46920, 0.11817), vec3(0.97545, 0.45740, 0.11305), vec3(0.97234, 0.44565, 0.10797), vec3(0.96904, 0.43399, 0.10294), vec3(0.96555, 0.42241, 0.09798), vec3(0.96187, 0.41093, 0.09310), vec3(0.95801, 0.39958, 0.08831), vec3(0.95398, 0.38836, 0.08362), vec3(0.94977, 0.37729, 0.07905), vec3(0.94538, 0.36638, 0.07461), vec3(0.94084, 0.35566, 0.07031), vec3(0.93612, 0.34513, 0.06616), vec3(0.93125, 0.33482, 0.06218), vec3(0.92623, 0.32473, 0.05837), vec3(0.92105, 0.31489, 0.05475), vec3(0.91572, 0.30530, 0.05134), vec3(0.91024, 0.29599, 0.04814), vec3(0.90463, 0.28696, 0.04516), vec3(0.89888, 0.27824, 0.04243), vec3(0.89298, 0.26981, 0.03993), vec3(0.88691, 0.26152, 0.03753), vec3(0.88066, 0.25334, 0.03521), vec3(0.87422, 0.24526, 0.03297), vec3(0.86760, 0.23730, 0.03082), vec3(0.86079, 0.22945, 0.02875), vec3(0.85380, 0.22170, 0.02677), vec3(0.84662, 0.21407, 0.02487), vec3(0.83926, 0.20654, 0.02305), vec3(0.83172, 0.19912, 0.02131), vec3(0.82399, 0.19182, 0.01966), vec3(0.81608, 0.18462, 0.01809), vec3(0.80799, 0.17753, 0.01660), vec3(0.79971, 0.17055, 0.01520), vec3(0.79125, 0.16368, 0.01387), vec3(0.78260, 0.15693, 0.01264), vec3(0.77377, 0.15028, 0.01148), vec3(0.76476, 0.14374, 0.01041), vec3(0.75556, 0.13731, 0.00942), vec3(0.74617, 0.13098, 0.00851), vec3(0.73661, 0.12477, 0.00769), vec3(0.72686, 0.11867, 0.00695), vec3(0.71692, 0.11268, 0.00629), vec3(0.70680, 0.10680, 0.00571), vec3(0.69650, 0.10102, 0.00522), vec3(0.68602, 0.09536, 0.00481), vec3(0.67535, 0.08980, 0.00449), vec3(0.66449, 0.08436, 0.00424), vec3(0.65345, 0.07902, 0.00408), vec3(0.64223, 0.07380, 0.00401), vec3(0.63082, 0.06868, 0.00401), vec3(0.61923, 0.06367, 0.00410), vec3(0.60746, 0.05878, 0.00427), vec3(0.59550, 0.05399, 0.00453), vec3(0.58336, 0.04931, 0.00486), vec3(0.57103, 0.04474, 0.00529), vec3(0.55852, 0.04028, 0.00579), vec3(0.54583, 0.03593, 0.00638), vec3(0.53295, 0.03169, 0.00705), vec3(0.51989, 0.02756, 0.00780), vec3(0.50664, 0.02354, 0.00863), vec3(0.49321, 0.01963, 0.00955), vec3(0.47960, 0.01583, 0.01055));
vec3 interpolateTurbo(float x) {
    if (x < 0.0) {
        return vec3(0.0);
    } else if (x > 1.0) {
        return vec3(1.0);
    }
    x *= 255.0;
    return turboCurve[int(x)] + (turboCurve[min(255, int(x) + 1)] - turboCurve[int(x)]) * fract(x);
}

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
#define DEBUG_TEX_NAME usam_overlays
#elif SETTING_DEBUG_TEMP_TEX == 7
#define DEBUG_TEX_NAME usam_geometryNormal
#endif

ivec2 _debug_texelPos = ivec2(0);

bool inViewPort(ivec4 originSize, out vec2 texCoord) {
    originSize = ivec4(vec4(originSize) * SETTING_DEBUG_SCALE);
    ivec2 min = originSize.xy;
    ivec2 max = originSize.xy + originSize.zw;
    texCoord = saturate((vec2(_debug_texelPos - min) + 0.5) / vec2(originSize.zw));
    if (all(greaterThanEqual(_debug_texelPos.xy, min)) && all(lessThan(_debug_texelPos.xy, max))) {
        return true;
    }
    return false;
}

float applyExposure(float color) {
    return color * exp2(SETTING_DEBUG_EXP);
}

vec2 applyExposure(vec2 color) {
    return color * exp2(SETTING_DEBUG_EXP);
}

vec3 applyExposure(vec3 color) {
    return color * exp2(SETTING_DEBUG_EXP);
}

vec4 applyExposure(vec4 color) {
    return color * exp2(SETTING_DEBUG_EXP);
}

#if defined(SETTING_DEBUG_GAMMA_CORRECT) && (SETTING_DEBUG_OUTPUT != 1)
float gammaCorrect(float color) {
    return colors_sRGB_encodeGamma(color);
}

vec2 gammaCorrect(vec2 color) {
    return colors_sRGB_encodeGamma(color);
}

vec3 gammaCorrect(vec3 color) {
    return colors_sRGB_encodeGamma(color);
}

vec4 gammaCorrect(vec4 color) {
    return vec4(colors_sRGB_encodeGamma(color.rgb), color.a);
}
#else
float gammaCorrect(float color) {
    return color;
}

vec2 gammaCorrect(vec2 color) {
    return color;
}

vec3 gammaCorrect(vec3 color) {
    return color;
}

vec4 gammaCorrect(vec4 color) {
    return color;
}
#endif

float expGamma(float color) {
    return gammaCorrect(applyExposure(color));
}

vec2 expGamma(vec2 color) {
    return gammaCorrect(applyExposure(color));
}

vec3 expGamma(vec3 color) {
    return gammaCorrect(applyExposure(color));
}

vec4 expGamma(vec4 color) {
    return gammaCorrect(applyExposure(color));
}

vec3 displayViewZ(float viewZ) {
    #ifdef DISTANT_HORIZONS
    float viewZRemapped = linearStep(-near, -dhFarPlane, viewZ);
    #else
    float viewZRemapped = linearStep(-near, -far, viewZ);
    #endif
    return interpolateTurbo(sqrt(viewZRemapped));
}

void debugOutput(ivec2 texelPos, inout vec4 outputColor) {
    _debug_texelPos = texelPos;
    beginText(texelPos >> ivec2(2), ivec2(0, uval_mainImageSizeI.y >> 2));
    printLine();
    printLine();
    text.fpPrecision = 4;

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

    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
    Material material = material_decode(gData);

    #if SETTING_DEBUG_GBUFFER_DATA == 1
    outputColor.rgb = displayViewZ(texelFetch(usam_gbufferViewZ, texelPos, 0).r);
    #elif SETTING_DEBUG_GBUFFER_DATA == 2
    outputColor.rgb = gammaCorrect(material.albedo);
    #elif SETTING_DEBUG_GBUFFER_DATA == 3 || SETTING_DEBUG_GBUFFER_DATA == 4

    #if SETTING_DEBUG_GBUFFER_DATA == 3
    outputColor.rgb = gData.normal;
    #else
    outputColor.rgb = gData.geomNormal;
    #endif

    #if SETTING_DEBUG_NORMAL_MODE == 0
    outputColor.rgb = mat3(gbufferModelViewInverse) * outputColor.rgb;
    #endif
    outputColor.r = linearStep(-SETTING_DEBUG_NORMAL_X_RANGE, SETTING_DEBUG_NORMAL_X_RANGE, outputColor.r);
    outputColor.g = linearStep(-SETTING_DEBUG_NORMAL_Y_RANGE, SETTING_DEBUG_NORMAL_Y_RANGE, outputColor.g);
    outputColor.b = linearStep(-SETTING_DEBUG_NORMAL_Z_RANGE, SETTING_DEBUG_NORMAL_Z_RANGE, outputColor.b);

    #elif SETTING_DEBUG_GBUFFER_DATA == 5
    outputColor.rgb = vec3(material.roughness);
    #elif SETTING_DEBUG_GBUFFER_DATA == 6
    outputColor.rgb = vec3(material.f0);
    #elif SETTING_DEBUG_GBUFFER_DATA == 7
    outputColor.rgb = vec3(material.porosity);
    #elif SETTING_DEBUG_GBUFFER_DATA == 8
    outputColor.rgb = vec3(material.sss);
    #elif SETTING_DEBUG_GBUFFER_DATA == 9
    outputColor.rgb = vec3(gData.pbrSpecular.a);
    #elif SETTING_DEBUG_GBUFFER_DATA == 10
    outputColor.rgb = vec3(gData.lmCoord.x);
    #elif SETTING_DEBUG_GBUFFER_DATA == 11
    outputColor.rgb = vec3(gData.lmCoord.y);
    #elif SETTING_DEBUG_GBUFFER_DATA == 12
    outputColor.rgb = vec3(float(gData.isHand));
    #endif

    #if SETTING_DEBUG_DENOISER != 0
//    uvec4 svgfData = texelFetch(usam_csrgba32ui, texelPos, 0);
    vec3 svgfColor;
    vec3 svgfFastColor;
    vec2 svgfMoments;
    float svgfHLen;
    svgf_unpack(svgfData, svgfColor, svgfFastColor, svgfMoments, svgfHLen);
    #if SETTING_DEBUG_DENOISER == 1
    outputColor.rgb = expGamma(svgfColor);
    #elif SETTING_DEBUG_DENOISER == 2
    outputColor.rgb = expGamma(svgfFastColor);
    #elif SETTING_DEBUG_DENOISER == 3
    outputColor.rgb = interpolateTurbo(1.0 - (svgfHLen - 2.0) / (SETTING_DENOISER_MAX_ACCUM - 2.0));
    #elif SETTING_DEBUG_DENOISER == 4
    outputColor.rgb = svgfMoments.xxx;
    #elif SETTING_DEBUG_DENOISER == 5
    outputColor.rgb = svgfMoments.yyy;
    #elif SETTING_DEBUG_DENOISER == 6
    outputColor.rgb = vec3(max(svgfMoments.g - svgfMoments.r * svgfMoments.r, 0.0));
    #endif
    #endif

    #if SETTING_DEBUG_GI_INPUTS != 0
    if (all(lessThan(texelPos, global_mipmapSizesI[1]))) {
        uvec2 radianceData = transient_packedZN_fetch(texelPos+ ivec2(0, global_mipmapSizesI[1].y)).xy;
        vec4 radiance = vec4(unpackHalf2x16(radianceData.x), unpackHalf2x16(radianceData.y));

        #if SETTING_DEBUG_GI_INPUTS == 1
        outputColor.rgb = radiance.rgb * 4.0;
        #elif SETTING_DEBUG_GI_INPUTS == 2
        outputColor.rgb = vec3(abs(radiance.a));
        #elif SETTING_DEBUG_GI_INPUTS == 3
        outputColor.rgb = vec3(saturate(sign(radiance.a)));
        #endif


        float prevZ;
        vec3 prevN;
        nzpacking_unpack(transient_packedZN_fetch(texelPos).xy, prevN, prevZ);
        #if SETTING_DEBUG_GI_INPUTS == 4
        outputColor.rgb = vec3(prevN * 0.5 + 0.5);
        #elif SETTING_DEBUG_GI_INPUTS == 5
        outputColor.rgb = displayViewZ(prevZ);
        #endif
    }
    #endif
    #if SETTING_DEBUG_GI_INPUTS == 6
//    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
//        uint packedGeometryNormal = texelFetch(usam_geometryNormal, texelPos, 0).r;
//        vec3 geometryNormal = unpackSnorm3x10(packedGeometryNormal);
//        outputColor.rgb = vec3(geometryNormal * 0.5 + 0.5);
//    }
    #endif

    vec2 screenPos = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
    vec2 debugTexCoord;

    #ifdef SETTING_DEBUG_RTWSM
    printIvec3(global_shadowAABBMin);
    printLine();
    printLine();
    printIvec3(global_shadowAABBMax);

    if (inViewPort(ivec4(0, 0, 512, 512), debugTexCoord)) {
        float linearDepth = rtwsm_linearDepth(texture(shadowtex0, debugTexCoord).r);
        float remappedDepth = linearStep(-global_shadowAABBMaxPrev.z, -global_shadowAABBMinPrev.z, linearDepth);
        outputColor.rgb = vec3(remappedDepth);
    }
    if (inViewPort(ivec4(0, 512, 512, 512), debugTexCoord)) {
        debugTexCoord.y = min(debugTexCoord.y * IMAP2D_V_RANGE, IMAP2D_V_CLAMP);
        outputColor.rgb = gammaCorrect(texture(usam_rtwsm_imap, debugTexCoord).r * 0.5).rrr;
    }
    if (inViewPort(ivec4(0, 1024 + 4, 512, 16), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_rtwsm_imap, vec2(debugTexCoord.x, IMAP1D_X_V)).r * 0.1).rrr;
    }
    if (inViewPort(ivec4(512 + 4, 512, 16, 512), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_rtwsm_imap, vec2(debugTexCoord.y, IMAP1D_Y_V)).r * 0.1).rrr;
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
    const float EPIPOLAR_SLICE_END_POINTS_V = 0.5 / float(EPIPOLAR_DATA_Y_SIZE);
    if (inViewPort(ivec4(0, 0, 1024, 16), debugTexCoord)) {
        outputColor.rgb = vec3(uintBitsToFloat(texture(usam_epipolarData, vec2(debugTexCoord.x, EPIPOLAR_SLICE_END_POINTS_V)).rg), 0.0);
    }
    if (inViewPort(ivec4(0, 16, 1024, 16), debugTexCoord)) {
        outputColor.rgb = vec3(uintBitsToFloat(texture(usam_epipolarData, vec2(debugTexCoord.x, EPIPOLAR_SLICE_END_POINTS_V)).ba), 0.0);
    }
    if (inViewPort(ivec4(0, 32, 256, 64), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_transmittanceLUT, debugTexCoord).rgb);
    }
    if (inViewPort(ivec4(0, 32 + 64, 256, 256), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_skyLUT, debugTexCoord).rgb * exp2(SETTING_DEBUG_EXP));
    }
    if (inViewPort(ivec4(0, 32 + 64 + 256, 256, 256), debugTexCoord)) {
        outputColor.rgb = gammaCorrect(texture(usam_multiSctrLUT, debugTexCoord).rgb);
    }
    float whRatio = float(SETTING_EPIPOLAR_SLICES) / float(SETTING_SLICE_SAMPLES);
    if (inViewPort(ivec4(256, 32, whRatio * 512, 768), debugTexCoord)) {
        ScatteringResult sampleResult;
        float viewZ;
        vec2 sampleTexCoord = debugTexCoord;
        sampleTexCoord.y = mix(1.0 / EPIPOLAR_DATA_Y_SIZE, 1.0, sampleTexCoord.y);
        unpackEpipolarData(texture(usam_epipolarData, sampleTexCoord), sampleResult, viewZ);
        outputColor.rgb = gammaCorrect(sampleResult.inScattering * exp2(SETTING_DEBUG_EXP));
    }
//    if (inViewPort(ivec4(256, 32 + 256, whRatio * 256, 256), debugTexCoord)) {
//        ScatteringResult sampleResult;
//        float viewZ;
//        vec2 sampleTexCoord = debugTexCoord;
//        sampleTexCoord.y = mix(1.0 / EPIPOLAR_DATA_Y_SIZE, 1.0, sampleTexCoord.y);
//        unpackEpipolarData(texture(usam_epipolarData, sampleTexCoord), sampleResult, viewZ);
//        outputColor.rgb = gammaCorrect(sampleResult.transmittance);
//    }
//    if (inViewPort(ivec4(256, 32 + 512, whRatio * 256, 256), debugTexCoord)) {
//        ScatteringResult sampleResult;
//        float viewZ;
//        unpackEpipolarData(texture(usam_epipolarData, debugTexCoord), sampleResult, viewZ);
//        float depthV = -viewZ.r / far;
//        outputColor.rgb = gammaCorrect(depthV).rrr;
//    }
    #endif

    #ifdef SETTING_DEBUG_SKY_VIEW_LUT
    for (int i = 0; i < SKYVIEW_LUT_LAYERS; i++) {
        if (inViewPort(ivec4(i * 256, 0, 256, 256), debugTexCoord)) {
        outputColor.rgb = expGamma(_atmospherics_air_lut_sampleSkyViewSlice(debugTexCoord, 0.0 + float(i * 3)));
        }
        if (inViewPort(ivec4(i * 256, 256, 256, 256), debugTexCoord)) {
            outputColor.rgb = expGamma(_atmospherics_air_lut_sampleSkyViewSlice(debugTexCoord, 1.0 + float(i * 3)));
        }
        if (inViewPort(ivec4(i * 256, 512, 256, 256), debugTexCoord)) {
            outputColor.rgb = gammaCorrect(_atmospherics_air_lut_sampleSkyViewSlice(debugTexCoord, 2.0 + float(i * 3)));
        }
    }
    #endif

    #ifdef SETTING_DEBUG_CLOUDS_AMBLUT
    #define CLOUDS_AMB_LUT_SIZE 128
    for (int i = 0; i < 6; i++) {
        if (inViewPort(ivec4(0, CLOUDS_AMB_LUT_SIZE * i, CLOUDS_AMB_LUT_SIZE, CLOUDS_AMB_LUT_SIZE), debugTexCoord)) {
            vec3 lutCoord = vec3(debugTexCoord, (float(i) + 0.5) / 6.0);
            outputColor.rgb = gammaCorrect(applyExposure(texture(usam_cloudsAmbLUT, lutCoord).rgb));
        }
        if (inViewPort(ivec4(CLOUDS_AMB_LUT_SIZE, CLOUDS_AMB_LUT_SIZE * i, CLOUDS_AMB_LUT_SIZE, CLOUDS_AMB_LUT_SIZE), debugTexCoord)) {
            vec3 lutCoord = vec3(debugTexCoord, (float(i) + 0.5) / 6.0);
            outputColor.rgb = interpolateTurbo(texture(usam_cloudsAmbLUT, lutCoord).a);
        }
    }
    #endif

    #if SETTING_DEBUG_CLOUDS_SS
    {
        CloudSSHistoryData historyData = clouds_ss_historyData_init();
//#        clouds_ss_historyData_unpack(history_lowCloud_fetch(texelPos), historyData);
        #if SETTING_DEBUG_CLOUDS_SS == 1
        outputColor.rgb = expGamma(historyData.inScattering);
        #elif SETTING_DEBUG_CLOUDS_SS == 2
        outputColor.rgb = expGamma(historyData.transmittance);
        #elif SETTING_DEBUG_CLOUDS_SS == 3
        outputColor.rgb = interpolateTurbo(linearStep(0.0, 64.0, historyData.viewZ));
        #elif SETTING_DEBUG_CLOUDS_SS == 4
        outputColor.rgb = interpolateTurbo(1.0 - (historyData.hLen - 2.0) / (float(CLOUDS_SS_MAX_ACCUM) - 2.0));
        #endif
    }
    #endif

    #ifdef SETTING_DEBUG_ENV_PROBE
    if (inViewPort(ivec4(0, 0, 512, 768), debugTexCoord)) {
        ivec2 texelPos = ivec2(debugTexCoord * ENV_PROBE_SIZE * vec2(2.0, 3.0));
        EnvProbeData envProbeData = envProbe_decode(texelFetch(usam_envProbe, texelPos, 0));
        bool envProbeIsSky = envProbe_isSky(envProbeData);
        outputColor.rgb = envProbeIsSky ? vec3(0.0) : envProbeData.radiance;
        outputColor.rgb *= exp2(SETTING_DEBUG_EXP);
        outputColor.rgb = gammaCorrect(outputColor.rgb);
    }
//    if (inViewPort(ivec4(0, 512, 512, 512), debugTexCoord)) {
//        ivec2 texelPos = ivec2(debugTexCoord * ENV_PROBE_SIZE);
//        EnvProbeData envProbeData = envProbe_decode(texelFetch(usam_envProbe, texelPos, 0));
//        bool envProbeIsSky = envProbe_isSky(envProbeData);
//        outputColor.rgb = envProbeIsSky ? vec3(0.0, 0.0, 1.0) : vec3(saturate(length(envProbeData.scenePos) / far));
//    }
//    if (inViewPort(ivec4(512, 0, 512, 512), debugTexCoord)) {
//        ivec2 texelPos = ivec2(debugTexCoord * ENV_PROBE_SIZE);
//        EnvProbeData envProbeData = envProbe_decode(texelFetch(usam_envProbe, texelPos, 0));
//        outputColor.rgb = envProbeData.normal * 0.5 + 0.5;
//    }
//    if (inViewPort(ivec4(512, 512, 512, 512), debugTexCoord)) {
//        ivec2 texelPos = ivec2(debugTexCoord * ENV_PROBE_SIZE);
//        texelPos.x += 512;
//        EnvProbeData envProbeData = envProbe_decode(texelFetch(usam_envProbe, texelPos, 0));
//        bool envProbeIsSky = envProbe_isSky(envProbeData);
//        outputColor.rgb = envProbeIsSky ? vec3(0.0) : envProbeData.radiance;
//        outputColor.rgb *= exp2(SETTING_DEBUG_EXP);
//        outputColor.rgb = gammaCorrect(outputColor.rgb);
//    }
    #endif

    #ifdef SETTING_DEBUG_AE
    printChar(_A);
    printChar(_V);
    printChar(_G);
    printChar(_space);
    printFloat(global_aeData.expValues.x);
    printLine();
    printChar(_H);
    printChar(_I);
    printChar(_S);
    printChar(_space);
    printFloat(global_aeData.expValues.y);
    printLine();
    printChar(_M);
    printChar(_I);
    printChar(_X);
    printChar(_space);
    printFloat(global_aeData.expValues.z);
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

    #ifdef SETTING_DEBUG_STARMAP
    outputColor.rgb = gammaCorrect(colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_WORKING_COLORSPACE, colors_LogLuv32ToSRGB(texture(usam_starmap, screenPos))));
    #endif

    #ifdef SETTING_DEBUG_EPIPOLAR_LINES
    {
        vec2 ndcPos = screenPos * 2.0 - 1.0;
        vec2 f2RayDir = normalize(ndcPos - uval_sunNdcPos);
        vec4 f4Boundaries = getOutermostScreenPixelCoords();
        vec4 f4HalfSpaceEquationTerms = (ndcPos.xxyy - f4Boundaries.xzyw) * f2RayDir.yyxx;
        uvec4 b4HalfSpaceFlags = uvec4(lessThan(f4HalfSpaceEquationTerms.xyyx, f4HalfSpaceEquationTerms.zzww));
        uvec4 b4SectorFlags = b4HalfSpaceFlags.wxyz & (1u - b4HalfSpaceFlags.xyzw);
        vec4 f4DistToBoundaries = (f4Boundaries - uval_sunNdcPos.xyxy) / (f2RayDir.xyxy + vec4(lessThan(abs(f2RayDir.xyxy), vec4(1e-6))));
        float fDistToExitBoundary = dot(vec4(b4SectorFlags), f4DistToBoundaries);
        vec2 f2ExitPoint = uval_sunNdcPos + f2RayDir * fDistToExitBoundary;
        vec4 f4EpipolarSlice = vec4(0.0, 0.25, 0.5, 0.75) +
        saturate((f2ExitPoint.yxyx - f4Boundaries.wxyz) * vec4(-1.0, 1.0, 1.0, -1.0) / (f4Boundaries.wzwz - f4Boundaries.yxyx)) / 4.0;
        float fEpipolarSlice = dot(vec4(b4SectorFlags), f4EpipolarSlice);
        float fEpipolarSliceIndex = fEpipolarSlice * SETTING_EPIPOLAR_SLICES;
        fEpipolarSliceIndex = fract(fEpipolarSliceIndex + 0.5);
        float lineWidth = SETTING_EPIPOLAR_SLICES / min2(uval_mainImageSize);
        lineWidth *= 0.25;
        lineWidth = saturate(lineWidth / distance(uval_sunNdcPos, ndcPos) * (1.0 + length(uval_sunNdcPos)));
        float lineAlpha = smoothstep(0.5 - lineWidth, 0.5, fEpipolarSliceIndex);
        lineAlpha *= smoothstep(0.5 + lineWidth, 0.5, fEpipolarSliceIndex);
        outputColor.rgb = mix(outputColor.rgb, vec3(1.0, 0.5, 0.5), lineAlpha);
    }
    #endif

    #ifdef SETTING_DEBUG_DEDICATED
    outputColor = expGamma(texelFetch(usam_debug, ivec2((vec2(texelPos) + 0.5) / SETTING_DEBUG_SCALE), 0));
    #endif

//    beginText(texelPos >> ivec2(2), ivec2(0, uval_mainImageSizeI.y >> 2));
//    printFloat(global_turbidity);
    endText(outputColor.rgb);
}