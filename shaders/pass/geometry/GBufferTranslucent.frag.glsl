ivec2 texelPos;
#include "/util/Colors2.glsl"
#include "/util/Dither.glsl"
#include "/techniques/Lighting.glsl"
#include "/techniques/WaterWave.glsl"

layout(r32i) uniform iimage2D uimg_translucentDepthLayers;

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

in vec4 frag_viewTangent;

in vec4 frag_colorMul;
in vec3 frag_viewNormal;
in vec2 frag_texCoord;
in vec2 frag_lmCoord;
flat in uint frag_materialID;

in float frag_viewZ;

vec3 viewPos = vec3(0.0);
float fuckO = 0.0;

#ifdef GBUFFER_PASS_DH
/* RENDERTARGETS:5 */
layout(location = 0) out vec4 rt_translucentColor;
#else
/* RENDERTARGETS:8,9,11 */
layout(location = 0) out uvec4 rt_gbufferData1;
layout(location = 1) out uvec4 rt_gbufferData2;
layout(location = 2) out vec4 rt_translucentColor;
#endif

vec4 processAlbedo() {
    vec4 albedo = frag_colorMul;
    if (frag_materialID != 3u) {
        #ifdef SETTING_SCREENSHOT_MODE
        albedo *= textureLod(gtexture, frag_texCoord, 0.0);
        #else
        albedo *= texture(gtexture, frag_texCoord);
        #endif
    }
    #ifdef SETTING_DEBUG_WHITE_WORLD
    return vec4(1.0);
    #else
    return albedo;
    #endif
}

GBufferData processOutput() {
    float bitangentSignF = frag_viewTangent.w < 0.0 ? -1.0 : 1.0;
    vec3 geomViewNormal = normalize(frag_viewNormal);
    vec3 geomViewTangent = normalize(frag_viewTangent.xyz);
    vec3 geomViewBitangent = normalize(cross(geomViewNormal, geomViewTangent) * bitangentSignF);

    GBufferData gData = gbufferData_init();
    gData.geomNormal = geomViewNormal;
    gData.geomTangent = geomViewTangent;
    gData.bitangentSign = int(bitangentSignF);

    float noiseIGN = rand_IGN(texelPos, frameCounter);

    #ifdef DISTANT_HORIZONS
    #ifndef GBUFFER_PASS_DH
    float edgeFactor = linearStep(min(far * 0.75, far - 24.0), far, length(frag_viewCoord));
    if (noiseIGN < edgeFactor) {
        discard;
    }
    #endif
    #endif

    #ifdef SETTING_SCREENSHOT_MODE
    vec4 normalSample = textureLod(normals, frag_texCoord, 0.0);
    vec4 specularSample = textureLod(specular, frag_texCoord, 0.0);
    #else
    vec4 normalSample = texture(normals, frag_texCoord);
    vec4 specularSample = texture(specular, frag_texCoord);
    #endif

    gData.pbrSpecular = specularSample;
    gData.lmCoord.y *= normalSample.b;

    const float _1o255 = 1.0 / 255.0;
    float emissiveS = linearStep(_1o255, 1.0, specularSample.a);
    emissiveS *= float(specularSample.a < 1.0);

    gData.pbrSpecular.a = emissiveS;

    #if !defined(SETTING_NORMAL_MAPPING)
    gData.normal = geomViewNormal;
    #else
    mat3 tbn = mat3(geomViewTangent, geomViewBitangent, geomViewNormal);
    vec3 tangentNormal;

    if (frag_materialID == 3u) {
        vec3 scenePos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;

        vec3 cameraPosWaveSpace = vec3(cameraPositionInt >> 5) + ldexp(vec3(cameraPositionInt & ivec3(31)), ivec3(-5));
        cameraPosWaveSpace = cameraPositionFract * WAVE_POS_BASE + cameraPosWaveSpace * 0.736;

        vec3 waveWorldPos = scenePos * WAVE_POS_BASE + cameraPosWaveSpace;

        vec3 viewDir = normalize(-viewPos);

        #ifdef SETTING_WATER_PARALLEX
        const uint MAX_STEPS = uint(SETTING_WATER_PARALLEX_STEPS);
        const float PARALLEX_STRENGTH = float(SETTING_WATER_PARALLEX_STRENGTH);

        vec3 rayDir = scenePos / abs(scenePos.y);
        rayDir.xz *= WAVE_POS_BASE * PARALLEX_STRENGTH;
        const float MAX_WAVE_HEIGHT = 1.7;
        float rayStepLength = MAX_WAVE_HEIGHT / float(MAX_STEPS);

        for (uint i = 0u; i < MAX_STEPS; i++) {
            float fi = float(i) + 0.5;
            vec3 sampleDelta = rayDir * fi * rayStepLength;

            vec3 samplePos = waveWorldPos + sampleDelta;
            samplePos.y = waveWorldPos.y;
            float sampleHeight = waveHeight(samplePos, false);

            float currHeight = MAX_WAVE_HEIGHT + sampleDelta.y;
            if (currHeight < sampleHeight) {
                waveWorldPos = samplePos;
                fuckO = (fi * rayStepLength + -0.5 * MAX_WAVE_HEIGHT) * PARALLEX_STRENGTH;
                break;
            }
        }
        #endif

        float NDotV = dot(geomViewNormal, viewDir);
        float weightHeightMul = 1.0;
        vec3 dVCdx = dFdx(viewPos);
        vec3 dVCdy = dFdy(viewPos);
        vec3 maxVec = max(abs(dVCdx), abs(dVCdy));
        float maxLen = length(maxVec);
        weightHeightMul *= saturate(rcp(maxLen) * 2.0);

        const float NORMAL_EPS = 0.05;
        const float NORMAL_WEIGHT = SETTING_WATER_NORMAL_SCALE;
        float waveHeightC = waveHeight(waveWorldPos, true) * weightHeightMul;
        float waveHeightX = waveHeight(waveWorldPos + vec3(NORMAL_EPS * WAVE_POS_BASE, 0.0, 0.0), true) * weightHeightMul;
        float waveHeightZ = waveHeight(waveWorldPos + vec3(0.0, 0.0, NORMAL_EPS * WAVE_POS_BASE), true) * weightHeightMul;
        vec3 waveNormal = vec3(
            waveHeightX,
            waveHeightZ,
            NORMAL_EPS
        );
        waveNormal.xy -= waveHeightC;
        waveNormal.xy *= NORMAL_WEIGHT;
        tangentNormal = waveNormal;
    } else {
        tangentNormal.xy = normalSample.rg * 2.0 - 1.0;
        tangentNormal.z = sqrt(saturate(1.0 - dot(tangentNormal.xy, tangentNormal.xy)));
        tangentNormal.xy *= exp2(SETTING_NORMAL_MAPPING_STRENGTH);
    }

    tangentNormal = normalize(tangentNormal);

    gData.normal = normalize(tbn * tangentNormal);
    #endif

    gData.lmCoord = frag_lmCoord;
    gData.materialID = frag_materialID;

    gData.lmCoord = dither_u8(gData.lmCoord, noiseIGN);

    return gData;
}

void main() {
    texelPos = ivec2(gl_FragCoord.xy);

    vec2 screenPos = gl_FragCoord.xy * global_mainImageSizeRcp;
    viewPos = coords_toViewCoord(screenPos, frag_viewZ, global_camProjInverse);

    float solidViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    if (solidViewZ > frag_viewZ) {
        discard;
    }

    gl_FragDepth = 0.0;
    vec4 inputAlbedo = processAlbedo();

    if (inputAlbedo.a < alphaTestRef) {
        discard;
    }

    lighting_gData = processOutput();

    bool isWater = frag_materialID == 3u;

    lighting_init(viewPos, texelPos);

    float alpha = inputAlbedo.a;
    vec3 materialColor = colors2_material_idt(inputAlbedo.rgb);

    vec3 t = materialColor;
    float lumaT = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, t);
    t *= saturate(0.3 / lumaT);

    t = pow(t, vec3(1.0 / 2.2));
    lumaT = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, t);
    float sat = isWater ? 0.3 : 1.0;
    t = lumaT + sat * (t - lumaT);
    t = pow(t, vec3(2.2));
    if (isWater) {
        t.g *= 1.9;
    }

    float tv = isWater ? 0.1 : 1.0;

    vec3 tAbsorption = -log(t) * (alpha * sqrt(alpha)) * tv;
    tAbsorption = max(tAbsorption, 0.0);
    vec3 tTransmittance = exp(-tAbsorption);
    lighting_gData.albedo = tTransmittance;

    rt_translucentColor = vec4(tTransmittance, 0.0);

    ivec2 farDepthTexelPos = texelPos;
    ivec2 nearDepthTexelPos = texelPos;
//    if (isWater) {
//        nearDepthTexelPos.x += global_mainImageSizeI.x;
//    } else {
        farDepthTexelPos.y += global_mainImageSizeI.y;
        nearDepthTexelPos += global_mainImageSizeI;
//    }

    float offsetViewZ = frag_viewZ;
    offsetViewZ -= fuckO;
    int cDepth = floatBitsToInt(-offsetViewZ);
    imageAtomicMax(uimg_translucentDepthLayers, farDepthTexelPos, cDepth);
    imageAtomicMin(uimg_translucentDepthLayers, nearDepthTexelPos, cDepth);

    gbufferData1_pack(rt_gbufferData1, lighting_gData);
    gbufferData2_pack(rt_gbufferData2, lighting_gData);
}