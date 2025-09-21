ivec2 texelPos;
#include "/util/Colors2.glsl"
#include "/util/Dither.glsl"
#include "/techniques/Lighting.glsl"

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

in vec3 frag_viewCoord;

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
    vec4 albedo = vec4(1.0);
    #ifdef SETTING_SCREENSHOT_MODE
    albedo *= textureLod(gtexture, frag_texCoord, 0.0);
    #else
    albedo *= texture(gtexture, frag_texCoord);
    #endif
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
    tangentNormal.xy = normalSample.rg * 2.0 - 1.0;
    tangentNormal.z = sqrt(saturate(1.0 - dot(tangentNormal.xy, tangentNormal.xy)));
    tangentNormal.xy *= exp2(SETTING_NORMAL_MAPPING_STRENGTH);
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
    float solidViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    if (solidViewZ > frag_viewCoord.z) {
        discard;
    }

    gl_FragDepth = 0.0;
    vec4 inputAlbedo = processAlbedo();

    if (inputAlbedo.a < alphaTestRef) {
        discard;
    }

    lighting_gData = processOutput();

    bool isWater = frag_materialID == 3u;

    lighting_init(frag_viewCoord, texelPos);

    float alpha = inputAlbedo.a;
    vec3 materialColor = colors2_material_idt(inputAlbedo.rgb);

    vec3 t = materialColor;
    float lumaT = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, t);
    t *= saturate(0.3 / lumaT);

    t = pow(t, vec3(1.0 / 2.2));
    lumaT = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, t);
    float sat = isWater ? 0.9 : 1.0;
    t = lumaT + sat * (t - lumaT);
    t = pow(t, vec3(2.2));

    float tv = isWater ? 1.0 : 1.0;

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

    int cDepth = floatBitsToInt(-frag_viewCoord.z);
    imageAtomicMax(uimg_translucentDepthLayers, farDepthTexelPos, cDepth);
    imageAtomicMin(uimg_translucentDepthLayers, nearDepthTexelPos, cDepth);

    gbufferData1_pack(rt_gbufferData1, lighting_gData);
    gbufferData2_pack(rt_gbufferData2, lighting_gData);
}