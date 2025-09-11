ivec2 texelPos;
#include "/util/Colors2.glsl"
#include "/util/Dither.glsl"
#include "/techniques/Lighting.glsl"

layout(r32i) uniform iimage2D uimg_translucentDepthLayers;

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

in vec3 frag_viewTangent;

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
/* RENDERTARGETS:11,12 */
layout(location = 0) out vec4 rt_translucentColor;
layout(location = 1) out vec4 rt_translucentData;
#endif

vec4 processAlbedo() {
    vec4 albedo = frag_colorMul;
    albedo *= texture(gtexture, frag_texCoord);
    #ifdef SETTING_DEBUG_WHITE_WORLD
    return vec4(1.0);
    #else
    return albedo;
    #endif
}

GBufferData processOutput() {
    GBufferData gData = gbufferData_init();

    float noiseIGN = rand_IGN(texelPos, frameCounter);

    #ifdef DISTANT_HORIZONS
    #ifndef GBUFFER_PASS_DH
    float edgeFactor = linearStep(min(far * 0.75, far - 24.0), far, length(frag_viewCoord));
    if (noiseIGN < edgeFactor) {
        discard;
    }
    #endif
    #endif

    vec4 normalSample = textureLod(normals, frag_texCoord, 0.0);
    vec4 specularSample = textureLod(specular, frag_texCoord, 0.0);

    gData.pbrSpecular = specularSample;
    gData.lmCoord.y *= normalSample.b;

    const float _1o255 = 1.0 / 255.0;
    float emissiveS = linearStep(_1o255, 1.0, specularSample.a);
    emissiveS *= float(specularSample.a < 1.0);

    gData.pbrSpecular.a = emissiveS;

    vec3 bitangent = cross(frag_viewTangent, frag_viewNormal);
    mat3 tbn = mat3(frag_viewTangent, bitangent, frag_viewNormal);

    #ifndef SETTING_NORMAL_MAPPING
    gData.normal = frag_viewNormal;
    #else
    vec3 tagentNormal;
    tagentNormal.xy = normalSample.rg * 2.0 - 1.0;
    tagentNormal.z = sqrt(saturate(1.0 - dot(tagentNormal.xy, tagentNormal.xy)));
    vec3 mappedNormal = normalize(tbn * tagentNormal);
    gData.normal = normalize(mix(frag_viewNormal, mappedNormal, SETTING_NORMAL_MAPPING_STRENGTH));
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
    lighting_gData = processOutput();

    bool isWater = frag_materialID == 3u;

    lighting_init(frag_viewCoord, texelPos);

    inputAlbedo.rgb = colors2_eotf(COLORS2_MATERIAL_TF, inputAlbedo.rgb);
    float alpha = inputAlbedo.a;
    vec3 materialColor = colors2_colorspaces_convert(COLORS2_MATERIAL_COLORSPACE, COLORS2_WORKING_COLORSPACE, inputAlbedo.rgb);

    rt_translucentColor = vec4(materialColor * pow2(inputAlbedo.a), 1.0);

    float luma = saturate(colors2_colorspaces_luma(COLORS2_MATERIAL_COLORSPACE, inputAlbedo.rgb));

    vec3 t = normalize(materialColor);
    float lumaT = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, t);
    float sat = isWater ? 0.3 : 0.6;
    t = lumaT + sat * (t - lumaT);

    float tv = isWater ? 0.1 : 0.5;

    vec3 tAbsorption = -log(t) * pow2(alpha) / sqrt(luma) * tv;
    tAbsorption = max(tAbsorption, 0.0);
    vec3 tTransmittance = exp(-tAbsorption);

    rt_translucentData = vec4(tTransmittance, float(alpha > 0.0));

    ivec2 farDepthTexelPos = texelPos;
    ivec2 nearDepthTexelPos = texelPos;
    if (isWater) {
        nearDepthTexelPos.x += global_mainImageSizeI.x;
    } else {
        farDepthTexelPos.y += global_mainImageSizeI.y;
        nearDepthTexelPos += global_mainImageSizeI;
    }

    int cDepth = floatBitsToInt(-frag_viewCoord.z);
    imageAtomicMax(uimg_translucentDepthLayers, farDepthTexelPos, cDepth);
    imageAtomicMin(uimg_translucentDepthLayers, nearDepthTexelPos, cDepth);
}