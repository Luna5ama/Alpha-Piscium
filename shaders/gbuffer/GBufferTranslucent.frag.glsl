ivec2 texelPos;
#include "/util/Dither.glsl"
#include "/general/Lighting.glsl"

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferViewZ;

in vec3 frag_viewTangent;

in vec4 frag_colorMul;
in vec3 frag_viewNormal;
in vec2 frag_texCoord;
in vec2 frag_lmCoord;
flat in uint frag_materialID;

in vec3 frag_viewCoord;

layout(early_fragment_tests) in;

#ifdef GBUFFER_PASS_DH
/* RENDERTARGETS:5 */
layout(location = 0) out vec4 rt_translucentColor;
#else
/* RENDERTARGETS:11 */
layout(location = 0) out vec4 rt_translucentColor;
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
    vec4 albedo = processAlbedo();
    lighting_gData = processOutput();

    lighting_init(frag_viewCoord, texelPos);

    rt_translucentColor.rgb = colors_srgbToLinear(albedo.rgb);
    rt_translucentColor.a = albedo.a;
}